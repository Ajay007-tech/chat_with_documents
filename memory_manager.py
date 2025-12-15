"""
Advanced memory management for preventing GPU memory leaks
"""

import torch
import gc
import os
import psutil
from typing import Optional, Dict, Any
from qwen_agent.log import logger
import weakref


class MemoryManager:
    """Manages GPU and CPU memory to prevent memory leaks"""
    
    def __init__(self):
        self.cache_references = weakref.WeakSet()
        self.initial_memory = None
        self.max_cache_size_gb = 2.0  # Maximum cache size before forcing cleanup
        
        if torch.cuda.is_available():
            # Set memory fraction to prevent using all GPU memory
            torch.cuda.set_per_process_memory_fraction(0.95)
            self.initial_memory = torch.cuda.memory_allocated()
            
    def cleanup_gpu_cache(self, force: bool = False):
        """Clean up GPU cache and free memory"""
        if not torch.cuda.is_available():
            return
            
        try:
            # Get current memory usage
            allocated_before = torch.cuda.memory_allocated() / 1024**3
            cached_before = torch.cuda.memory_reserved() / 1024**3
            
            # Clear cache
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # Force cleanup if memory usage is high
            if force or cached_before > self.max_cache_size_gb:
                # Clear all cached allocations
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()
                
                # Run garbage collection multiple times
                for _ in range(3):
                    gc.collect()
                
                torch.cuda.empty_cache()
                
            allocated_after = torch.cuda.memory_allocated() / 1024**3
            cached_after = torch.cuda.memory_reserved() / 1024**3
            
            if allocated_before - allocated_after > 0.1:  # If freed more than 100MB
                logger.info(f"Freed GPU memory: {allocated_before:.2f}GB -> {allocated_after:.2f}GB")
                logger.info(f"Cache: {cached_before:.2f}GB -> {cached_after:.2f}GB")
                
        except Exception as e:
            logger.warning(f"Error during GPU cleanup: {e}")
    
    def cleanup_tensors(self):
        """Clean up any lingering tensors"""
        if not torch.cuda.is_available():
            return
            
        try:
            # Find and delete CUDA tensors
            for obj in gc.get_objects():
                try:
                    if torch.is_tensor(obj) and obj.is_cuda:
                        del obj
                except:
                    pass
                    
            gc.collect()
            torch.cuda.empty_cache()
            
        except Exception as e:
            logger.warning(f"Error cleaning tensors: {e}")
    
    def reset_memory_stats(self):
        """Reset memory statistics"""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.reset_accumulated_memory_stats()
    
    def get_memory_info(self) -> Dict[str, float]:
        """Get current memory information"""
        info = {}
        
        if torch.cuda.is_available():
            info['gpu_allocated_gb'] = torch.cuda.memory_allocated() / 1024**3
            info['gpu_reserved_gb'] = torch.cuda.memory_reserved() / 1024**3
            info['gpu_free_gb'] = (torch.cuda.get_device_properties(0).total_memory - 
                                  torch.cuda.memory_allocated()) / 1024**3
        
        # CPU memory
        process = psutil.Process()
        info['cpu_memory_gb'] = process.memory_info().rss / 1024**3
        info['cpu_percent'] = psutil.cpu_percent(interval=0.1)
        
        return info
    
    def check_memory_pressure(self) -> bool:
        """Check if system is under memory pressure"""
        if not torch.cuda.is_available():
            return False
            
        # Check if we're using more than 90% of GPU memory
        total_memory = torch.cuda.get_device_properties(0).total_memory
        used_memory = torch.cuda.memory_allocated()
        
        return (used_memory / total_memory) > 0.9
    
    def emergency_cleanup(self):
        """Emergency cleanup when memory is critically low"""
        logger.warning("Performing emergency memory cleanup!")
        
        # Force full cleanup
        self.cleanup_tensors()
        self.cleanup_gpu_cache(force=True)
        
        # Clear Python caches
        gc.collect(2)  # Full collection
        
        # If still high, try to free more
        if self.check_memory_pressure():
            logger.warning("Memory still high after cleanup, forcing aggressive cleanup")
            
            # Clear all matplotlib figures if any
            try:
                import matplotlib.pyplot as plt
                plt.close('all')
            except:
                pass
            
            # Clear any dataframes
            try:
                import pandas as pd
                for obj in gc.get_objects():
                    if isinstance(obj, pd.DataFrame):
                        del obj
            except:
                pass
            
            # Final aggressive cleanup
            for _ in range(5):
                gc.collect()
            torch.cuda.empty_cache()


class GenerationMemoryManager:
    """Manages memory during text generation"""
    
    def __init__(self, model):
        self.model = model
        self.memory_manager = MemoryManager()
        self.generation_count = 0
        self.cleanup_frequency = 2  # Cleanup every N generations
        
    def pre_generation(self):
        """Called before generation starts"""
        self.generation_count += 1
        
        # Regular cleanup
        if self.generation_count % self.cleanup_frequency == 0:
            self.memory_manager.cleanup_gpu_cache()
        
        # Check memory pressure
        if self.memory_manager.check_memory_pressure():
            logger.warning("High memory pressure detected before generation")
            self.memory_manager.emergency_cleanup()
            
    def post_generation(self):
        """Called after generation completes"""
        # Clear any generation caches
        if hasattr(self.model, 'model'):
            model = self.model.model
            
            # Clear KV cache if exists
            if hasattr(model, 'clear_cache'):
                model.clear_cache()
            
            # Clear attention caches
            for module in model.modules():
                if hasattr(module, 'attention_cache'):
                    module.attention_cache = None
                if hasattr(module, 'kv_cache'):
                    module.kv_cache = None
        
        # Basic cleanup
        self.memory_manager.cleanup_gpu_cache()
        
        # Log memory status
        info = self.memory_manager.get_memory_info()
        if 'gpu_allocated_gb' in info:
            logger.debug(f"Post-generation GPU memory: {info['gpu_allocated_gb']:.2f}GB allocated, "
                        f"{info['gpu_free_gb']:.2f}GB free")
    
    def reset_cache_states(self):
        """Reset all cache states in the model"""
        if hasattr(self.model, 'model'):
            model = self.model.model
            
            # Reset transformer caches
            if hasattr(model, 'transformer'):
                for layer in model.transformer.h:
                    if hasattr(layer, 'attn'):
                        if hasattr(layer.attn, 'cache_k'):
                            layer.attn.cache_k = None
                        if hasattr(layer.attn, 'cache_v'):
                            layer.attn.cache_v = None


# Global memory manager instance
_global_memory_manager = None

def get_memory_manager() -> MemoryManager:
    """Get global memory manager instance"""
    global _global_memory_manager
    if _global_memory_manager is None:
        _global_memory_manager = MemoryManager()
    return _global_memory_manager

def cleanup_memory():
    """Quick cleanup function"""
    manager = get_memory_manager()
    manager.cleanup_gpu_cache()

def emergency_cleanup():
    """Emergency cleanup function"""
    manager = get_memory_manager()
    manager.emergency_cleanup()