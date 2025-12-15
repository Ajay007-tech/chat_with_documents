import os
from typing import List, Tuple, Union, Optional
from web_ui import WebUI
import math
import torch
import gc
import psutil

# Set dummy DashScope API key to bypass the check (we don't actually use it)
os.environ['DASHSCOPE_API_KEY'] = 'dummy_key_for_local_use_only'

from qwen_agent.agents import Assistant
from qwen_agent.tools.base import register_tool
from qwen_agent.tools.doc_parser import Record
from qwen_agent.tools.search_tools.base_search import RefMaterialOutput, BaseSearch
from qwen_agent.log import logger
from qwen_agent.gui.gradio import gr

# Import vector database and custom assistant
from vector_db import VectorDatabase, RAGSearch
from custom_assistant import VectorDBAssistant

POSITIVE_INFINITY = math.inf

# Update this path to your local model path
LOCAL_MODEL_PATH = r"E:\ds-team\huggingface_models\test_models\models--Qwen--Qwen2.5-7B-Instruct-1M\snapshots\e28526f7bb80e2a9c8af03b831a9af3812f18fba"

# Import local_llm to register the model type
import local_llm

# Import memory manager
from memory_manager import get_memory_manager, cleanup_memory

@register_tool('_vector_search')
class VectorSearch(BaseSearch):
    """Vector database search for long documents"""
    
    def __init__(self, cfg: Optional[dict] = None):
        # Initialize with minimal config to avoid DashScope
        cfg = cfg or {}
        cfg['rag_searchers'] = []  # Disable default searchers
        super().__init__(cfg)
        
        # Initialize vector database only if available
        try:
            self.vector_db = VectorDatabase(
                persist_directory="./vector_db_storage",
                embedding_model="all-MiniLM-L6-v2",  # Lightweight model
                use_cpu=True  # Use CPU for embeddings to save GPU
            )
            self.rag_search = RAGSearch(self.vector_db)
            self.enabled = self.vector_db.enabled
        except Exception as e:
            logger.warning(f"Vector database initialization failed: {e}")
            self.vector_db = None
            self.rag_search = None
            self.enabled = False
        
    def call(self, params: Union[str, dict], docs: List[Union[Record, str, List[str]]] = None, **kwargs) -> list:
        """Process documents with vector database"""
        if not self.enabled:
            logger.warning("Vector database not enabled, falling back to standard search")
            return super().call(params, docs, **kwargs)
        
        params = self._verify_json_format_args(params)
        query = params.get('query', '')
        
        if not docs:
            return []
        
        # Add documents to vector database
        logger.info(f"Adding {len(docs)} documents to vector database...")
        success = self.rag_search.add_documents(docs)
        
        if not success:
            logger.warning("Failed to add documents to vector DB, using fallback")
            # Fallback to standard search
            return super().call(params, docs, **kwargs)
        
        # Get relevant chunks based on query
        if query:
            logger.info(f"Searching for relevant chunks for query: {query[:100]}...")
            relevant_chunks = self.rag_search.search_relevant_chunks(query, max_chunks=5)
            
            # Format as expected by qwen-agent
            if relevant_chunks:
                return [{
                    'url': 'vector_search',
                    'text': relevant_chunks
                }]
        
        # If no query, return summary of what's in the database
        doc_count = self.vector_db.get_document_count()
        return [{
            'url': 'vector_search',
            'text': [f"Vector database contains {doc_count} document chunks. Ask questions to search."]
        }]

    def sort_by_scores(self, query: str, docs: List[Record], max_ref_token: int, **kwargs):
        raise NotImplementedError

@register_tool('no_search')
class NoSearch(BaseSearch):
    """Traditional search without vector database"""
    
    def call(self, params: Union[str, dict], docs: List[Union[Record, str, List[str]]] = None, **kwargs) -> list:
        params = self._verify_json_format_args(params)
        max_ref_token = kwargs.get('max_ref_token', self.max_ref_token)

        if not docs:
            return []
        return self._get_the_front_part(docs, max_ref_token)

    @staticmethod
    def _get_the_front_part(docs: List[Record], max_ref_token: int) -> list:
        all_tokens = 0
        _ref_list = []
        
        for doc in docs:
            text = []
            doc_tokens = 0
            
            for page in doc.raw:
                if all_tokens + page.token > max_ref_token:
                    if text:
                        logger.warning(f"Document truncated at {all_tokens} tokens (limit: {max_ref_token})")
                        now_ref_list = RefMaterialOutput(url=doc.url, text=text).to_dict()
                        _ref_list.append(now_ref_list)
                    break
                    
                text.append(page.content)
                all_tokens += page.token
                doc_tokens += page.token
            else:
                if text:
                    now_ref_list = RefMaterialOutput(url=doc.url, text=text).to_dict()
                    _ref_list.append(now_ref_list)
            
            logger.info(f"Document {doc.url}: {doc_tokens} tokens")

        logger.info(f'Total tokens used: {all_tokens} (limit: {max_ref_token})')
        
        if all_tokens > max_ref_token:
            logger.warning(f"Documents were truncated to fit within {max_ref_token} token limit")
        
        return _ref_list

    def sort_by_scores(self, query: str, docs: List[Record], max_ref_token: int, **kwargs):
        raise NotImplementedError

class ModelManager:
    """Manage model loading and switching with lazy loading"""
    
    def __init__(self, model_path):
        self.model_path = model_path
        self.current_bot = None
        self.current_quantization = "4bit"
        self.current_llm_instance = None  # Store LLM instance for cleanup
        
        # Define context limits based on quantization
        self.context_limits = {
            "4bit": {
                "max_context_tokens": 32768,
                "max_ref_token": 30000,
                "max_new_tokens": 8000,
                "use_vector_db": True,  # Enable vector DB for 4-bit
                "description": "4-bit: Vector DB enabled for unlimited document size"
            },
            "8bit": {
                "max_context_tokens": 16384,
                "max_ref_token": 14000,
                "max_new_tokens": 1536,
                "use_vector_db": False,
                "description": "8-bit: Handles up to 16k tokens, balanced quality"
            },
            "full": {
                "max_context_tokens": 8192,
                "max_ref_token": 6000,
                "max_new_tokens": 1024,
                "use_vector_db": False,
                "description": "Full precision: Handles up to 8k tokens, best quality"
            }
        }
    
    def unload_current_model(self):
        """Completely unload current model from memory with aggressive cleanup"""
        if self.current_bot is not None:
            logger.info("Unloading current model...")
            
            # Delete the LLM instance
            if hasattr(self.current_bot, 'llm'):
                if hasattr(self.current_bot.llm, 'model'):
                    # Clear model caches first
                    if hasattr(self.current_bot.llm, '_clear_model_cache'):
                        self.current_bot.llm._clear_model_cache()
                    # Delete the actual model
                    del self.current_bot.llm.model
                # Delete the LLM wrapper
                del self.current_bot.llm
            
            # Delete the bot
            del self.current_bot
            self.current_bot = None
            self.current_llm_instance = None
            
            # Aggressive memory cleanup
            for _ in range(3):
                gc.collect()
            
            # Clear CUDA cache multiple times
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            
            # Use memory manager for thorough cleanup
            cleanup_memory()
            
            # Log memory freed
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                logger.info(f"Model unloaded. GPU memory now: {allocated:.2f} GB")
            
            # Give system time to free memory
            import time
            time.sleep(1)
    
    def load_model(self, quantization="4bit"):
        """Load model with specified quantization (lazy loading)"""
        # First unload any existing model
        self.unload_current_model()
        
        # Get configuration for this quantization level
        
        
        
        config = self.context_limits[quantization]
        
        logger.info(f"Loading model with {quantization} quantization...")
        logger.info(f"Configuration: {config['description']}")
        
        # Log system resources before loading
        if torch.cuda.is_available():
            allocated_before = torch.cuda.memory_allocated() / 1024**3
            total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"GPU memory before loading: {allocated_before:.2f}/{total_vram:.2f} GB")
        
        # Create LLM configuration
        llm_config = {
            "model_type": "qwen_local",
            "local_path": self.model_path,
            "load_in_4bit": quantization == "4bit",
            "load_in_8bit": quantization == "8bit",
            "max_context_tokens": config["max_context_tokens"],
            "generate_cfg": {
                "max_input_tokens": config["max_context_tokens"],
                "max_new_tokens": config["max_new_tokens"],
                "temperature": 0.7,
                "top_p": 0.9,
                "repetition_penalty": 1.05,
            }
        }
        
        name_suffix = {
            "4bit": "4-bit (Vector DB + Long Context)",
            "8bit": "8-bit (Balanced)", 
            "full": "Full Precision (Best Quality)"
        }[quantization]
        
        # Use custom assistant for vector DB mode, standard for others
        if config['use_vector_db']:
            # Use custom assistant with vector database
            self.current_bot = VectorDBAssistant(
                llm=llm_config,
                name=f"",
                description=config['description'],
                use_vector_db=True,
                max_ref_token=config['max_ref_token']
            )
        else:
            # Standard assistant with traditional search
            self.current_bot = Assistant(
                llm=llm_config,
                name=f"",
                description=config['description'],
                rag_cfg={
                    'max_ref_token': config['max_ref_token'],
                    'rag_searchers': ['no_search']
                },
            )
        
        self.current_quantization = quantization
        logger.info(f"Successfully loaded model with {quantization} quantization")
        
        # Log memory usage after loading
        if torch.cuda.is_available():
            allocated_after = torch.cuda.memory_allocated() / 1024**3
            logger.info(f"GPU Memory after loading: {allocated_after:.2f} GB (used {allocated_after - allocated_before:.2f} GB)")
        
        # Log CPU memory
        process = psutil.Process()
        cpu_memory = process.memory_info().rss / 1024**3
        logger.info(f"CPU Memory usage: {cpu_memory:.2f} GB")
        
        return self.current_bot
    
    def get_current_limits(self):
        """Get the current context limits"""
        if self.current_quantization:
            return self.context_limits[self.current_quantization]
        return self.context_limits["4bit"]
    
    def is_model_loaded(self):
        """Check if a model is currently loaded"""
        return self.current_bot is not None

# Global model manager
model_manager = None

def app_gui():
    global model_manager
    
    # Set CUDA memory management
    if torch.cuda.is_available():
        # Enable memory efficient settings
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        
        # Log GPU info
        gpu_name = torch.cuda.get_device_name(0)
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"GPU detected: {gpu_name} with {total_vram:.2f} GB VRAM")
        
        # Clear any existing cache
        torch.cuda.empty_cache()
    
    # Initialize model manager (without loading any model)
    model_manager = ModelManager(LOCAL_MODEL_PATH)
    
    # Don't load any model initially - let user choose
    logger.info("Model manager initialized. No model loaded yet.")
    logger.info("User will select and load model through the interface.")
    
    bot = model_manager.load_model("4bit")

    chatbot_config = {
        'input.placeholder': '''Ask anything or use "/clear" to clean the screen''',
        'verbose': True,
        'quantization_selector': False,
        'lazy_loading': True,  # Enable lazy loading
    }
    
    # Create a placeholder assistant (won't be used until model is loaded)
    placeholder_bot = None
    
    # Pass the model manager to WebUI
    web_ui = WebUI(bot, chatbot_config=chatbot_config, model_manager=model_manager)
    web_ui.run(share=True, server_name="127.0.0.1", server_port=1585, reload = True)

if __name__ == '__main__':
    import patching  # patch qwen-agent to accelerate processing
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        logger.info(f"CUDA is available. GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        logger.warning("CUDA is not available. Running on CPU (will be slower)")
    
    # Log system memory
    memory = psutil.virtual_memory()
    logger.info(f"System RAM: {memory.total / 1024**3:.2f} GB (Available: {memory.available / 1024**3:.2f} GB)")
    
    try:
        app_gui()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        if model_manager:
            model_manager.unload_current_model()
        torch.cuda.empty_cache()
        gc.collect()
