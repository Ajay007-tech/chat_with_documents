"""
Local LLM handler for Qwen models with proper qwen_agent integration and memory management
"""

import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from typing import List, Iterator, Optional, Dict, Any, Union
from http import HTTPStatus
import time
import re
from threading import Thread
from qwen_agent.log import logger

# Import the necessary qwen_agent components
from qwen_agent.llm.base import register_llm, BaseChatModel
from qwen_agent.llm.schema import Message, ASSISTANT, USER, SYSTEM, CONTENT, DEFAULT_SYSTEM_MESSAGE
from qwen_agent.utils.utils import print_traceback

@register_llm("qwen_local")
class QwenLocalModel(BaseChatModel):
    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        super().__init__(cfg)
        self.model_path = cfg.get("local_path")
        self.load_in_4bit = cfg.get("load_in_4bit", False)
        self.load_in_8bit = cfg.get("load_in_8bit", False)
        self.model = None
        self.tokenizer = None
        
        # Set max context based on quantization
        if self.load_in_4bit:
            self.max_context_tokens = cfg.get("max_context_tokens", 32768)  # 32k for 4-bit
        elif self.load_in_8bit:
            self.max_context_tokens = cfg.get("max_context_tokens", 16384)  # 16k for 8-bit
        else:
            self.max_context_tokens = cfg.get("max_context_tokens", 8192)   # 8k for full precision
            
        logger.info(f"Max context tokens set to: {self.max_context_tokens}")
        self._load_model()

    def _load_model(self):
        """Load model with specified quantization"""
        if self.model is not None:
            # Clear existing model
            del self.model
            torch.cuda.empty_cache()
            gc.collect()
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=True)
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        if self.load_in_4bit:
            from transformers import BitsAndBytesConfig
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map="auto",
                quantization_config=quant_config,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )
            logger.info("Loaded model in 4-bit quantization")
        elif self.load_in_8bit:
            from transformers import BitsAndBytesConfig
            quant_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.float16
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map="auto",
                quantization_config=quant_config,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )
            logger.info("Loaded model in 8-bit quantization")
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map="auto",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            logger.info("Loaded model in full precision (16-bit)")
        
        self.model.eval()
        
        # Clear cache after loading
        torch.cuda.empty_cache()

    def _convert_messages_to_list(self, messages: List[Message]) -> List[Dict[str, str]]:
        """Convert Message objects to dict format for tokenizer"""
        msg_list = []
        for msg in messages:
            if hasattr(msg, 'role') and hasattr(msg, 'content'):
                content = msg.content
                if isinstance(content, list) and len(content) > 0:
                    # Extract text from content list
                    text = ""
                    for item in content:
                        if isinstance(item, dict) and 'text' in item:
                            text += item['text']
                        elif isinstance(item, str):
                            text += item
                    content = text
                elif isinstance(content, str):
                    pass
                else:
                    content = str(content)
                
                msg_list.append({
                    "role": msg.role,
                    "content": content
                })
            elif isinstance(msg, dict):
                msg_list.append(msg)
        return msg_list

    def _truncate_messages(self, messages: List[Dict[str, str]], max_tokens: int) -> List[Dict[str, str]]:
        """Truncate messages to fit within max_tokens"""
        # Apply chat template to get token count
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        tokens = self.tokenizer.encode(text)
        current_length = len(tokens)
        
        if current_length <= max_tokens:
            return messages
        
        logger.warning(f"Context length ({current_length}) exceeds maximum ({max_tokens}). Truncating...")
        
        # Keep system message and last user message, truncate middle content
        truncated_messages = []
        
        # Always keep system message if present
        if messages and messages[0]['role'] == 'system':
            truncated_messages.append(messages[0])
            remaining_messages = messages[1:]
        else:
            remaining_messages = messages
        
        # Keep the last user message (most important)
        if remaining_messages and remaining_messages[-1]['role'] == 'user':
            last_user_msg = remaining_messages[-1].copy()
            
            # Truncate the content of the last user message if it's too long
            if len(last_user_msg['content']) > max_tokens * 3:  # Rough estimate
                last_user_msg['content'] = last_user_msg['content'][:max_tokens * 3] + "\n\n[Content truncated due to length...]"
            
            truncated_messages.append(last_user_msg)
        
        return truncated_messages

    def _chat_stream(
        self,
        messages: List[Message],
        delta_stream: bool = False,
        generate_cfg: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Iterator[List[Message]]:
        """Stream chat responses with memory management"""
        generate_cfg = generate_cfg or {}
        
        try:
            # Convert messages to proper format
            msg_list = self._convert_messages_to_list(messages)
            
            # Reserve tokens for generation
            reserved_tokens = generate_cfg.get("max_new_tokens", 512)
            available_context = self.max_context_tokens - reserved_tokens
            
            # Truncate if necessary
            msg_list = self._truncate_messages(msg_list, available_context)
            
            # Apply chat template
            text = self.tokenizer.apply_chat_template(
                msg_list,
                tokenize=False,
                add_generation_prompt=True
            )
            
            inputs = self.tokenizer(
                text, 
                return_tensors="pt",
                truncation=True,
                max_length=available_context
            ).to(self.model.device)
            
            logger.info(f"Input tokens: {inputs['input_ids'].shape[-1]}")
            
            # Set up streaming
            streamer = TextIteratorStreamer(
                self.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True
            )
            
            generation_kwargs = dict(
                **inputs,
                max_new_tokens=min(generate_cfg.get("max_new_tokens", 512), 2048),  # Cap at 2048
                temperature=generate_cfg.get("temperature", 0.7),
                top_p=generate_cfg.get("top_p", 0.9),
                do_sample=generate_cfg.get("temperature", 0.7) > 0,
                repetition_penalty=generate_cfg.get("repetition_penalty", 1.05),
                streamer=streamer,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
            # Start generation in a separate thread
            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()
            
            # Stream the output
            generated_text = ""
            for new_text in streamer:
                generated_text += new_text
                if delta_stream:
                    yield [Message(role=ASSISTANT, content=new_text)]
                else:
                    yield [Message(role=ASSISTANT, content=generated_text)]
            
            thread.join()
            
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"CUDA OOM: {e}")
            torch.cuda.empty_cache()
            gc.collect()
            error_msg = (
                "⚠️ GPU memory exceeded. The document is too long for the current settings.\n\n"
                "Please try:\n"
                "1. Using 4-bit quantization for longer documents\n"
                "2. Uploading smaller documents\n"
                "3. Clearing the chat history with /clear"
            )
            yield [Message(role=ASSISTANT, content=error_msg)]
        except Exception as e:
            logger.error(f"Generation error: {e}")
            print_traceback()
            yield [Message(role=ASSISTANT, content=f"Error during generation: {str(e)}")]

    def _chat_no_stream(
        self,
        messages: List[Message],
        generate_cfg: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Message]:
        """Non-streaming chat with memory management"""
        generate_cfg = generate_cfg or {}
        
        try:
            # Convert messages to proper format
            msg_list = self._convert_messages_to_list(messages)
            
            # Reserve tokens for generation
            reserved_tokens = generate_cfg.get("max_new_tokens", 512)
            available_context = self.max_context_tokens - reserved_tokens
            
            # Truncate if necessary
            msg_list = self._truncate_messages(msg_list, available_context)
            
            # Apply chat template
            text = self.tokenizer.apply_chat_template(
                msg_list,
                tokenize=False,
                add_generation_prompt=True
            )
            
            inputs = self.tokenizer(
                text, 
                return_tensors="pt",
                truncation=True,
                max_length=available_context
            ).to(self.model.device)
            
            logger.info(f"Input tokens: {inputs['input_ids'].shape[-1]}")
            
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=min(generate_cfg.get("max_new_tokens", 512), 2048),  # Cap at 2048
                    temperature=generate_cfg.get("temperature", 0.7),
                    top_p=generate_cfg.get("top_p", 0.9),
                    do_sample=generate_cfg.get("temperature", 0.7) > 0,
                    repetition_penalty=generate_cfg.get("repetition_penalty", 1.05),
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode only the generated part
            generated_ids = output_ids[0][inputs['input_ids'].shape[-1]:]
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            # Clear cache after generation
            torch.cuda.empty_cache()
            
            return [Message(role=ASSISTANT, content=generated_text)]
            
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"CUDA OOM: {e}")
            torch.cuda.empty_cache()
            gc.collect()
            error_msg = (
                "⚠️ GPU memory exceeded. The document is too long for the current settings.\n\n"
                "Please try:\n"
                "1. Using 4-bit quantization for longer documents\n"
                "2. Uploading smaller documents\n"
                "3. Clearing the chat history with /clear"
            )
            return [Message(role=ASSISTANT, content=error_msg)]
        except Exception as e:
            logger.error(f"Generation error: {e}")
            print_traceback()
            return [Message(role=ASSISTANT, content=f"Error during generation: {str(e)}")]

    def _chat_with_functions(
        self,
        messages: List[Message],
        functions: List[Dict[str, Any]],
        stream: bool = False,
        delta_stream: bool = False,
        generate_cfg: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Union[List[Message], Iterator[List[Message]]]:
        """Handle function calling (fallback to regular chat for now)"""
        if stream:
            return self._chat_stream(messages, delta_stream=delta_stream, generate_cfg=generate_cfg, **kwargs)
        else:
            return self._chat_no_stream(messages, generate_cfg=generate_cfg, **kwargs)