"""
Patching module to optimize qwen-agent for long document processing with vector database support
"""

import os
import re
from typing import List, Union, Iterator
from http import HTTPStatus
from time import time
import json
import torch
import gc

# Set dummy DashScope API key to bypass validation (we use local models only)
os.environ['DASHSCOPE_API_KEY'] = 'dummy_key_for_local_use_only'

from qwen_agent.agents import Assistant
from qwen_agent.agents import assistant
from qwen_agent.agents.assistant import Assistant, get_basename_from_url
from qwen_agent.memory.memory import Memory
from qwen_agent.llm.schema import ASSISTANT, USER, Message, SYSTEM, CONTENT
from qwen_agent.llm.qwen_dashscope import QwenChatAtDS
import qwen_agent.llm.base
from qwen_agent.llm.base import ModelServiceError
from qwen_agent.utils.utils import extract_text_from_message, print_traceback
from qwen_agent.utils.tokenization_qwen import count_tokens, tokenizer
from qwen_agent.utils.utils import (get_file_type, hash_sha256, is_http_url,
                                    sanitize_chrome_file_path, save_url_to_local_work_dir)
from qwen_agent.log import logger
from qwen_agent.gui.gradio import gr
from qwen_agent.tools.storage import KeyNotExistsError
from qwen_agent.tools.simple_doc_parser import (SimpleDocParser, PARSER_SUPPORTED_FILE_TYPES, parse_pdf, 
                                    parse_word, parse_ppt, parse_txt, parse_html_bs, parse_csv,
                                    parse_tsv, parse_excel, get_plain_doc)

# Import vector database if available
try:
    from vector_db import VectorDatabase, RAGSearch
    VECTOR_DB_AVAILABLE = True
except ImportError:
    VECTOR_DB_AVAILABLE = False
    logger.info("Vector database not available. Using standard search.")

# ============= Memory Run Patch for Vector DB Support =============
def memory_run(self, messages: List[Message], lang: str = 'en', **kwargs) -> Iterator[List[Message]]:
    """Enhanced memory run with vector database support for long documents"""
    
    # Process files in messages
    rag_files = self.get_rag_files(messages)

    if not rag_files:
        yield [Message(role=ASSISTANT, content='', name='memory')]
    else:
        query = ''
        # Only retrieval content according to the last user query if exists
        if messages and messages[-1].role == USER:
            query = extract_text_from_message(messages[-1], add_upload_info=False)

        # Check if we should use vector search
        use_vector_search = kwargs.get('use_vector_search', False)
        
        if use_vector_search and VECTOR_DB_AVAILABLE:
            logger.info("Using vector database for document retrieval")
            # Vector search will be handled by the vector_search tool
            content = self.function_map['vector_search'].call(
                {
                    'query': query,
                    'files': rag_files
                },
                **kwargs,
            )
        else:
            # Standard retrieval
            content = self.function_map['retrieval'].call(
                {
                    'query': query,
                    'files': rag_files
                },
                **kwargs,
            )
        
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False, indent=4)

        yield [Message(role=ASSISTANT, content=content, name='memory')]

Memory._run = memory_run

# ============= Common Programming Language Extensions =============
common_programming_language_extensions = [
    "py",   # Python
    "java", # Java
    "cpp",  # C++
    "c",    # C
    "h",    # C/C++ header
    "hpp",  # C++ header
    "cs",   # C#
    "js",   # JavaScript
    "ts",   # TypeScript
    "jsx",  # React
    "tsx",  # TypeScript React
    "rb",   # Ruby
    "php",  # PHP
    "swift",# Swift
    "go",   # Go
    "rs",   # Rust
    "kt",   # Kotlin
    "scala",# Scala
    "m",    # Objective-C
    "css",  # CSS
    "scss", # SASS
    "sql",  # SQL
    "sh",   # Shell
    "bash", # Bash
    "pl",   # Perl
    "r",    # R
    "jl",   # Julia
    "dart", # Dart
    "lua",  # Lua
    "vim",  # Vim script
    "json", # JSON
    "xml",  # XML
    "yml",  # YAML
    "yaml", # YAML
    "toml", # TOML
    "ini",  # INI
    "cfg",  # Config
    "conf", # Config
    "md",   # Markdown
    "rst",  # reStructuredText
    "tex",  # LaTeX
]

# ============= SimpleDocParser Call Patch for Better Performance =============
def SimpleDocParser_call(self, params: Union[str, dict], **kwargs) -> Union[str, list]:
    params = self._verify_json_format_args(params)
    path = params['url']
    cached_name_ori = f'{hash_sha256(path)}_ori'
    
    try:
        # Try to load from cache first
        parsed_file = self.db.get(cached_name_ori)
        parsed_file = json.loads(parsed_file)
        logger.info(f'Read parsed {path} from cache.')
    except KeyNotExistsError:
        logger.info(f'Start parsing {path}...')
        time1 = time()

        f_type = get_file_type(path)
        if f_type in PARSER_SUPPORTED_FILE_TYPES + common_programming_language_extensions:
            if path.startswith('https://') or path.startswith('http://') or re.match(
                    r'^[A-Za-z]:\\', path) or re.match(r'^[A-Za-z]:/', path):
                path = path
            else:
                path = sanitize_chrome_file_path(path)

        os.makedirs(self.data_root, exist_ok=True)
        if is_http_url(path):
            # Download online url
            tmp_file_root = os.path.join(self.data_root, hash_sha256(path))
            os.makedirs(tmp_file_root, exist_ok=True)
            path = save_url_to_local_work_dir(path, tmp_file_root)

        # Parse based on file type
        if f_type == 'pdf':
            parsed_file = parse_pdf(path, self.extract_image)
        elif f_type == 'docx':
            parsed_file = parse_word(path, self.extract_image)
        elif f_type == 'pptx':
            parsed_file = parse_ppt(path, self.extract_image)
        elif f_type == 'txt' or f_type in common_programming_language_extensions:
            parsed_file = parse_txt(path)
        elif f_type == 'html':
            parsed_file = parse_html_bs(path, self.extract_image)
        elif f_type == 'csv':
            parsed_file = parse_csv(path, self.extract_image)
        elif f_type == 'tsv':
            parsed_file = parse_tsv(path, self.extract_image)
        elif f_type in ['xlsx', 'xls']:
            parsed_file = parse_excel(path, self.extract_image)
        else:
            raise ValueError(
                f'Failed: The current parser does not support this file type! Supported types: {"/".join(PARSER_SUPPORTED_FILE_TYPES + common_programming_language_extensions)}'
            )
        
        # Add token counts
        for page in parsed_file:
            for para in page['content']:
                para['token'] = count_tokens(para.get('text', para.get('table', '')))
        
        time2 = time()
        logger.info(f'Finished parsing {path}. Time spent: {time2 - time1:.2f} seconds.')
        
        # Cache the parsing doc
        self.db.put(cached_name_ori, json.dumps(parsed_file, ensure_ascii=False, indent=2))

    if not self.structured_doc:
        return get_plain_doc(parsed_file)
    else:
        return parsed_file

SimpleDocParser.call = SimpleDocParser_call

# ============= Message Truncation for Long Documents =============
def _truncate_input_messages_roughly(messages: List[Message], max_tokens: int) -> List[Message]:
    """Enhanced truncation with vector database awareness"""
    
    sys_msg = messages[0]
    assert sys_msg.role == SYSTEM
    
    if len([m for m in messages if m.role == SYSTEM]) >= 2:
        raise gr.Error(
            'The input messages must contain no more than one system message.'
        )

    turns = []
    for m in messages[1:]:
        if m.role == USER:
            turns.append([m])
        else:
            if turns:
                turns[-1].append(m)
            else:
                raise gr.Error(
                    'The input messages (excluding the system message) must start with a user message.'
                )

    def _count_tokens(msg: Message) -> int:
        try:
            text = extract_text_from_message(msg, add_upload_info=True)
            return tokenizer.count_tokens(text)
        except:
            # Fallback to character-based estimation
            text = extract_text_from_message(msg, add_upload_info=True)
            return len(text) // 4  # Rough estimate: 1 token ≈ 4 characters

    token_cnt = _count_tokens(sys_msg)
    truncated = []
    
    # Check if we're using vector search (indicated by large max_tokens)
    using_vector_search = max_tokens > 100000
    
    if using_vector_search:
        # With vector search, we don't need to include all history
        # Just include system message and last few turns
        max_turns_to_keep = 3
        recent_turns = turns[-max_turns_to_keep:] if len(turns) > max_turns_to_keep else turns
        
        for turn in recent_turns:
            cur_turn_msgs = []
            cur_token_cnt = 0
            for m in turn:
                cur_turn_msgs.append(m)
                cur_token_cnt += _count_tokens(m)
            truncated.extend(cur_turn_msgs)
            token_cnt += cur_token_cnt
    else:
        # Standard truncation for non-vector search
        for i, turn in enumerate(reversed(turns)):
            cur_turn_msgs = []
            cur_token_cnt = 0
            for m in reversed(turn):
                cur_turn_msgs.append(m)
                cur_token_cnt += _count_tokens(m)
            
            # Keep at least one user message
            if (i == 0) or (token_cnt + cur_token_cnt <= max_tokens):
                truncated.extend(cur_turn_msgs)
                token_cnt += cur_token_cnt
            else:
                break
    
    # Always include the system message
    truncated.append(sys_msg)
    truncated.reverse()

    if len(truncated) < 2:
        raise gr.Error(
            'At least one user message should be provided.'
        )
    
    # Only check token limit for non-vector search
    if not using_vector_search and token_cnt > max_tokens:
        logger.warning(f'Input tokens ({token_cnt}) exceed limit ({max_tokens}). Consider using vector search mode.')
        # Try to truncate content within messages
        for msg in truncated[1:]:  # Skip system message
            if hasattr(msg, 'content') and isinstance(msg.content, str) and len(msg.content) > 1000:
                msg.content = msg.content[:1000] + "\n\n[Content truncated due to length...]"
    
    return truncated

qwen_agent.llm.base._truncate_input_messages_roughly = _truncate_input_messages_roughly

# ============= Format Knowledge for Vector Search Results =============
def format_knowledge_to_source_and_content(result: Union[str, List[dict]]) -> List[dict]:
    """Enhanced formatting with vector search support"""
    knowledge = []
    
    if isinstance(result, str):
        result = f'{result}'.strip()
        try:
            docs = json.loads(result)
        except Exception:
            print_traceback()
            knowledge.append({'source': 'Uploaded Document', 'content': result})
            return knowledge
    else:
        docs = result
    
    try:
        _tmp_knowledge = []
        assert isinstance(docs, list)
        
        for doc in docs:
            # Handle vector search results
            if doc.get('url') == 'vector_search':
                snippets = doc.get('text', [])
                if snippets:
                    _tmp_knowledge.append({
                        'source': '📚 Vector Database Results',
                        'content': '\n\n---\n\n'.join(snippets) if isinstance(snippets, list) else snippets
                    })
            else:
                # Standard document results
                url = doc.get('url', 'unknown')
                snippets = doc.get('text', [])
                if isinstance(snippets, list):
                    content = '\n\n...\n\n'.join(snippets)
                else:
                    content = snippets
                
                _tmp_knowledge.append({
                    'source': f'[Document]({get_basename_from_url(url)})',
                    'content': content
                })
        
        knowledge.extend(_tmp_knowledge)
    except Exception:
        print_traceback()
        knowledge.append({'source': 'Uploaded Document', 'content': str(result)})
    
    return knowledge

assistant.format_knowledge_to_source_and_content = format_knowledge_to_source_and_content

# ============= Streaming Output with Performance Stats =============
HINT_PATTERN = "\n<summary>Input: {input_tokens} tokens | Output: {output_tokens} tokens | Speed: {decode_speed:.1f} tok/s</summary>"

@staticmethod
def _full_stream_output(response):
    """Enhanced streaming with performance metrics"""
    start_time = time()
    total_tokens = 0
    
    for chunk in response:
        if chunk.status_code == HTTPStatus.OK:
            try:
                # Calculate tokens per second
                elapsed = time() - start_time
                if elapsed > 0 and hasattr(chunk, 'usage'):
                    total_tokens = chunk.usage.output_tokens
                    tokens_per_sec = total_tokens / elapsed if elapsed > 0 else 0
                    
                    hint = HINT_PATTERN.format(
                        input_tokens=chunk.usage.input_tokens,
                        output_tokens=total_tokens,
                        decode_speed=tokens_per_sec
                    )
                    content = chunk.output.choices[0].message.content + hint
                else:
                    content = chunk.output.choices[0].message.content
                
                yield [Message(ASSISTANT, content)]
            except Exception as e:
                logger.error(f"Stream output error: {e}")
                yield [Message(ASSISTANT, chunk.output.choices[0].message.content)]
        else:
            raise ModelServiceError(code=chunk.code, message=chunk.message)

# Only patch if QwenChatAtDS is available
try:
    QwenChatAtDS._full_stream_output = _full_stream_output
except:
    pass

# ============= Assistant Run with Vector DB Support =============
def assistant_run(self, messages, lang="en", knowledge="", **kwargs):
    """Enhanced assistant run with vector database integration"""
    
    # Check if files are being uploaded
    if any([len(message.get(CONTENT, [])) > 1 for message in messages]):
        yield [Message(ASSISTANT, "📁 Uploading and parsing files...")]
    
    # Prepare messages with knowledge
    new_messages = self._prepend_knowledge_prompt(
        messages=messages, 
        lang=lang, 
        knowledge=knowledge, 
        **kwargs
    )
    
    # Check if we're using vector search
    using_vector_search = False
    if hasattr(self, 'rag_cfg'):
        searchers = self.rag_cfg.get('rag_searchers', [])
        using_vector_search = 'vector_search' in searchers
    
    if using_vector_search:
        yield [Message(ASSISTANT, "🔍 Searching vector database...")]
    else:
        yield [Message(ASSISTANT, "💭 Processing...")]
    
    start_time = time()
    
    try:
        # Run the base assistant
        for chunk in super(Assistant, self)._run(
            messages=new_messages, 
            lang=lang, 
            **kwargs
        ):
            # Add performance metrics if available
            if chunk and chunk[0].get(CONTENT):
                elapsed = time() - start_time
                if elapsed > 1:  # Only show after 1 second
                    # Remove old pattern and add new one
                    content = chunk[0][CONTENT]
                    content = re.sub(r'\n<summary>.*?</summary>', '', content)
                    
                    # Add performance info if not already present
                    if '</summary>' not in content:
                        tokens_estimate = len(content) // 4  # Rough estimate
                        speed = tokens_estimate / elapsed if elapsed > 0 else 0
                        perf_info = f"\n<summary>Time: {elapsed:.1f}s | Est. speed: {speed:.1f} tok/s</summary>"
                        chunk[0][CONTENT] = content + perf_info
            
            yield chunk
            
    except torch.cuda.OutOfMemoryError:
        logger.error("CUDA OOM in assistant run")
        torch.cuda.empty_cache()
        gc.collect()
        
        error_msg = (
            "⚠️ **GPU Memory Exceeded**\n\n"
            "The document or conversation is too long. Please try:\n"
            "1. Switch to 4-bit mode with vector database\n"
            "2. Clear the chat history with /clear\n"
            "3. Upload smaller documents"
        )
        yield [Message(ASSISTANT, error_msg)]
        
    except Exception as e:
        logger.error(f"Assistant run error: {e}")
        print_traceback()
        yield [Message(ASSISTANT, f"Error: {str(e)}")]

Assistant._run = assistant_run

# ============= Memory Management Utilities =============
def clear_gpu_memory():
    """Utility to clear GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()
        logger.info("GPU memory cleared")

def get_memory_usage():
    """Get current memory usage stats"""
    stats = {}
    
    if torch.cuda.is_available():
        stats['gpu_allocated'] = torch.cuda.memory_allocated() / 1024**3
        stats['gpu_reserved'] = torch.cuda.memory_reserved() / 1024**3
        stats['gpu_free'] = (torch.cuda.get_device_properties(0).total_memory - 
                             torch.cuda.memory_allocated()) / 1024**3
    
    import psutil
    process = psutil.Process()
    stats['cpu_memory'] = process.memory_info().rss / 1024**3
    stats['cpu_percent'] = psutil.cpu_percent()
    
    return stats

# ============= Export Utilities =============
__all__ = [
    'common_programming_language_extensions',
    'clear_gpu_memory',
    'get_memory_usage',
]

logger.info("Patching module loaded successfully with vector database support")