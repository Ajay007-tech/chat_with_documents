import os
import re
from typing import List, Union, Iterator
from http import HTTPStatus
from time import time
import time
import json

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

# Import OCR processor
try:
    from ocr_processor import get_ocr_processor, IMAGE_EXTENSIONS
    ocr_processor = get_ocr_processor()
    OCR_AVAILABLE = ocr_processor is not None
    logger.info("status of ocr",OCR_AVAILABLE)
except ImportError:
    OCR_AVAILABLE = False
    ocr_processor = None
    IMAGE_EXTENSIONS = ['jpg', 'jpeg', 'png', 'bmp', 'gif', 'tiff', 'tif', 'webp']
    logger.warning("OCR processor not available")

def memory_run(self, messages: List[Message], lang: str = 'en', **kwargs) -> Iterator[List[Message]]:
        """This agent is responsible for processing the input files in the message.

         This method stores the files in the knowledge base, and retrievals the relevant parts
         based on the query and returning them.
         The currently supported file types include: .pdf, .docx, .pptx, .txt, .csv, .tsv, .xlsx, .xls and html.

         Args:
             messages: A list of messages.
             lang: Language.

        Yields:
            The message of retrieved documents.
        """
        # process files in messages
        rag_files = self.get_rag_files(messages)

        if not rag_files:
            yield [Message(role=ASSISTANT, content='', name='memory')]
        else:
            query = ''
            # Only retrieval content according to the last user query if exists
            if messages and messages[-1].role == USER:
                query = extract_text_from_message(messages[-1], add_upload_info=False)

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

common_programming_language_extensions = [
    "py",  # Python
    "java",  # Java
    "cpp",  # C++
    "c",  # C
    "h",  # C/C++ header files
    "cs",  # C#
    "js",  # JavaScript
    "ts",  # TypeScript
    "rb",  # Ruby
    "php",  # PHP
    "swift",  # Swift
    "go",  # Go
    "rs",  # Rust
    "kt",  # Kotlin
    "scala",  # Scala
    "m",  # Objective-C
    "css",  # CSS
    "sql",  # SQL
    "sh",  # Shell
    "pl",  # Perl
    "r",  # R
    "jl",  # Julia
    "dart",  # Dart
    "json",  # JSON
    "xml",  # XML
    "yml",  # YAML
    "toml",  # TOML
]

def SimpleDocParser_call(self, params: Union[str, dict], **kwargs) -> Union[str, list]:
    params = self._verify_json_format_args(params)
    path = params['url']
    cached_name_ori = f'{hash_sha256(path)}_ori'
    try:
        # Directly load the parsed doc
        parsed_file = self.db.get(cached_name_ori)
        # [PATCH]: disable json5 for faster processing
        parsed_file = json.loads(parsed_file)
        logger.info(f'Read parsed {path} from cache.')
    except KeyNotExistsError:
        logger.info(f'Start parsing {path}...')
        time1 = time.time()

        f_type = get_file_type(path)
        if f_type in PARSER_SUPPORTED_FILE_TYPES + common_programming_language_extensions + IMAGE_EXTENSIONS:
            if path.startswith('https://') or path.startswith('http://') or re.match(
                    r'^[A-Za-z]:\\', path) or re.match(r'^[A-Za-z]:/', path):
                path = path
            else:
                path = sanitize_chrome_file_path(path)

        os.makedirs(self.data_root, exist_ok=True)
        if is_http_url(path):
            # download online url
            tmp_file_root = os.path.join(self.data_root, hash_sha256(path))
            os.makedirs(tmp_file_root, exist_ok=True)
            path = save_url_to_local_work_dir(path, tmp_file_root)

        # Handle different file types with OCR support
        if f_type == 'pdf':
            # Use OCR-enabled PDF parser if available
            if OCR_AVAILABLE and ocr_processor:
                parsed_file = ocr_processor.process_pdf_with_ocr(path, self.extract_image)
            else:
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
        elif f_type in IMAGE_EXTENSIONS:
            # Handle image files with OCR
            if OCR_AVAILABLE and ocr_processor:
                parsed_file = ocr_processor.process_image_file(path)
            else:
                # Fallback if OCR not available
                parsed_file = [{
                    'page': 1,
                    'content': [{'text': '[Image file - OCR not available]'}]
                }]
        else:
            raise ValueError(
                f'Failed: The current parser does not support this file type! Supported types: {"/".join(PARSER_SUPPORTED_FILE_TYPES + common_programming_language_extensions + IMAGE_EXTENSIONS)}'
            )
        for page in parsed_file:
            for para in page['content']:
                # Todo: More attribute types
                para['token'] = count_tokens(para.get('text', para.get('table')))
        time2 = time.time()
        logger.info(f'Finished parsing {path}. Time spent: {time2 - time1} seconds.')
        # Cache the parsing doc
        self.db.put(cached_name_ori, json.dumps(parsed_file, ensure_ascii=False, indent=2))

    if not self.structured_doc:
        return get_plain_doc(parsed_file)
    else:
        return parsed_file

SimpleDocParser.call = SimpleDocParser_call


def _truncate_input_messages_roughly(messages: List[Message], max_tokens: int) -> List[Message]:
    sys_msg = messages[0]
    assert sys_msg.role == SYSTEM  # The default system is prepended if none exists
    if len([m for m in messages if m.role == SYSTEM]) >= 2:
        raise gr.Error(
            'The input messages must contain no more than one system message. '
            ' And the system message, if exists, must be the first message.',
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
                    'The input messages (excluding the system message) must start with a user message.',
                )

    def _count_tokens(msg: Message) -> int:
        return tokenizer.count_tokens(extract_text_from_message(msg, add_upload_info=True))

    token_cnt = _count_tokens(sys_msg)
    truncated = []
    for i, turn in enumerate(reversed(turns)):
        cur_turn_msgs = []
        cur_token_cnt = 0
        for m in reversed(turn):
            cur_turn_msgs.append(m)
            cur_token_cnt += _count_tokens(m)
        # Check "i == 0" so that at least one user message is included
        # [PATCH] Do not do truncate for this demo
        # if (i == 0) or (token_cnt + cur_token_cnt <= max_tokens):
        truncated.extend(cur_turn_msgs)
        token_cnt += cur_token_cnt
        # else:
        #     break
    # Always include the system message
    truncated.append(sys_msg)
    truncated.reverse()

    if len(truncated) < 2:  # one system message + one or more user messages
        raise gr.Error(
            code='400',
            message='At least one user message should be provided.',
        )
    if token_cnt > max_tokens:
        raise gr.Error(
            f'The input messages (around {token_cnt} tokens) exceed the maximum context length ({max_tokens} tokens).'
        )
    return truncated

qwen_agent.llm.base._truncate_input_messages_roughly = _truncate_input_messages_roughly



def format_knowledge_to_source_and_content(result: Union[str, List[dict]]) -> List[dict]:
    knowledge = []
    if isinstance(result, str):
        result = f'{result}'.strip()
        try:
            # [PATCH]: disable json5 for faster processing
            docs = json.loads(result)
        except Exception:
            print_traceback()
            knowledge.append({'source': '上传的文档', 'content': result})
            return knowledge
    else:
        docs = result
    try:
        _tmp_knowledge = []
        assert isinstance(docs, list)
        for doc in docs:
            url, snippets = doc['url'], doc['text']
            assert isinstance(snippets, list)
            _tmp_knowledge.append({
                'source': f'[文件]({get_basename_from_url(url)})',
                'content': '\n\n...\n\n'.join(snippets)
            })
        knowledge.extend(_tmp_knowledge)
    except Exception:
        print_traceback()
        knowledge.append({'source': '上传的文档', 'content': result})
    return knowledge

assistant.format_knowledge_to_source_and_content = format_knowledge_to_source_and_content


HINT_PATTERN = "\n<summary>input tokens: {input_tokens}, prefill time: [[<PrefillCost>]]s, output tokens: {output_tokens}, decode speed: [[<DecodeSpeed>]] tokens/s</summary>"

@staticmethod
def _full_stream_output(response):
    for chunk in response:
        if chunk.status_code == HTTPStatus.OK:
            # [PATCH]: add speed statistics
            yield [Message(ASSISTANT, chunk.output.choices[0].message.content + HINT_PATTERN.format(
                    input_tokens=chunk.usage.input_tokens,
                    output_tokens=chunk.usage.output_tokens,)
            )]
        else:
            raise ModelServiceError(code=chunk.code, message=chunk.message)

QwenChatAtDS._full_stream_output = _full_stream_output

def assistant_run(self,
        messages,
        lang="en",
        knowledge="",
        **kwargs):
    
    if any([len(message[CONTENT]) > 1 for message in messages]):
        yield [Message(ASSISTANT, "Uploading and Parsing Files...")]
    new_messages = self._prepend_knowledge_prompt(messages=messages, lang=lang, knowledge=knowledge, **kwargs)
    start_prefill_time = time.time()
    
    yield [Message(ASSISTANT, "model is thinking...")]

    start_decode_time = None
    for chunk in super(Assistant, self)._run(messages=new_messages, lang=lang, **kwargs):
    
        if start_decode_time is None:
            end_prefill_time = time.time()
            start_decode_time = time.time() - 0.5

        # [PATCH]: compute speed statstics
        pattern = re.search(HINT_PATTERN.format(input_tokens="\d+", output_tokens="(\d+)").replace("[", "\[").replace("]", "\]"), chunk[0][CONTENT])
        if pattern:
            output_tokens = int(pattern.group(1))
            chunk[0][CONTENT] = chunk[0][CONTENT].replace("[[<PrefillCost>]]", "%.2f" % (end_prefill_time - start_prefill_time)).replace("[[<DecodeSpeed>]]", "%.2f" % (output_tokens/(time.time() - start_decode_time)))

        yield chunk

Assistant._run = assistant_run