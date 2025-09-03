"""
Custom Assistant that handles vector database without DashScope and supports OCR
"""

import os
import hashlib
from typing import List, Iterator, Dict, Any, Optional, Tuple
from qwen_agent.agents import Assistant
from qwen_agent.llm.schema import Message, ASSISTANT, USER, SYSTEM, CONTENT
from qwen_agent.log import logger
from qwen_agent.utils.utils import extract_text_from_message
from qwen_agent.tools.simple_doc_parser import SimpleDocParser
from vector_db import VectorDatabase, RAGSearch
from ocr_processor import OCRProcessor, process_image_or_pdf
from memory_manager import cleanup_memory


class VectorDBAssistant(Assistant):
    """Custom assistant that uses local vector database and OCR support"""
    
    def __init__(self, llm: Dict[str, Any], name: str = "", description: str = "", 
                 use_vector_db: bool = True, max_ref_token: int = 30000):
        # Initialize parent without RAG to avoid DashScope
        super().__init__(
            llm=llm,
            name=name,
            description=description,
            # No rag_cfg to avoid DashScope
        )
        
        self.use_vector_db = use_vector_db
        self.max_ref_token = max_ref_token
        
        # Initialize document parser
        self.doc_parser = SimpleDocParser()
        
        # Initialize OCR processor
        try:
            self.ocr_processor = OCRProcessor(languages=['en', 'chi_sim'])  # English and Chinese
            logger.info("OCR processor initialized")
        except Exception as e:
            logger.warning(f"Could not initialize OCR processor: {e}")
            self.ocr_processor = None
        
        # Initialize vector database if needed
        if use_vector_db:
            try:
                self.vector_db = VectorDatabase(
                    persist_directory="./vector_db_storage",
                    embedding_model="all-MiniLM-L6-v2",
                    use_cpu=True
                )
                self.rag_search = RAGSearch(self.vector_db)
                logger.info("Vector database initialized for assistant")
            except Exception as e:
                logger.warning(f"Could not initialize vector database: {e}")
                self.vector_db = None
                self.rag_search = None
                self.use_vector_db = False
        else:
            self.vector_db = None
            self.rag_search = None
    
    def _process_files(self, messages: List[Message]) -> Tuple[List[Any], str]:
        """Extract and process files from messages with OCR support"""
        files = []
        file_contents = []
        
        for msg in messages:
            if msg.role == USER and isinstance(msg.content, list):
                for item in msg.content:
                    # Handle image files
                    if isinstance(item, dict) and 'image' in item:
                        image_path = item['image']
                        if image_path.startswith('file://'):
                            image_path = image_path[7:]  # Remove 'file://' prefix
                        
                        files.append(image_path)
                        
                        # Process image with OCR
                        if self.ocr_processor:
                            try:
                                logger.info(f"Processing image with OCR: {image_path}")
                                text = self.ocr_processor.extract_text_from_image(image_path)
                                if text:
                                    file_contents.append(f"[Image content via OCR]:\n{text}")
                                else:
                                    file_contents.append("[Image - no text detected]")
                            except Exception as e:
                                logger.error(f"OCR failed for image {image_path}: {e}")
                                file_contents.append(f"[Image - OCR failed: {image_path}]")
                        else:
                            file_contents.append(f"[Image - OCR not available: {image_path}]")
                    
                    # Handle regular files
                    elif isinstance(item, dict) and 'file' in item:
                        file_path = item['file']
                        files.append(file_path)
                        
                        # Check if it's an image or potentially scanned PDF
                        file_ext = os.path.splitext(file_path)[1].lower()
                        
                        if file_ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp']:
                            # Process as image
                            if self.ocr_processor:
                                try:
                                    logger.info(f"Processing image file with OCR: {file_path}")
                                    text = self.ocr_processor.extract_text_from_image(file_path)
                                    if text:
                                        file_contents.append(f"[Image content via OCR from {os.path.basename(file_path)}]:\n{text}")
                                    else:
                                        file_contents.append(f"[Image {os.path.basename(file_path)} - no text detected]")
                                except Exception as e:
                                    logger.error(f"OCR failed for {file_path}: {e}")
                                    file_contents.append(f"[Image - OCR failed: {os.path.basename(file_path)}]")
                            else:
                                file_contents.append(f"[Image file: {os.path.basename(file_path)} - OCR not available]")
                        
                        elif file_ext == '.pdf':
                            # Check if it's a scanned PDF
                            is_scanned = False
                            if self.ocr_processor:
                                is_scanned = self.ocr_processor.is_scanned_pdf(file_path)
                            
                            if is_scanned and self.ocr_processor:
                                # Process scanned PDF with OCR
                                try:
                                    logger.info(f"Processing scanned PDF with OCR: {file_path}")
                                    text = self.ocr_processor.extract_text_from_pdf_images(file_path, max_pages=20)
                                    if text:
                                        file_contents.append(f"[Scanned PDF content via OCR from {os.path.basename(file_path)}]:\n{text}")
                                    else:
                                        file_contents.append(f"[Scanned PDF {os.path.basename(file_path)} - no text detected]")
                                except Exception as e:
                                    logger.error(f"OCR failed for scanned PDF {file_path}: {e}")
                                    # Try regular parsing as fallback
                                    content = self._parse_file_with_doc_parser(file_path)
                                    if content:
                                        file_contents.append(content)
                                    else:
                                        file_contents.append(f"[Scanned PDF - OCR failed: {os.path.basename(file_path)}]")
                            else:
                                # Regular PDF with text - use SimpleDocParser
                                content = self._parse_file_with_doc_parser(file_path)
                                if content:
                                    file_contents.append(content)
                        
                        else:
                            # Other file types - use SimpleDocParser
                            content = self._parse_file_with_doc_parser(file_path)
                            if content:
                                file_contents.append(content)
        
        return files, "\n\n".join(file_contents)
    
    def _parse_file_with_doc_parser(self, file_path: str) -> str:
        """Parse file using SimpleDocParser"""
        try:
            parsed_content = self.doc_parser.call({'url': file_path})
            if isinstance(parsed_content, list):
                # Structured document
                content = ""
                for page in parsed_content:
                    if 'content' in page:
                        for para in page['content']:
                            if 'text' in para:
                                content += para['text'] + "\n"
                            elif 'table' in para:
                                content += str(para['table']) + "\n"
                return content
            else:
                # Plain text document
                return str(parsed_content)
        except Exception as e:
            logger.warning(f"Could not parse file {file_path}: {e}")
            # Fallback to simple text reading
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()
            except:
                return f"[Could not read file: {file_path}]"
    
    def _split_into_chunks(self, text: str, chunk_size: int = 1000) -> List[str]:
        """Split text into chunks"""
        chunks = []
        words = text.split()
        current_chunk = []
        current_size = 0
        
        for word in words:
            current_chunk.append(word)
            current_size += len(word) + 1
            
            if current_size >= chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_size = 0
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def _run(self, messages: List[Message], lang: str = 'en', **kwargs) -> Iterator[List[Message]]:
        """Custom run method that handles vector database and OCR"""
        
        # Clean up memory before processing
        cleanup_memory()
        
        # Check for files in messages
        files, file_content = self._process_files(messages)
        
        if files and self.use_vector_db and self.vector_db:
            # Process with vector database
            if file_content:
                yield [Message(ASSISTANT, "📁 Processing documents with vector database...")]
                
                # Add documents to vector DB
                doc_id = hashlib.md5(str(files).encode()).hexdigest()
                
                # Split content into chunks
                chunks = self._split_into_chunks(file_content, chunk_size=1000)
                
                if chunks:
                    success = self.vector_db.add_document(
                        document_id=doc_id,
                        chunks=chunks,
                        metadata={'files': files}
                    )
                    
                    if success:
                        logger.info(f"Added {len(chunks)} chunks to vector database")
                        
                        # Get query from last user message
                        query = ""
                        for msg in reversed(messages):
                            if msg.role == USER:
                                query = extract_text_from_message(msg, add_upload_info=False)
                                break
                        
                        # Search for relevant chunks
                        if query:
                            yield [Message(ASSISTANT, "🔍 Searching for relevant information...")]
                            relevant_chunks = self.rag_search.search_relevant_chunks(query, max_chunks=5)
                            
                            # Create context from relevant chunks
                            if relevant_chunks:
                                context = "\n\n---\n\n".join(relevant_chunks)
                                
                                # Create new message with context
                                context_msg = f"""Based on the following document excerpts:

{context}

Now, answering your question: {query}"""
                                
                                # Replace file references with context
                                new_messages = []
                                for msg in messages:
                                    if msg.role == USER and isinstance(msg.content, list):
                                        # Create new content without files but with context
                                        new_content = []
                                        for item in msg.content:
                                            if isinstance(item, dict) and 'text' in item:
                                                new_content.append({'text': context_msg})
                                                break
                                        if new_content:
                                            new_msg = Message(role=USER, content=new_content)
                                            new_messages.append(new_msg)
                                        else:
                                            new_messages.append(msg)
                                    else:
                                        new_messages.append(msg)
                                
                                messages = new_messages
                        else:
                            # No specific query, just acknowledge the upload
                            yield [Message(ASSISTANT, f"✅ Processed {len(chunks)} chunks from your documents. Please ask a question about them.")]
                            return
        
        elif files and not self.use_vector_db:
            # For non-vector DB mode, truncate content if needed
            if file_content:
                if len(file_content) > self.max_ref_token * 4:  # Rough char to token ratio
                    file_content = file_content[:self.max_ref_token * 4] + "\n\n[Content truncated due to length]"
                
                # Add content to messages
                new_messages = []
                for msg in messages:
                    if msg.role == USER and isinstance(msg.content, list):
                        new_content = []
                        has_file = False
                        for item in msg.content:
                            if isinstance(item, dict) and ('file' in item or 'image' in item):
                                has_file = True
                            elif isinstance(item, dict) and 'text' in item:
                                new_content.append(item)
                        
                        if has_file:
                            # Add file content as context
                            text = new_content[0]['text'] if new_content else ""
                            context_text = f"Document content:\n{file_content}\n\nQuestion: {text}"
                            new_msg = Message(role=USER, content=[{'text': context_text}])
                            new_messages.append(new_msg)
                        else:
                            new_messages.append(msg)
                    else:
                        new_messages.append(msg)
                
                messages = new_messages
        
        # Run the base assistant
        yield [Message(ASSISTANT, "💭 Generating response...")]
        
        try:
            for response in super()._run(messages=messages, lang=lang, **kwargs):
                yield response
        finally:
            # Clean up memory after response
            cleanup_memory()