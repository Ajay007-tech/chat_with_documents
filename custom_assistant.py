"""
Custom Assistant that handles all document types with vector database and OCR support
Preserves original functionality for DOCX, CSV, Excel, Python files, etc.
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
from memory_manager import cleanup_memory

# Try to import OCR processor
try:
    from ocr_processor import OCRProcessor, get_ocr_processor, process_image_or_pdf
    OCR_AVAILABLE = True
    logger.info("OCR processor imported successfully")
except ImportError:
    OCR_AVAILABLE = False
    logger.warning("OCR processor not available for images/scanned PDFs")


class VectorDBAssistant(Assistant):
    """Custom assistant that handles all document types including OCR"""
    
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
        
        # Initialize document parser for regular documents
        self.doc_parser = SimpleDocParser()
        logger.info("SimpleDocParser initialized for regular documents")
        
        # Initialize OCR processor if available
        self.ocr_processor = None
        if OCR_AVAILABLE:
            try:
                self.ocr_processor = get_ocr_processor(['en'])  # English and Chinese
                if self.ocr_processor:
                    logger.info("OCR processor initialized successfully")
                else:
                    logger.warning("OCR processor could not be initialized")
            except Exception as e:
                logger.warning(f"Could not initialize OCR processor: {e}")
                self.ocr_processor = None
        else:
            logger.info("OCR not available - install easyocr for image/scanned PDF support")
        
        # Initialize vector database if needed
        if use_vector_db:
            try:
                self.vector_db = VectorDatabase(
                    persist_directory="./vector_db_storage",
                    embedding_model="all-MiniLM-L6-v2",
                    use_cpu=True
                )
                self.rag_search = RAGSearch(self.vector_db)
                logger.info("Vector database initialized")
            except Exception as e:
                logger.warning(f"Could not initialize vector database: {e}")
                self.vector_db = None
                self.rag_search = None
                self.use_vector_db = False
        else:
            self.vector_db = None
            self.rag_search = None
    
    def _process_files(self, messages: List[Message]) -> Tuple[List[Any], str]:
        """
        Extract and process files from messages
        Handles all file types: documents, code, images, and scanned PDFs
        """
        files = []
        file_contents = []
        
        for msg in messages:
            if msg.role == USER and isinstance(msg.content, list):
                for item in msg.content:
                    # Handle image files directly in message
                    if isinstance(item, dict) and 'image' in item:
                        image_path = item['image']
                        if image_path.startswith('file://'):
                            image_path = image_path[7:]  # Remove 'file://' prefix
                        
                        files.append(image_path)
                        
                        # Process image with OCR if available
                        if self.ocr_processor:
                            try:
                                logger.info(f"Processing image with OCR: {image_path}")
                                text = self.ocr_processor.extract_text_from_image(image_path)
                                if text and not text.startswith("[OCR"):
                                    file_contents.append(f"[Image content from {os.path.basename(image_path)}]:\n{text}")
                                    logger.info(f"OCR extracted {len(text)} characters from image")
                                else:
                                    file_contents.append(f"[Image {os.path.basename(image_path)} - no text detected]")
                            except Exception as e:
                                logger.error(f"OCR failed for image {image_path}: {e}")
                                file_contents.append(f"[Image {os.path.basename(image_path)} - OCR failed]")
                        else:
                            file_contents.append(f"[Image {os.path.basename(image_path)} - OCR not available]")
                    
                    # Handle all other files
                    elif isinstance(item, dict) and 'file' in item:
                        file_path = item['file']
                        files.append(file_path)
                        
                        # Get file extension
                        file_ext = os.path.splitext(file_path)[1].lower()
                        file_name = os.path.basename(file_path)
                        
                        logger.info(f"Processing file: {file_name} (type: {file_ext})")
                        
                        # Determine file type and process accordingly
                        processed = False
                        
                        # 1. Check if it's an image file that needs OCR
                        if file_ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.tif', '.webp']:
                            if self.ocr_processor:
                                try:
                                    logger.info(f"Processing image file with OCR: {file_name}")
                                    text = self.ocr_processor.extract_text_from_image(file_path)
                                    if text and not text.startswith("[OCR"):
                                        file_contents.append(f"[Image content from {file_name}]:\n{text}")
                                        logger.info(f"OCR extracted {len(text)} characters")
                                        processed = True
                                    else:
                                        file_contents.append(f"[Image {file_name} - no text detected]")
                                        processed = True
                                except Exception as e:
                                    logger.error(f"OCR failed for {file_name}: {e}")
                                    file_contents.append(f"[Image {file_name} - OCR failed]")
                                    processed = True
                            else:
                                file_contents.append(f"[Image {file_name} - OCR not available]")
                                processed = True
                        
                        # 2. Check if it's a PDF that might be scanned
                        elif file_ext == '.pdf':
                            # First check if it's scanned
                            is_scanned = False
                            if self.ocr_processor:
                                try:
                                    is_scanned = self.ocr_processor.is_scanned_pdf(file_path)
                                    logger.info(f"PDF {file_name} - Scanned: {is_scanned}")
                                except Exception as e:
                                    logger.error(f"Error checking if PDF is scanned: {e}")
                                    is_scanned = False
                            
                            if is_scanned and self.ocr_processor:
                                # Process scanned PDF with OCR
                                try:
                                    logger.info(f"Processing scanned PDF with OCR: {file_name}")
                                    text = self.ocr_processor.extract_text_from_pdf_images(file_path, max_pages=20)
                                    if text and not text.startswith("["):
                                        file_contents.append(f"[Scanned PDF content from {file_name}]:\n{text}")
                                        logger.info(f"OCR extracted {len(text)} characters from scanned PDF")
                                        processed = True
                                    else:
                                        logger.warning(f"No text extracted from scanned PDF: {file_name}")
                                except Exception as e:
                                    logger.error(f"OCR failed for scanned PDF {file_name}: {e}")
                            
                            # If not scanned or OCR failed, use regular parser
                            if not processed:
                                logger.info(f"Processing PDF with SimpleDocParser: {file_name}")
                                content = self._parse_regular_document(file_path, file_name)
                                if content:
                                    file_contents.append(content)
                                    processed = True
                        
                        # 3. For all other files (DOCX, TXT, CSV, Excel, Python, etc.), use SimpleDocParser
                        if not processed:
                            logger.info(f"Processing {file_ext} file with SimpleDocParser: {file_name}")
                            content = self._parse_regular_document(file_path, file_name)
                            if content:
                                file_contents.append(content)
                            else:
                                # If parser fails, try direct reading as fallback
                                logger.info(f"SimpleDocParser failed, trying direct read for: {file_name}")
                                content = self._read_file_directly(file_path, file_name)
                                if content:
                                    file_contents.append(content)
        
        return files, "\n\n".join(file_contents)
    
    def _parse_regular_document(self, file_path: str, file_name: str) -> str:
        """Parse regular documents using SimpleDocParser"""
        try:
            # Use SimpleDocParser to parse the file
            parsed_content = self.doc_parser.call({'url': file_path})
            
            if isinstance(parsed_content, list):
                # Structured document (PDF, DOCX, etc.)
                content_parts = []
                for page_num, page in enumerate(parsed_content):
                    if 'content' in page:
                        page_content = []
                        for para in page['content']:
                            if 'text' in para and para['text'].strip():
                                page_content.append(para['text'])
                            elif 'table' in para:
                                page_content.append(str(para['table']))
                        
                        if page_content:
                            # Add page number for multi-page documents
                            if len(parsed_content) > 1:
                                content_parts.append(f"[Page {page_num + 1}]")
                            content_parts.extend(page_content)
                
                if content_parts:
                    return f"[Document: {file_name}]\n" + "\n".join(content_parts)
            else:
                # Plain text document
                content = str(parsed_content).strip()
                if content:
                    return f"[Document: {file_name}]\n{content}"
            
            return ""
            
        except Exception as e:
            logger.warning(f"SimpleDocParser failed for {file_name}: {e}")
            return ""
    
    def _read_file_directly(self, file_path: str, file_name: str) -> str:
        """Direct file reading as fallback"""
        try:
            # Determine encoding
            encodings = ['utf-8', 'latin-1', 'cp1252', 'gbk']
            
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read()
                        if content:
                            logger.info(f"Successfully read {file_name} with {encoding} encoding")
                            return f"[File: {file_name}]\n{content}"
                except (UnicodeDecodeError, UnicodeError):
                    continue
            
            # If all encodings fail, try binary read for certain file types
            file_ext = os.path.splitext(file_path)[1].lower()
            if file_ext in ['.pdf', '.docx', '.xlsx', '.xls', '.pptx']:
                return f"[Binary file {file_name} - use SimpleDocParser for proper parsing]"
            
            return f"[Could not read file: {file_name}]"
            
        except Exception as e:
            logger.error(f"Failed to read file {file_name}: {e}")
            return f"[Error reading file: {file_name}]"
    
    def _split_into_chunks(self, text: str, chunk_size: int = 1000) -> List[str]:
        """Split text into chunks for vector database"""
        chunks = []
        
        # Split by paragraphs first
        paragraphs = text.split('\n\n')
        
        current_chunk = []
        current_size = 0
        
        for para in paragraphs:
            para_size = len(para)
            
            if current_size + para_size > chunk_size and current_chunk:
                # Save current chunk
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = [para]
                current_size = para_size
            else:
                current_chunk.append(para)
                current_size += para_size
        
        # Add last chunk
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
        
        return chunks
    
    def _run(self, messages: List[Message], lang: str = 'en', **kwargs) -> Iterator[List[Message]]:
        """
        Custom run method that handles vector database
        Works with all document types
        """
        
        # Clean up memory before processing
        cleanup_memory()
        
        # Check for files in messages
        files, file_content = self._process_files(messages)
        
        if files and file_content:
            logger.info(f"Processed {len(files)} files, total content length: {len(file_content)} characters")
            
            # Show message if OCR was used
            for file in files:
                file_ext = os.path.splitext(file)[1].lower()
                if file_ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.tif', '.webp']:
                    if self.ocr_processor:
                        yield [Message(ASSISTANT, "📸 Processing image with OCR...")]
                    break
                elif file_ext == '.pdf' and self.ocr_processor:
                    if self.ocr_processor.is_scanned_pdf(file):
                        yield [Message(ASSISTANT, "📄 Processing scanned PDF with OCR...")]
                        break
            
            if self.use_vector_db and self.vector_db:
                # Process with vector database
                yield [Message(ASSISTANT, "🔍 Processing documents with vector database...")]
                
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
                            yield [Message(ASSISTANT, "🔎 Searching for relevant information...")]
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
            
            else:
                # For non-vector DB mode, add content directly
                if len(file_content) > self.max_ref_token * 4:  # Rough char to token ratio
                    file_content = file_content[:self.max_ref_token * 4] + "\n\n[Content truncated due to length]"
                
                # Add content to messages
                new_messages = []
                for msg in messages:
                    if msg.role == USER and isinstance(msg.content, list):
                        new_content = []
                        has_file = False
                        text_content = ""
                        
                        for item in msg.content:
                            if isinstance(item, dict) and ('file' in item or 'image' in item):
                                has_file = True
                            elif isinstance(item, dict) and 'text' in item:
                                text_content = item['text']
                        
                        if has_file:
                            # Combine file content with query
                            combined_text = f"Document content:\n{file_content}\n\n"
                            if text_content:
                                combined_text += f"Question: {text_content}"
                            else:
                                combined_text += "Please analyze this document."
                            
                            new_msg = Message(role=USER, content=[{'text': combined_text}])
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