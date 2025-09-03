"""
OCR Processor for handling images and scanned PDFs
Supports multiple OCR backends with fallback options
"""

import os
import io
import base64
from typing import List, Optional, Union, Tuple
from PIL import Image
import numpy as np
from qwen_agent.log import logger

# Try to import OCR libraries
OCR_AVAILABLE = False
OCR_BACKEND = None

# Try EasyOCR first (better for multilingual)
try:
    import easyocr
    OCR_AVAILABLE = True
    OCR_BACKEND = "easyocr"
    logger.info("EasyOCR available for image text extraction")
except ImportError:
    pass

# Try Tesseract as fallback
if not OCR_AVAILABLE:
    try:
        import pytesseract
        from pytesseract import Output
        OCR_AVAILABLE = True
        OCR_BACKEND = "tesseract"
        logger.info("Tesseract available for image text extraction")
    except ImportError:
        pass

# Try PaddleOCR as another option
if not OCR_AVAILABLE:
    try:
        from paddleocr import PaddleOCR
        OCR_AVAILABLE = True
        OCR_BACKEND = "paddle"
        logger.info("PaddleOCR available for image text extraction")
    except ImportError:
        pass

# PDF handling
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    logger.warning("PyMuPDF not available for PDF image extraction")

try:
    import pdf2image
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False
    logger.warning("pdf2image not available for PDF conversion")


class OCRProcessor:
    """Process images and scanned PDFs to extract text"""
    
    def __init__(self, backend: Optional[str] = None, languages: List[str] = ['en']):
        """
        Initialize OCR processor
        Args:
            backend: 'easyocr', 'tesseract', 'paddle', or None for auto-detect
            languages: List of language codes for OCR
        """
        self.backend = backend or OCR_BACKEND
        self.languages = languages
        self.ocr_reader = None
        
        if not OCR_AVAILABLE:
            logger.warning("No OCR library available. Install easyocr, pytesseract, or paddleocr")
            return
        
        # Initialize OCR reader based on backend
        if self.backend == "easyocr":
            try:
                # Use GPU if available, else CPU
                import torch
                use_gpu = torch.cuda.is_available()
                self.ocr_reader = easyocr.Reader(languages, gpu=use_gpu)
                logger.info(f"EasyOCR initialized with languages: {languages}, GPU: {use_gpu}")
            except Exception as e:
                logger.error(f"Failed to initialize EasyOCR: {e}")
                self.backend = None
                
        elif self.backend == "paddle":
            try:
                self.ocr_reader = PaddleOCR(use_angle_cls=True, lang='en' if 'en' in languages else 'ch')
                logger.info("PaddleOCR initialized")
            except Exception as e:
                logger.error(f"Failed to initialize PaddleOCR: {e}")
                self.backend = None
    
    def extract_text_from_image(self, image: Union[str, Image.Image, np.ndarray]) -> str:
        """
        Extract text from an image
        Args:
            image: Path to image, PIL Image, or numpy array
        Returns:
            Extracted text
        """
        if not OCR_AVAILABLE or not self.backend:
            return "[OCR not available - install easyocr or pytesseract]"
        
        try:
            # Convert to PIL Image if needed
            if isinstance(image, str):
                if os.path.exists(image):
                    pil_image = Image.open(image)
                else:
                    logger.error(f"Image file not found: {image}")
                    return ""
            elif isinstance(image, np.ndarray):
                pil_image = Image.fromarray(image)
            else:
                pil_image = image
            
            # Perform OCR based on backend
            if self.backend == "easyocr" and self.ocr_reader:
                # EasyOCR expects numpy array
                img_array = np.array(pil_image)
                results = self.ocr_reader.readtext(img_array)
                # Extract text from results
                text = "\n".join([result[1] for result in results])
                
            elif self.backend == "tesseract":
                # Tesseract
                text = pytesseract.image_to_string(pil_image)
                
            elif self.backend == "paddle" and self.ocr_reader:
                # PaddleOCR
                img_array = np.array(pil_image)
                results = self.ocr_reader.ocr(img_array, cls=True)
                # Extract text from results
                text_list = []
                for line in results[0] if results and results[0] else []:
                    if line and len(line) > 1:
                        text_list.append(line[1][0])
                text = "\n".join(text_list)
            else:
                text = ""
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            return f"[OCR failed: {str(e)}]"
    
    def is_scanned_pdf(self, pdf_path: str) -> bool:
        """
        Check if a PDF is scanned (image-based) or has text
        Args:
            pdf_path: Path to PDF file
        Returns:
            True if scanned/image PDF, False if has extractable text
        """
        if not PYMUPDF_AVAILABLE:
            # Assume it might be scanned if we can't check
            return True
        
        try:
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            pages_with_text = 0
            
            for page_num in range(min(3, total_pages)):  # Check first 3 pages
                page = doc[page_num]
                text = page.get_text().strip()
                if len(text) > 50:  # If page has substantial text
                    pages_with_text += 1
            
            doc.close()
            
            # If most checked pages have text, it's not scanned
            return pages_with_text < 2
            
        except Exception as e:
            logger.error(f"Error checking PDF type: {e}")
            return True
    
    def extract_text_from_pdf_images(self, pdf_path: str, max_pages: Optional[int] = None) -> str:
        """
        Extract text from scanned PDF using OCR
        Args:
            pdf_path: Path to PDF file
            max_pages: Maximum number of pages to process (None for all)
        Returns:
            Extracted text
        """
        if not OCR_AVAILABLE:
            return "[OCR not available for scanned PDF - install easyocr or pytesseract]"
        
        extracted_text = []
        
        try:
            if PYMUPDF_AVAILABLE:
                # Use PyMuPDF to extract images
                doc = fitz.open(pdf_path)
                total_pages = len(doc)
                pages_to_process = min(max_pages or total_pages, total_pages)
                
                logger.info(f"Processing {pages_to_process} pages from scanned PDF")
                
                for page_num in range(pages_to_process):
                    page = doc[page_num]
                    
                    # Try to get text first (in case some pages have text)
                    text = page.get_text().strip()
                    
                    if len(text) > 50:
                        # Page has text, use it
                        extracted_text.append(f"--- Page {page_num + 1} ---\n{text}")
                    else:
                        # Convert page to image and OCR it
                        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x scale for better OCR
                        img_data = pix.tobytes("png")
                        img = Image.open(io.BytesIO(img_data))
                        
                        # Perform OCR
                        page_text = self.extract_text_from_image(img)
                        if page_text:
                            extracted_text.append(f"--- Page {page_num + 1} (OCR) ---\n{page_text}")
                        
                        # Clean up
                        pix = None
                    
                    # Log progress
                    if (page_num + 1) % 5 == 0:
                        logger.info(f"Processed {page_num + 1}/{pages_to_process} pages")
                
                doc.close()
                
            elif PDF2IMAGE_AVAILABLE:
                # Use pdf2image as fallback
                from pdf2image import convert_from_path
                
                images = convert_from_path(pdf_path, first_page=1, 
                                          last_page=max_pages or None)
                
                for i, img in enumerate(images):
                    page_text = self.extract_text_from_image(img)
                    if page_text:
                        extracted_text.append(f"--- Page {i + 1} (OCR) ---\n{page_text}")
                    
                    if (i + 1) % 5 == 0:
                        logger.info(f"Processed {i + 1}/{len(images)} pages")
            else:
                return "[Cannot process scanned PDF - install PyMuPDF or pdf2image]"
            
            return "\n\n".join(extracted_text)
            
        except Exception as e:
            logger.error(f"Failed to extract text from scanned PDF: {e}")
            return f"[Failed to process scanned PDF: {str(e)}]"
    
    def process_file(self, file_path: str) -> Tuple[str, bool]:
        """
        Process any file (image, PDF, or text) and extract text
        Args:
            file_path: Path to file
        Returns:
            Tuple of (extracted_text, is_ocr_processed)
        """
        file_ext = os.path.splitext(file_path)[1].lower()
        
        # Image files
        if file_ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp']:
            logger.info(f"Processing image file: {file_path}")
            text = self.extract_text_from_image(file_path)
            return text, True
        
        # PDF files
        elif file_ext == '.pdf':
            # Check if it's scanned
            if self.is_scanned_pdf(file_path):
                logger.info(f"Processing scanned PDF: {file_path}")
                text = self.extract_text_from_pdf_images(file_path)
                return text, True
            else:
                # Regular PDF with text - let the normal parser handle it
                logger.info(f"PDF has extractable text, using normal parser")
                return "", False
        
        # Other files - not for OCR
        else:
            return "", False
    
    def extract_text_from_base64_image(self, base64_string: str) -> str:
        """
        Extract text from base64 encoded image
        Args:
            base64_string: Base64 encoded image
        Returns:
            Extracted text
        """
        try:
            # Decode base64
            img_data = base64.b64decode(base64_string)
            img = Image.open(io.BytesIO(img_data))
            
            # Extract text
            return self.extract_text_from_image(img)
            
        except Exception as e:
            logger.error(f"Failed to process base64 image: {e}")
            return ""


# Global OCR processor instance
_global_ocr_processor = None

def get_ocr_processor(languages: List[str] = ['en']) -> Optional[OCRProcessor]:
    """Get global OCR processor instance"""
    global _global_ocr_processor
    
    if not OCR_AVAILABLE:
        logger.warning("No OCR library available")
        return None
    
    if _global_ocr_processor is None:
        _global_ocr_processor = OCRProcessor(languages=languages)
    
    return _global_ocr_processor

def process_image_or_pdf(file_path: str) -> Tuple[str, bool]:
    """
    Quick function to process image or PDF file
    Returns: (extracted_text, was_ocr_used)
    """
    processor = get_ocr_processor()
    if processor:
        return processor.process_file(file_path)
    return "", False