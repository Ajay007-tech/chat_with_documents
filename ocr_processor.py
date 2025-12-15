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
    logger.warning("EasyOCR not installed. Install with: pip install easyocr")

# PDF handling
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
    logger.info("PyMuPDF available for PDF processing")
except ImportError:
    PYMUPDF_AVAILABLE = False
    logger.warning("PyMuPDF not available. Install with: pip install PyMuPDF")


class OCRProcessor:
    """Process images and scanned PDFs to extract text"""
    
    def __init__(self, backend: Optional[str] = None, languages: List[str] = None):
        """
        Initialize OCR processor
        Args:
            backend: 'easyocr' or None for auto-detect
            languages: List of language codes for OCR (default: ['en', 'ch_sim'])
        """
        self.backend = backend or OCR_BACKEND
        self.languages = languages or ['en']  # English and Simplified Chinese
        self.ocr_reader = None
        
        if not OCR_AVAILABLE:
            logger.warning("No OCR library available. Install easyocr for OCR support")
            return
        
        # Initialize OCR reader based on backend
        if self.backend == "easyocr":
            try:
                # Use GPU if available, else CPU
                import torch
                use_gpu = torch.cuda.is_available()
                self.ocr_reader = easyocr.Reader(self.languages, gpu=use_gpu)
                logger.info(f"EasyOCR initialized with languages: {self.languages}, GPU: {use_gpu}")
            except Exception as e:
                logger.error(f"Failed to initialize EasyOCR: {e}")
                self.backend = None
                OCR_AVAILABLE = False
    
    def extract_text_from_image(self, image: Union[str, Image.Image, np.ndarray]) -> str:
        """
        Extract text from an image
        Args:
            image: Path to image, PIL Image, or numpy array
        Returns:
            Extracted text
        """
        if not OCR_AVAILABLE or not self.backend or not self.ocr_reader:
            logger.warning("OCR not available")
            return "[OCR not available - install easyocr]"
        
        try:
            # Convert to appropriate format for EasyOCR
            if isinstance(image, str):
                if os.path.exists(image):
                    # EasyOCR can work directly with file paths
                    logger.info(f"Processing image file: {image}")
                    results = self.ocr_reader.readtext(image)
                else:
                    logger.error(f"Image file not found: {image}")
                    return ""
            elif isinstance(image, Image.Image):
                # Convert PIL Image to numpy array
                img_array = np.array(image)
                results = self.ocr_reader.readtext(img_array)
            elif isinstance(image, np.ndarray):
                results = self.ocr_reader.readtext(image)
            else:
                logger.error(f"Unsupported image type: {type(image)}")
                return ""
            
            # Extract text from results
            text_parts = []
            for detection in results:
                # Each result is (bbox, text, confidence)
                if len(detection) >= 2:
                    text_parts.append(detection[1])
            
            text = "\n".join(text_parts)
            
            if text:
                logger.info(f"OCR extracted {len(text)} characters")
            else:
                logger.info("No text detected in image")
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            import traceback
            traceback.print_exc()
            return f"[OCR failed: {str(e)}]"
    
    def is_scanned_pdf(self, pdf_path: str, text_threshold: int = 100) -> bool:
        """
        Check if a PDF is scanned (image-based) or has text
        Args:
            pdf_path: Path to PDF file
            text_threshold: Minimum characters to consider a page as having text
        Returns:
            True if scanned/image PDF, False if has extractable text
        """
        if not PYMUPDF_AVAILABLE:
            logger.warning("PyMuPDF not available, assuming PDF might be scanned")
            return True
        
        try:
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            pages_to_check = min(3, total_pages)  # Check first 3 pages
            
            total_text_length = 0
            for page_num in range(pages_to_check):
                page = doc[page_num]
                text = page.get_text().strip()
                total_text_length += len(text)
                
                # If any page has substantial text, it's not scanned
                if len(text) > text_threshold:
                    doc.close()
                    logger.info(f"PDF has extractable text (page {page_num + 1} has {len(text)} chars)")
                    return False
            
            doc.close()
            
            # Average text per checked page
            avg_text = total_text_length / pages_to_check if pages_to_check > 0 else 0
            is_scanned = avg_text < text_threshold
            
            if is_scanned:
                logger.info(f"PDF appears to be scanned (avg {avg_text:.0f} chars per page)")
            
            return is_scanned
            
        except Exception as e:
            logger.error(f"Error checking PDF type: {e}")
            return False  # Assume not scanned if we can't check
    
    def extract_text_from_pdf_images(self, pdf_path: str, max_pages: Optional[int] = 20) -> str:
        """
        Extract text from scanned PDF using OCR
        Args:
            pdf_path: Path to PDF file
            max_pages: Maximum number of pages to process (None for all)
        Returns:
            Extracted text
        """
        if not OCR_AVAILABLE or not self.ocr_reader:
            return "[OCR not available for scanned PDF - install easyocr]"
        
        if not PYMUPDF_AVAILABLE:
            return "[PyMuPDF not available - install PyMuPDF to process PDFs]"
        
        extracted_text = []
        
        try:
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            pages_to_process = min(max_pages or total_pages, total_pages)
            
            logger.info(f"Processing {pages_to_process} pages from scanned PDF: {pdf_path}")
            
            for page_num in range(pages_to_process):
                page = doc[page_num]
                
                # First try to get text (in case some pages have text)
                text = page.get_text().strip()
                
                if len(text) > 100:  # Page has text
                    extracted_text.append(f"[Page {page_num + 1}]\n{text}")
                    logger.info(f"Page {page_num + 1}: Used extracted text ({len(text)} chars)")
                else:
                    # Convert page to image for OCR
                    # Increase resolution for better OCR accuracy
                    mat = fitz.Matrix(2.0, 2.0)  # 2x scale
                    pix = page.get_pixmap(matrix=mat)
                    
                    # Convert to PIL Image
                    img_data = pix.tobytes("png")
                    img = Image.open(io.BytesIO(img_data))
                    
                    # Perform OCR
                    logger.info(f"Applying OCR to page {page_num + 1}")
                    page_text = self.extract_text_from_image(img)
                    
                    if page_text and page_text != "[OCR not available - install easyocr]":
                        extracted_text.append(f"[Page {page_num + 1} - OCR]\n{page_text}")
                        logger.info(f"Page {page_num + 1}: OCR extracted {len(page_text)} chars")
                    else:
                        logger.warning(f"Page {page_num + 1}: No text extracted")
                    
                    # Clean up
                    pix = None
                    img.close()
                
                # Log progress for long PDFs
                if (page_num + 1) % 5 == 0:
                    logger.info(f"Progress: {page_num + 1}/{pages_to_process} pages processed")
            
            doc.close()
            
            final_text = "\n\n".join(extracted_text)
            logger.info(f"Total OCR extraction: {len(final_text)} characters from {len(extracted_text)} pages")
            
            return final_text if final_text else "[No text could be extracted from the scanned PDF]"
            
        except Exception as e:
            logger.error(f"Failed to extract text from scanned PDF: {e}")
            import traceback
            traceback.print_exc()
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
        if file_ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.tif', '.webp']:
            logger.info(f"Processing image file with OCR: {file_path}")
            text = self.extract_text_from_image(file_path)
            return text, True
        
        # PDF files
        elif file_ext == '.pdf':
            # Check if it's scanned
            if self.is_scanned_pdf(file_path):
                logger.info(f"Processing scanned PDF with OCR: {file_path}")
                text = self.extract_text_from_pdf_images(file_path)
                return text, True
            else:
                # Regular PDF with text - let the normal parser handle it
                logger.info(f"PDF has extractable text, using normal parser")
                return "", False
        
        # Other files - not for OCR
        else:
            return "", False


# Global OCR processor instance
_global_ocr_processor = None

def get_ocr_processor(languages: List[str] = None) -> Optional[OCRProcessor]:
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