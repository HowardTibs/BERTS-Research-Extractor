import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import os
import re
import numpy as np
from docx import Document
import subprocess
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Process various document formats and extract text."""
    
    def __init__(self, ocr_lang='eng'):
        """Initialize document processor.
        
        Args:
            ocr_lang (str): Language for OCR processing
        """
        self.ocr_lang = ocr_lang
        logger.info(f"Initialized DocumentProcessor with OCR language: {ocr_lang}")
    
    def process(self, file_path):
        """Process document based on its type and extract text.
        
        Args:
            file_path (str): Path to the document file
            
        Returns:
            str: Extracted text from the document
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_extension = os.path.splitext(file_path)[1].lower()
        logger.info(f"Processing file: {file_path} with extension {file_extension}")
        
        # PDF Processing
        if file_extension == '.pdf':
            return self._process_pdf(file_path)
        # Image Processing (for scanned documents)
        elif file_extension in ['.jpg', '.jpeg', '.png', '.tiff', '.tif']:
            return self._process_image(file_path)
        # Word document processing
        elif file_extension in ['.doc', '.docx']:
            return self._process_word(file_path)
        # Plain text
        elif file_extension == '.txt':
            return self._process_text(file_path)
        # Other formats
        else:
            logger.warning(f"Unsupported file format: {file_extension}")
            try:
                return self._process_text(file_path)
            except Exception as e:
                logger.error(f"Error processing file: {str(e)}")
                return ""
    
    def _process_pdf(self, file_path):
        """Extract text from PDF, handling both digital and scanned PDFs.
        
        Args:
            file_path (str): Path to PDF file
            
        Returns:
            str: Extracted text
        """
        logger.info(f"Opening PDF: {file_path}")
        doc = fitz.open(file_path)
        text = ""
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            logger.debug(f"Processing page {page_num+1}/{len(doc)}")
            
            # Try to extract text directly first
            page_text = page.get_text()
            
            # If minimal text is extracted, it might be a scanned page
            if len(page_text.strip()) < 20:  # Arbitrary threshold
                logger.info(f"Page {page_num+1} appears to be scanned. Using OCR.")
                # Get page as image and use OCR
                pix = page.get_pixmap()
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                page_text = pytesseract.image_to_string(img, lang=self.ocr_lang)
            
            text += page_text + "\n\n"
        
        doc.close()
        logger.info(f"Extracted {len(text)} characters from PDF")
        return text
    
    def _process_image(self, file_path):
        """Process scanned document using OCR.
        
        Args:
            file_path (str): Path to image file
            
        Returns:
            str: Extracted text
        """
        logger.info(f"Processing image with OCR: {file_path}")
        img = Image.open(file_path)
        text = pytesseract.image_to_string(img, lang=self.ocr_lang)
        logger.info(f"Extracted {len(text)} characters from image")
        return text
    
    def _process_word(self, file_path):
        """Process Word documents.
        
        Args:
            file_path (str): Path to Word document
            
        Returns:
            str: Extracted text
        """
        logger.info(f"Processing Word document: {file_path}")
        
        # Use python-docx for .docx files
        if file_path.endswith('.docx'):
            try:
                doc = Document(file_path)
                text = '\n'.join([para.text for para in doc.paragraphs])
                logger.info(f"Extracted {len(text)} characters from DOCX")
                return text
            except Exception as e:
                logger.error(f"Error processing DOCX: {str(e)}")
                # Fallback to OCR if needed
                return self._process_as_image(file_path)
        
        # For .doc files, convert to text (requires antiword)
        else:
            try:
                text = subprocess.check_output(['antiword', file_path]).decode('utf-8')
                logger.info(f"Extracted {len(text)} characters using antiword")
                return text
            except Exception as e:
                logger.error(f"Error using antiword: {str(e)}")
                # Fallback to OCR
                return self._process_as_image(file_path)
    
    def _process_text(self, file_path):
        """Process plain text file.
        
        Args:
            file_path (str): Path to text file
            
        Returns:
            str: Extracted text
        """
        logger.info(f"Processing text file: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
                logger.info(f"Read {len(text)} characters from text file")
                return text
        except UnicodeDecodeError:
            # Try different encodings
            for encoding in ['latin-1', 'iso-8859-1', 'cp1252']:
                try:
                    with open(file_path, 'r', encoding=encoding) as file:
                        text = file.read()
                        logger.info(f"Read {len(text)} characters using {encoding} encoding")
                        return text
                except UnicodeDecodeError:
                    continue
            
            logger.error("Failed to decode text file with multiple encodings")
            return ""
    
    def _process_as_image(self, file_path):
        """Fallback method to process document as image using OCR.
        
        Args:
            file_path (str): Path to document
            
        Returns:
            str: Extracted text
        """
        logger.info(f"Attempting to process {file_path} as image using OCR")
        try:
            # Convert first page to image
            if file_path.endswith('.pdf'):
                doc = fitz.open(file_path)
                page = doc.load_page(0)
                pix = page.get_pixmap()
                img = Image.open(io.BytesIO(pix.tobytes("png")))
            else:
                # For non-PDF, use external tool to convert to image
                # This is a placeholder - implementation depends on available tools
                return "Document conversion not supported"
            
            # Apply OCR
            text = pytesseract.image_to_string(img, lang=self.ocr_lang)
            logger.info(f"Extracted {len(text)} characters via OCR fallback")
            return text
        
        except Exception as e:
            logger.error(f"Error in OCR fallback: {str(e)}")
            return "Error processing document"