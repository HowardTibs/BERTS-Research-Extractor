import re
import nltk
from nltk.tokenize import sent_tokenize
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure NLTK resources are downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class TextProcessor:
    """Process and segment extracted text for section classification."""
    
    def __init__(self, window_size=3):
        """Initialize text processor.
        
        Args:
            window_size (int): Number of paragraphs to include in context window
        """
        self.window_size = window_size
        logger.info(f"Initialized TextProcessor with window size: {window_size}")
        
        # Compile common section header patterns
        self.section_patterns = [
            # Standard numbered sections (1. Introduction, 2. Methodology)
            re.compile(r'^(?:\d+\.)?\s*(?:introduction|abstract|methodology|conclusion|discussion|results|related\s+work|background|evaluation|implementation|experiments?|references)s?\s*$', re.IGNORECASE),
            # Roman numeral sections (I. Introduction, II. Methodology)
            re.compile(r'^(?:[IVX]+\.)?\s*(?:introduction|abstract|methodology|conclusion|discussion|results|related\s+work|background|evaluation|implementation|experiments?|references)s?\s*$', re.IGNORECASE),
            # ACM/IEEE style sections
            re.compile(r'^(?:\d+\s+)?(?:INTRODUCTION|ABSTRACT|METHODOLOGY|CONCLUSION|DISCUSSION|RESULTS|RELATED\s+WORK|BACKGROUND|EVALUATION|IMPLEMENTATION|EXPERIMENTS?|REFERENCES)S?\s*$'),
        ]
    
    def preprocess(self, text):
        """Preprocess extracted text for better section identification.
        
        Args:
            text (str): Raw text from document
            
        Returns:
            list: List of processed paragraph objects
        """
        logger.info("Preprocessing text")
        
        # Clean text
        text = self._clean_text(text)
        
        # Split into paragraphs
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
        logger.info(f"Split text into {len(paragraphs)} paragraphs")
        
        # Process each paragraph
        processed_paras = []
        for i, p in enumerate(paragraphs):
            # Detect if paragraph is likely a section header
            is_header = self._is_likely_header(p)
            
            processed_paras.append({
                'id': i,
                'text': p,
                'is_potential_header': is_header,
                'length': len(p),
                'line_position': i / len(paragraphs)  # Normalized position in document
            })
        
        logger.info(f"Identified {sum(1 for p in processed_paras if p['is_potential_header'])} potential headers")
        return processed_paras
    
    def _clean_text(self, text):
        """Clean text by removing control characters, excessive whitespace, etc.
        
        Args:
            text (str): Raw text
            
        Returns:
            str: Cleaned text
        """
        # Replace control characters
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
        
        # Normalize whitespace (not removing line breaks)
        text = re.sub(r' +', ' ', text)
        
        # Normalize line breaks
        text = re.sub(r'\r\n', '\n', text)
        text = re.sub(r'\r', '\n', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()
    
    def _is_likely_header(self, text):
        """Check if text is likely a section header.
        
        Args:
            text (str): Text to check
            
        Returns:
            bool: True if likely a header, False otherwise
        """
        # Check length (headers are typically short)
        if len(text) > 150:
            return False
        
        # Check for common section header patterns
        for pattern in self.section_patterns:
            if pattern.match(text.strip()):
                return True
        
        # Check for uppercase headers
        if text.isupper() and len(text) < 50:
            return True
        
        # Check for numbered or lettered headers
        if re.match(r'^[0-9A-Z][0-9A-Z]?\.', text) and len(text) < 100:
            return True
        
        return False
    
    def create_classification_windows(self, processed_paras):
        """Create text windows for classification.
        
        Args:
            processed_paras (list): List of processed paragraph objects
            
        Returns:
            list: List of windows for classification
        """
        windows = []
        
        for i, para in enumerate(processed_paras):
            # Start and end indices for window with current paragraph in center
            start = max(0, i - self.window_size)
            end = min(len(processed_paras), i + self.window_size + 1)
            
            # Get surrounding paragraphs for context
            context_before = " ".join([p['text'] for p in processed_paras[start:i]])
            current_text = para['text']
            context_after = " ".join([p['text'] for p in processed_paras[i+1:end]])
            
            # Combine with separators
            window_text = f"{context_before} [SEP] {current_text} [SEP] {context_after}"
            
            # Create window object
            window = {
                'id': para['id'],
                'text': para['text'],
                'window_text': window_text,
                'is_potential_header': para['is_potential_header'],
                'position': para['line_position'],
                'length': para['length']
            }
            
            windows.append(window)
        
        logger.info(f"Created {len(windows)} classification windows")
        return windows
    
    def segment_into_sections(self, processed_paras, section_labels):
        """Segment document into sections based on classification results.
        
        Args:
            processed_paras (list): List of processed paragraph objects
            section_labels (list): List of section labels for each paragraph
            
        Returns:
            dict: Dictionary of extracted sections
        """
        logger.info("Segmenting document into sections")
        sections = {}
        current_section = "unknown"
        current_text = []
        
        for para, label in zip(processed_paras, section_labels):
            # Check for section transition
            if para['is_potential_header'] or label != current_section:
                # Save accumulated text from previous section
                if current_text and current_section != "unknown":
                    section_content = " ".join(current_text)
                    if current_section in sections:
                        sections[current_section] += " " + section_content
                    else:
                        sections[current_section] = section_content
                
                # Start new section
                current_section = label
                current_text = [para['text']]
            else:
                # Continue current section
                current_text.append(para['text'])
        
        # Add the last section
        if current_text and current_section != "unknown":
            section_content = " ".join(current_text)
            if current_section in sections:
                sections[current_section] += " " + section_content
            else:
                sections[current_section] = section_content
        
        logger.info(f"Extracted {len(sections)} sections: {', '.join(sections.keys())}")
        return sections