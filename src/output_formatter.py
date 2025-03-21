import os
import csv
import json
import pandas as pd
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OutputFormatter:
    """Format and export extracted sections."""
    
    def __init__(self, output_dir='./output'):
        """
        Initialize output formatter.
        
        Args:
            output_dir (str): Directory to save outputs
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Initialized OutputFormatter with output directory: {output_dir}")
    
    def format_preview(self, sections, max_length=100):
        """
        Format sections for preview display.
        
        Args:
            sections (dict): Dictionary of section name to content
            max_length (int): Maximum length for preview text
            
        Returns:
            dict: Dictionary of formatted preview sections
        """
        preview = {}
        
        for section, content in sections.items():
            if content and len(content) > max_length:
                preview[section] = content[:max_length] + "..."
            else:
                preview[section] = content
        
        return preview
    
    def export_csv(self, sections, file_path, document_name=None):
        """
        Export sections to CSV file (single line format).
        
        Args:
            sections (dict): Dictionary of section name to content
            file_path (str): Path to save CSV file
            document_name (str): Name of source document
            
        Returns:
            str: Path to saved CSV file
        """
        if document_name is None:
            document_name = f"doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Create a single row for the document
        row = {'document_name': document_name}
        row.update(sections)
        
        # Clean the content to ensure it's CSV-friendly
        for key, value in row.items():
            if isinstance(value, str):
                # Replace newlines, tabs, and quotes
                row[key] = value.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
                # Replace multiple spaces with a single space
                row[key] = ' '.join(row[key].split())
        
        # Get all field names
        fieldnames = ['document_name'] + list(sections.keys())
        
        # Write to CSV
        with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(row)
        
        logger.info(f"Exported {len(sections)} sections to {file_path}")
        return file_path
    
    def export_json(self, sections, file_path, document_name=None, metadata=None):
        """
        Export sections to JSON file.
        
        Args:
            sections (dict): Dictionary of section name to content
            file_path (str): Path to save JSON file
            document_name (str): Name of source document
            metadata (dict): Additional metadata
            
        Returns:
            str: Path to saved JSON file
        """
        if document_name is None:
            document_name = f"doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Create output object
        output = {
            'document_name': document_name,
            'sections': sections,
            'extraction_date': datetime.now().isoformat()
        }
        
        # Add metadata if provided
        if metadata:
            output['metadata'] = metadata
        
        # Write to JSON
        with open(file_path, 'w', encoding='utf-8') as jsonfile:
            json.dump(output, jsonfile, indent=2, ensure_ascii=False)
        
        logger.info(f"Exported {len(sections)} sections to {file_path}")
        return file_path
    
    def combine_csv_files(self, csv_files, output_file):
        """
        Combine multiple CSV files into a single CSV file.
        
        Args:
            csv_files (list): List of CSV file paths
            output_file (str): Path to save combined CSV file
            
        Returns:
            str: Path to saved combined CSV file
        """
        if not csv_files:
            logger.warning("No CSV files provided for combining")
            return None
        
        # Read all CSV files
        dfs = []
        
        for file in csv_files:
            try:
                df = pd.read_csv(file)
                dfs.append(df)
            except Exception as e:
                logger.error(f"Error reading {file}: {str(e)}")
        
        if not dfs:
            logger.error("No valid CSV files could be read")
            return None
        
        # Combine dataframes
        combined_df = pd.concat(dfs, ignore_index=True)
        
        # Save combined dataframe
        combined_df.to_csv(output_file, index=False)
        logger.info(f"Combined {len(dfs)} CSV files into {output_file}")
        
        return output_file
    
    def generate_stats(self, sections, document_name=None):
        """
        Generate statistics for extracted sections.
        
        Args:
            sections (dict): Dictionary of section name to content
            document_name (str): Name of source document
            
        Returns:
            dict: Dictionary of section statistics
        """
        stats = {
            'document_name': document_name or 'Unknown',
            'total_sections': len(sections),
            'section_stats': {}
        }
        
        total_words = 0
        total_chars = 0
        
        for section, content in sections.items():
            # Count words and characters
            words = len(content.split())
            chars = len(content)
            
            # Add to totals
            total_words += words
            total_chars += chars
            
            # Add to section stats
            stats['section_stats'][section] = {
                'word_count': words,
                'char_count': chars,
                'word_percentage': 0  # Will be calculated after total is known
            }
        
        # Calculate percentages
        stats['total_words'] = total_words
        stats['total_chars'] = total_chars
        
        if total_words > 0:
            for section in stats['section_stats']:
                stats['section_stats'][section]['word_percentage'] = (
                    stats['section_stats'][section]['word_count'] / total_words * 100
                )
        
        return stats