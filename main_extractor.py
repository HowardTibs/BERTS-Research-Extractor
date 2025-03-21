import os
import logging
import argparse
import torch
from datetime import datetime
from pathlib import Path
import json
import time

# Import project modules
from src.document_processor import DocumentProcessor
from src.text_processor import TextProcessor
from src.section_classifier import (
    DistilBERTClassifier,
    TinyBERTClassifier,
    ALBERTClassifier,
    EnsembleClassifier
)
from src.output_formatter import OutputFormatter
from src.performance_monitor import PerformanceMonitor
from src.model_evaluator import ModelEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('extraction.log')
    ]
)
logger = logging.getLogger(__name__)

class ResearchExtractor:
    """Main application for extracting sections from research papers."""
    
    def __init__(self, models_dir='./models', output_dir='./output', metrics_dir='./data/metrics'):
        """
        Initialize research extractor.
        
        Args:
            models_dir (str): Directory containing trained models
            output_dir (str): Directory to save outputs
            metrics_dir (str): Directory to save metrics
        """
        self.models_dir = models_dir
        self.output_dir = output_dir
        self.metrics_dir = metrics_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(metrics_dir, exist_ok=True)
        
        # Initialize performance monitoring
        self.performance_monitor = PerformanceMonitor(metrics_dir=metrics_dir)
        self.model_evaluator = ModelEvaluator(metrics_dir=metrics_dir)
        
        # Initialize components
        self.doc_processor = DocumentProcessor()
        self.text_processor = TextProcessor()
        self.output_formatter = OutputFormatter(output_dir)
        
        # Load models
        self.models = self._load_models(models_dir)
        if self.models:
            self.classifier = EnsembleClassifier(self.models)
            logger.info(f"Initialized ensemble classifier with {len(self.models)} models")
        else:
            logger.error("No models could be loaded. Extraction will be limited.")
        
        logger.info("Research Extractor initialized")
    
    def _load_models(self, models_dir):
        """
        Load trained classification models.
        
        Args:
            models_dir (str): Directory containing trained models
            
        Returns:
            list: List of loaded models
        """
        models = []
        
        # Define model directories
        model_paths = {
            'distilbert': os.path.join(models_dir, 'distilbert_sections'),
            'tinybert': os.path.join(models_dir, 'tinybert_sections'),
            'albert': os.path.join(models_dir, 'albert_sections')
        }
        
        # Check if device supports cuda
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        # Load DistilBERT model
        if os.path.exists(model_paths['distilbert']):
            try:
                logger.info("Loading DistilBERT model...")
                distilbert = DistilBERTClassifier('distilbert', model_path=model_paths['distilbert'])
                models.append(distilbert)
                logger.info("DistilBERT model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading DistilBERT model: {str(e)}")
        
        # Load TinyBERT model
        if os.path.exists(model_paths['tinybert']):
            try:
                logger.info("Loading TinyBERT model...")
                tinybert = TinyBERTClassifier('tinybert', model_path=model_paths['tinybert'])
                models.append(tinybert)
                logger.info("TinyBERT model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading TinyBERT model: {str(e)}")
        
        # Load ALBERT model
        if os.path.exists(model_paths['albert']):
            try:
                logger.info("Loading ALBERT model...")
                albert = ALBERTClassifier('albert', model_path=model_paths['albert'])
                models.append(albert)
                logger.info("ALBERT model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading ALBERT model: {str(e)}")
        
        logger.info(f"Loaded {len(models)} models")
        return models
    
    def extract_sections(self, file_path, extract_sections=None, model_type=None):
        """
        Extract sections from a research paper.
        
        Args:
            file_path (str): Path to research paper file
            extract_sections (list): List of sections to extract (None = all)
            model_type (str): Type of model to use for extraction
            
        Returns:
            dict: Dictionary of extracted sections
        """
        # Get document name for tracking
        document_name = os.path.basename(file_path)
        
        # Start performance monitoring
        self.performance_monitor.start_monitoring(document_name)
        
        logger.info(f"Processing file: {file_path}")
        
        # 1. Process document and extract text
        self.performance_monitor.start_stage("document_processing")
        text = self.doc_processor.process(file_path)
        self.performance_monitor.end_stage()
        
        if not text:
            logger.error("Failed to extract text from document")
            # End monitoring and save metrics
            metrics = self.performance_monitor.end_monitoring()
            self.performance_monitor.save_metrics(metrics, model_type)
            return {}
        
        logger.info(f"Extracted {len(text)} characters of text")
        
        # 2. Preprocess text for classification
        self.performance_monitor.start_stage("text_processing")
        processed_paras = self.text_processor.preprocess(text)
        logger.info(f"Preprocessed text into {len(processed_paras)} paragraphs")
        
        # Create classification windows
        windows = self.text_processor.create_classification_windows(processed_paras)
        self.performance_monitor.end_stage()
        
        # 3. Classify sections
        self.performance_monitor.start_stage("classification")
        if self.models:
            # Get specific model if requested
            if model_type and model_type != 'ensemble':
                selected_model = None
                for model in self.models:
                    if model.model_type.lower() == model_type.lower():
                        selected_model = model
                        break
                
                if selected_model:
                    # Extract window text for classification
                    window_texts = [window['window_text'] for window in windows]
                    
                    # Run classification with selected model
                    section_labels = selected_model.predict(window_texts)
                    logger.info(f"Classified {len(section_labels)} paragraphs using {model_type} model")
                else:
                    logger.warning(f"Requested model type '{model_type}' not found, using ensemble")
                    # Extract window text for classification
                    window_texts = [window['window_text'] for window in windows]
                    
                    # Run classification with ensemble
                    section_labels = self.classifier.predict(window_texts)
                    logger.info(f"Classified {len(section_labels)} paragraphs using ensemble")
            else:
                # Extract window text for classification
                window_texts = [window['window_text'] for window in windows]
                
                # Run classification with ensemble
                section_labels = self.classifier.predict(window_texts)
                logger.info(f"Classified {len(section_labels)} paragraphs using ensemble")
        else:
            # Fallback: use simple heuristics if no models are available
            section_labels = self._fallback_classification(processed_paras)
            logger.info(f"Used fallback classification for {len(section_labels)} paragraphs")
        self.performance_monitor.end_stage()
        
        # 4. Segment text into sections
        self.performance_monitor.start_stage("output_formatting")
        sections = self.text_processor.segment_into_sections(processed_paras, section_labels)
        logger.info(f"Extracted {len(sections)} sections")
        
        # 5. Filter sections if specific ones were requested
        if extract_sections:
            filtered_sections = {k: v for k, v in sections.items() if k in extract_sections}
            logger.info(f"Filtered to {len(filtered_sections)} requested sections")
            result = filtered_sections
        else:
            result = sections
        
        self.performance_monitor.end_stage()
        
        # End monitoring and save metrics
        metrics = self.performance_monitor.end_monitoring()
        self.performance_monitor.save_metrics(metrics, model_type or 'ensemble')
        
        return result
    
    def _fallback_classification(self, processed_paras):
        """
        Fallback classification method using heuristics.
        
        Args:
            processed_paras (list): List of processed paragraphs
            
        Returns:
            list: List of section labels
        """
        labels = []
        current_label = "unknown"
        
        # Common section patterns
        title_patterns = ["title", "research", "study", "analysis"]
        abstract_patterns = ["abstract", "summary"]
        intro_patterns = ["introduction", "background", "overview"]
        methods_patterns = ["method", "methodology", "approach", "procedure"]
        results_patterns = ["result", "finding", "outcome"]
        discussion_patterns = ["discussion", "implication"]
        conclusion_patterns = ["conclusion", "summary"]
        
        # Process each paragraph
        for para in processed_paras:
            text = para['text'].lower()
            is_header = para['is_potential_header']
            
            # Detect different section types
            if is_header:
                if any(pattern in text for pattern in title_patterns) and len(text) < 150:
                    current_label = "title"
                elif any(pattern in text for pattern in abstract_patterns):
                    current_label = "abstract"
                elif any(pattern in text for pattern in intro_patterns):
                    current_label = "introduction"
                elif any(pattern in text for pattern in methods_patterns):
                    current_label = "methodology"
                elif any(pattern in text for pattern in results_patterns):
                    current_label = "results"
                elif any(pattern in text for pattern in discussion_patterns):
                    current_label = "discussion"
                elif any(pattern in text for pattern in conclusion_patterns):
                    current_label = "conclusion"
                elif "references" in text or "bibliography" in text:
                    current_label = "references"
                elif "keywords" in text:
                    current_label = "keywords"
                elif "author" in text:
                    current_label = "authors"
            
            labels.append(current_label)
        
        return labels
    
    def process_file(self, file_path, output_format='csv', extract_sections=None, model_type=None):
        """
        Process a single file and save extracted sections.
        
        Args:
            file_path (str): Path to research paper file
            output_format (str): Output format ('csv' or 'json')
            extract_sections (list): List of sections to extract (None = all)
            model_type (str): Type of model to use for extraction
            
        Returns:
            tuple: (extracted sections, output file path)
        """
        start_time = time.time()
        
        # Extract sections
        sections = self.extract_sections(file_path, extract_sections, model_type)
        
        extraction_time = time.time() - start_time
        logger.info(f"Total extraction time: {extraction_time:.4f} seconds")
        
        if not sections:
            logger.warning(f"No sections were extracted from {file_path}")
            return {}, None
        
        # Create output filename
        file_name = os.path.basename(file_path)
        base_name = os.path.splitext(file_name)[0]
        document_name = base_name
        
        # Generate stats
        stats = self.output_formatter.generate_stats(sections, document_name)
        
        # Export based on format
        if output_format.lower() == 'json':
            output_path = os.path.join(self.output_dir, f"{base_name}_extracted.json")
            self.output_formatter.export_json(sections, output_path, document_name, stats)
        else:
            output_path = os.path.join(self.output_dir, f"{base_name}_extracted.csv")
            self.output_formatter.export_csv(sections, output_path, document_name)
        
        logger.info(f"Processed {file_path} and saved to {output_path}")
        return sections, output_path
    
    def process_directory(self, directory_path, output_format='csv', extract_sections=None, recursive=False, model_type=None):
        """
        Process all files in a directory.
        
        Args:
            directory_path (str): Path to directory containing research papers
            output_format (str): Output format ('csv' or 'json')
            extract_sections (list): List of sections to extract (None = all)
            recursive (bool): Process subdirectories recursively
            model_type (str): Type of model to use for extraction
            
        Returns:
            list: List of processed file paths
        """
        logger.info(f"Processing directory: {directory_path}")
        
        # Get all files in directory
        if recursive:
            file_paths = []
            for root, _, files in os.walk(directory_path):
                for file in files:
                    file_paths.append(os.path.join(root, file))
        else:
            file_paths = [os.path.join(directory_path, f) for f in os.listdir(directory_path) 
                         if os.path.isfile(os.path.join(directory_path, f))]
        
        logger.info(f"Found {len(file_paths)} files")
        
        # Process each file
        processed_files = []
        csv_outputs = []
        
        total_start_time = time.time()
        
        for file_path in file_paths:
            try:
                _, output_path = self.process_file(file_path, output_format, extract_sections, model_type)
                if output_path:
                    processed_files.append(file_path)
                    if output_format.lower() == 'csv':
                        csv_outputs.append(output_path)
            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")
        
        total_time = time.time() - total_start_time
        logger.info(f"Total processing time for all files: {total_time:.4f} seconds")
        
        # Combine CSV files if needed
        if output_format.lower() == 'csv' and len(csv_outputs) > 1:
            combined_path = os.path.join(self.output_dir, f"combined_extracted_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            self.output_formatter.combine_csv_files(csv_outputs, combined_path)
        
        logger.info(f"Successfully processed {len(processed_files)} files")
        return processed_files
    
    def evaluate_model(self, test_file, model_type=None):
        """
        Evaluate model performance on test data.
        
        Args:
            test_file (str): Path to test data file
            model_type (str): Type of model to evaluate (None = ensemble)
            
        Returns:
            dict: Evaluation metrics
        """
        import pandas as pd
        
        logger.info(f"Evaluating model on {test_file}")
        
        # Load test data
        try:
            test_df = pd.read_csv(test_file)
            test_texts = test_df['text'].tolist()
            test_labels = test_df['label'].tolist()
        except Exception as e:
            logger.error(f"Error loading test data: {str(e)}")
            return None
        
        # Select model for evaluation
        if model_type and model_type != 'ensemble':
            selected_model = None
            for model in self.models:
                if model.model_type.lower() == model_type.lower():
                    selected_model = model
                    break
            
            if not selected_model:
                logger.warning(f"Model type '{model_type}' not found, using ensemble")
                selected_model = self.classifier
                model_type = 'ensemble'
        else:
            selected_model = self.classifier
            model_type = 'ensemble'
        
        # Evaluate model
        metrics, report = self.model_evaluator.evaluate_model(
            selected_model, test_texts, test_labels, model_type
        )
        
        # Create confusion matrix
        unique_labels = sorted(set(test_labels))
        pred_labels = selected_model.predict(test_texts)
        self.model_evaluator.plot_confusion_matrix(
            test_labels, pred_labels, unique_labels, model_type
        )
        
        logger.info(f"Evaluation complete for {model_type}")
        return metrics
    
    def get_available_section_types(self):
        """
        Get list of section types the model can extract.
        
        Returns:
            list: List of section types
        """
        if self.models:
            # Get from the first model's label mapping
            return list(self.models[0].id2label.values())
        else:
            # Default sections if no models are loaded
            return [
                "title", "authors", "abstract", "keywords", "introduction",
                "methodology", "results", "discussion", "conclusion", "references", "other"
            ]
    
    def get_available_models(self):
        """
        Get list of available models.
        
        Returns:
            list: List of model types
        """
        model_types = []
        if self.models:
            for model in self.models:
                model_types.append(model.model_type)
            model_types.append('ensemble')
        return model_types

def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Extract sections from research papers")
    parser.add_argument("--input", required=True, help="Input file or directory path")
    parser.add_argument("--output_dir", default="./output", help="Output directory")
    parser.add_argument("--models_dir", default="./models", help="Directory with trained models")
    parser.add_argument("--metrics_dir", default="./data/metrics", help="Directory for metrics data")
    parser.add_argument("--format", choices=["csv", "json"], default="csv", help="Output format")
    parser.add_argument("--sections", nargs="+", help="Specific sections to extract")
    parser.add_argument("--recursive", action="store_true", help="Process directories recursively")
    parser.add_argument("--model", help="Specific model to use (distilbert, tinybert, albert, ensemble)")
    parser.add_argument("--evaluate", help="Path to test data for model evaluation")
    
    args = parser.parse_args()
    
    # Initialize extractor
    extractor = ResearchExtractor(
        models_dir=args.models_dir, 
        output_dir=args.output_dir,
        metrics_dir=args.metrics_dir
    )
    
    # Run evaluation if requested
    if args.evaluate:
        metrics = extractor.evaluate_model(args.evaluate, args.model)
        if metrics:
            logger.info(f"Evaluation metrics: accuracy={metrics['accuracy']:.4f}, f1_macro={metrics['f1_macro']:.4f}")
        return
    
    # Process input
    input_path = args.input
    if os.path.isfile(input_path):
        # Process single file
        sections, output_path = extractor.process_file(
            input_path, args.format, args.sections, args.model
        )
        
        if sections:
            logger.info(f"Extracted sections: {', '.join(sections.keys())}")
            logger.info(f"Output saved to: {output_path}")
        else:
            logger.error("No sections were extracted")
    
    elif os.path.isdir(input_path):
        # Process directory
        processed_files = extractor.process_directory(
            input_path, args.format, args.sections, args.recursive, args.model
        )
        logger.info(f"Processed {len(processed_files)} files")
    
    else:
        logger.error(f"Input path not found: {input_path}")

if __name__ == "__main__":
    main()