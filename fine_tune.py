import os
import argparse
import logging
import pandas as pd
import time
from datetime import datetime
from model_trainer import fine_tune_workflow
from src.model_evaluator import ModelEvaluator
from src.performance_monitor import PerformanceMonitor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('fine_tuning.log')
    ]
)
logger = logging.getLogger(__name__)

def create_sample_dataset(output_dir, num_samples=50):
    """
    Create sample dataset for fine-tuning demonstration.
    
    Args:
        output_dir (str): Directory to save sample data
        num_samples (int): Number of samples to create
        
    Returns:
        str: Path to created sample dataset
    """
    logger.info(f"Creating sample dataset with {num_samples} samples")
    
    # Ensure directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create sample data for each section type
    data = []
    
    # Title samples
    titles = [
        "Efficient Text Extraction and Classification from Research Papers",
        "A Comparative Study of Lightweight BERT Models for Document Section Classification",
        "Neural Approaches to Automatic Document Structure Analysis",
        "TinyBERT: Distilling BERT for Natural Language Understanding",
        "ALBERT: A Lite BERT for Self-supervised Learning of Language Representations",
        "Improving Document Classification with Reduced Transformer Models",
        "Extractive Information Retrieval from Scientific Publications"
    ]
    
    # Author samples
    authors = [
        "John Smith, Jane Doe, University of AI Research",
        "Zhang Wei, Maria Garcia, Advanced Computing Institute",
        "Ahmed Hassan, Sarah Johnson, Department of Computer Science",
        "Robert Chen, Lisa Wong, AI Laboratory",
        "Michael Brown, Emily Davis, Information Extraction Group"
    ]
    
    # Abstract samples
    abstracts = [
        "This paper presents a novel approach to extract structured information from research papers. We propose a lightweight system using distilled transformer models to identify and classify different sections of academic publications. Our method achieves state-of-the-art results while maintaining efficiency.",
        "We introduce a framework for automatically extracting and categorizing sections from scientific papers across multiple formats and publishers. The system uses an ensemble of compact BERT variants to ensure both accuracy and computational efficiency. Experimental results show significant improvements over previous approaches.",
        "Extracting structured information from research publications is challenging due to variations in formatting and style. In this work, we present a scalable solution that can identify key sections using minimally supervised learning. Our approach demonstrates robust performance across ACM, IEEE, and other publication formats."
    ]
    
    # Introduction samples
    introductions = [
        "The exponential growth of scientific literature has created a need for automated tools to extract structured information from research papers. Traditional methods rely on rule-based systems or template matching, which fail to generalize across different publication formats. Recent advancements in natural language processing, particularly transformer-based models, have shown promise in addressing these challenges.",
        "Scientific knowledge is primarily disseminated through research papers, which follow structured formats with sections like abstract, introduction, methodology, and results. Automatically extracting this structured information enables better indexing, searching, and knowledge discovery. However, variations in formatting across publishers and domains make this a challenging task."
    ]
    
    # Methodology samples
    methodologies = [
        "We employed a three-stage pipeline for section extraction: (1) document processing to convert various formats to plain text, (2) text segmentation to identify potential section boundaries, and (3) section classification using pre-trained language models. For classification, we fine-tuned three lightweight models: DistilBERT, TinyBERT, and ALBERT, combining them in an ensemble for improved accuracy.",
        "Our approach consists of preprocessing the document using OCR when necessary, segmenting the text into coherent blocks, and classifying each segment using lightweight transformer models. We fine-tuned the models on a dataset of 1,000 manually annotated research papers spanning multiple formats and domains."
    ]
    
    # Create dataset by combining samples
    section_types = [
        ("title", titles),
        ("authors", authors),
        ("abstract", abstracts),
        ("introduction", introductions),
        ("methodology", methodologies)
    ]
    
    # Generate samples by combining sections
    for i in range(num_samples):
        for label, samples in section_types:
            sample = samples[i % len(samples)]
            data.append({
                "text": sample,
                "label": label,
                "source": f"sample_{i//len(section_types)}"
            })
    
    # Convert to DataFrame and save
    df = pd.DataFrame(data)
    output_file = os.path.join(output_dir, "sample_training_data.csv")
    df.to_csv(output_file, index=False)
    
    logger.info(f"Created sample dataset with {len(df)} examples at {output_file}")
    return output_file

def evaluate_model_performance(models, test_file, metrics_dir):
    """
    Evaluate model performance and write detailed metrics.
    
    Args:
        models (list): List of trained models
        test_file (str): Path to test data file
        metrics_dir (str): Directory to save metrics
        
    Returns:
        dict: Evaluation results for each model
    """
    logger.info(f"Evaluating model performance on {test_file}")
    
    # Ensure metrics directory exists
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Initialize evaluator
    evaluator = ModelEvaluator(metrics_dir=metrics_dir)
    
    # Load test data
    test_df = pd.read_csv(test_file)
    test_texts = test_df['text'].tolist()
    test_labels = test_df['label'].tolist()
    
    # Initialize metrics dict
    metrics_results = {}
    
    # Evaluate each model
    for model in models:
        model_type = model.model_type
        logger.info(f"Evaluating {model_type} model")
        
        # Start performance monitoring
        performance_monitor = PerformanceMonitor(metrics_dir=metrics_dir)
        performance_monitor.start_monitoring(f"evaluate_{model_type}")
        
        # Get model metrics
        metrics, report = evaluator.evaluate_model(model, test_texts, test_labels, model_type)
        
        # Create confusion matrix
        unique_labels = sorted(set(test_labels))
        pred_labels = model.predict(test_texts)
        evaluator.plot_confusion_matrix(test_labels, pred_labels, unique_labels, model_type)
        
        # End performance monitoring
        perf_metrics = performance_monitor.end_monitoring()
        performance_monitor.save_metrics(perf_metrics, model_type)
        
        # Store metrics
        metrics_results[model_type] = metrics
    
    # Evaluate ensemble if we have multiple models
    if len(models) > 1:
        from section_classifier import EnsembleClassifier
        
        logger.info("Evaluating ensemble model")
        
        # Create ensemble
        ensemble = EnsembleClassifier(models)
        
        # Start performance monitoring
        performance_monitor = PerformanceMonitor(metrics_dir=metrics_dir)
        performance_monitor.start_monitoring("evaluate_ensemble")
        
        # Get ensemble metrics
        metrics, report = evaluator.evaluate_model(ensemble, test_texts, test_labels, "ensemble")
        
        # Create confusion matrix
        unique_labels = sorted(set(test_labels))
        pred_labels = ensemble.predict(test_texts)
        evaluator.plot_confusion_matrix(test_labels, pred_labels, unique_labels, "ensemble")
        
        # End performance monitoring
        perf_metrics = performance_monitor.end_monitoring()
        performance_monitor.save_metrics(perf_metrics, "ensemble")
        
        # Store metrics
        metrics_results["ensemble"] = metrics
    
    # Create comparison plots
    evaluator.plot_model_comparison()
    
    # Create performance comparison
    performance_monitor = PerformanceMonitor(metrics_dir=metrics_dir)
    performance_monitor.plot_performance_comparison()
    
    logger.info(f"Evaluation complete, metrics saved to {metrics_dir}")
    return metrics_results

def compare_model_sizes(models, metrics_dir):
    """
    Compare model sizes and create a summary.
    
    Args:
        models (list): List of trained models
        metrics_dir (str): Directory to save metrics
    """
    logger.info("Comparing model sizes")
    
    # Ensure metrics directory exists
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Get model sizes
    model_sizes = []
    
    for model in models:
        model_type = model.model_type
        model_dir = f"./models/{model_type}_sections"
        
        # Calculate directory size
        total_size = 0
        for dirpath, _, filenames in os.walk(model_dir):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                total_size += os.path.getsize(fp)
        
        # Convert to MB
        size_mb = total_size / (1024 * 1024)
        
        model_sizes.append({
            'model_type': model_type,
            'size_mb': size_mb
        })
    
    # Create DataFrame
    df = pd.DataFrame(model_sizes)
    
    # Save to CSV
    output_file = os.path.join(metrics_dir, 'model_sizes.csv')
    df.to_csv(output_file, index=False)
    
    # Log sizes
    logger.info("Model sizes:")
    for _, row in df.iterrows():
        logger.info(f"  {row['model_type']}: {row['size_mb']:.2f} MB")
    
    logger.info(f"Model size comparison saved to {output_file}")

def main():
    """Main function for fine-tuning script."""
    parser = argparse.ArgumentParser(description="Fine-tune models for research paper section extraction")
    parser.add_argument("--input_dir", help="Directory with annotated papers")
    parser.add_argument("--output_dir", default="./output", help="Output directory")
    parser.add_argument("--metrics_dir", default="./data/metrics", help="Directory for metrics data")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--create_sample", action="store_true", help="Create sample dataset for demonstration")
    parser.add_argument("--sample_size", type=int, default=50, help="Number of samples to create")
    
    args = parser.parse_args()
    
    # Record start time
    start_time = time.time()
    
    # Set up directories
    output_dir = args.output_dir
    metrics_dir = args.metrics_dir
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    
    # If no input directory specified or create_sample flag is set, create sample dataset
    if args.create_sample or not args.input_dir:
        logger.info("Creating sample dataset for demonstration")
        sample_dir = os.path.join(output_dir, "sample_data")
        input_dir = create_sample_dataset(sample_dir, args.sample_size)
    else:
        input_dir = args.input_dir
    
    # Run fine-tuning workflow
    logger.info("Starting fine-tuning workflow")
    
    # Initialize performance monitor for the whole process
    performance_monitor = PerformanceMonitor(metrics_dir=metrics_dir)
    performance_monitor.start_monitoring("fine_tuning_workflow")
    
    # Run fine-tuning
    models = fine_tune_workflow(
        input_data_dir=input_dir,
        output_dir=output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    # Record fine-tuning time
    fine_tuning_time = time.time() - start_time
    logger.info(f"Fine-tuning completed in {fine_tuning_time:.2f} seconds")
    
    # Evaluate models and collect metrics
    test_file = os.path.join(output_dir, "data", "training_data_test.csv")
    if os.path.exists(test_file) and models:
        logger.info("Evaluating models on test data")
        evaluate_model_performance(models, test_file, metrics_dir)
        
        # Compare model sizes
        compare_model_sizes(models, metrics_dir)
    
    # End performance monitoring
    metrics = performance_monitor.end_monitoring()
    performance_monitor.save_metrics(metrics, "fine_tuning_workflow")
    
    # Log total time
    total_time = time.time() - start_time
    logger.info(f"Total process completed in {total_time:.2f} seconds")

if __name__ == "__main__":
    main()