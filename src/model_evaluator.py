import os
import json
import numpy as np
import pandas as pd
import time
import logging
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    classification_report, 
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Evaluate model performance and store detailed metrics."""
    
    def __init__(self, metrics_dir='./data/metrics'):
        """
        Initialize model evaluator.
        
        Args:
            metrics_dir (str): Directory to store metrics data
        """
        self.metrics_dir = metrics_dir
        os.makedirs(metrics_dir, exist_ok=True)
        self.model_metrics_file = os.path.join(metrics_dir, 'model_metrics.csv')
        logger.info(f"Initialized ModelEvaluator with metrics directory: {metrics_dir}")
    
    def evaluate_model(self, model, test_data, test_labels, model_type):
        """
        Evaluate model performance on test data.
        
        Args:
            model: Trained model instance
            test_data (list): Test data texts
            test_labels (list): True labels for test data
            model_type (str): Type of model being evaluated
            
        Returns:
            dict: Evaluation metrics
        """
        logger.info(f"Evaluating {model_type} model")
        start_time = time.time()
        
        # Make predictions
        pred_labels = model.predict(test_data)
        
        # Calculate evaluation time
        eval_time = time.time() - start_time
        
        # Convert string labels to numeric if needed
        if isinstance(test_labels[0], str):
            # Get unique labels
            unique_labels = sorted(set(test_labels))
            label_map = {label: i for i, label in enumerate(unique_labels)}
            
            # Convert to numeric
            test_labels_numeric = [label_map[label] for label in test_labels]
            pred_labels_numeric = [label_map[label] if label in label_map else -1 for label in pred_labels]
        else:
            test_labels_numeric = test_labels
            pred_labels_numeric = pred_labels
        
        # Calculate metrics
        metrics = {
            'model_type': model_type,
            'accuracy': accuracy_score(test_labels, pred_labels),
            'precision_micro': precision_score(test_labels_numeric, pred_labels_numeric, average='micro', zero_division=0),
            'precision_macro': precision_score(test_labels_numeric, pred_labels_numeric, average='macro', zero_division=0),
            'recall_micro': recall_score(test_labels_numeric, pred_labels_numeric, average='micro', zero_division=0),
            'recall_macro': recall_score(test_labels_numeric, pred_labels_numeric, average='macro', zero_division=0),
            'f1_micro': f1_score(test_labels_numeric, pred_labels_numeric, average='micro', zero_division=0),
            'f1_macro': f1_score(test_labels_numeric, pred_labels_numeric, average='macro', zero_division=0),
            'eval_time_seconds': eval_time,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'num_samples': len(test_data)
        }
        
        # Create detailed classification report
        report = classification_report(test_labels, pred_labels, output_dict=True)
        
        # Log metrics
        logger.info(f"Evaluation metrics for {model_type}:")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  Precision (macro): {metrics['precision_macro']:.4f}")
        logger.info(f"  Recall (macro): {metrics['recall_macro']:.4f}")
        logger.info(f"  F1 (macro): {metrics['f1_macro']:.4f}")
        logger.info(f"  Evaluation time: {metrics['eval_time_seconds']:.4f} seconds")
        
        # Save metrics
        self.save_metrics(metrics, report, model_type)
        
        return metrics, report
    
    def save_metrics(self, metrics, report, model_type):
        """
        Save evaluation metrics to CSV and JSON.
        
        Args:
            metrics (dict): Evaluation metrics
            report (dict): Classification report
            model_type (str): Type of model
        """
        # Save metrics to CSV
        metrics_df = pd.DataFrame([metrics])
        
        if os.path.exists(self.model_metrics_file):
            # Append to existing file
            existing_df = pd.read_csv(self.model_metrics_file)
            updated_df = pd.concat([existing_df, metrics_df], ignore_index=True)
            updated_df.to_csv(self.model_metrics_file, index=False)
        else:
            # Create new file
            metrics_df.to_csv(self.model_metrics_file, index=False)
        
        # Save detailed report to JSON
        report_file = os.path.join(self.metrics_dir, f'{model_type}_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Saved evaluation metrics to {self.model_metrics_file}")
        logger.info(f"Saved detailed classification report to {report_file}")
    
    def plot_confusion_matrix(self, true_labels, pred_labels, label_names, model_type):
        """
        Create and save confusion matrix plot.
        
        Args:
            true_labels (list): True labels
            pred_labels (list): Predicted labels
            label_names (list): Names of the labels
            model_type (str): Type of model
            
        Returns:
            str: Path to saved plot
        """
        # Create confusion matrix
        cm = confusion_matrix(true_labels, pred_labels, labels=label_names)
        
        # Normalize confusion matrix
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Create plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', 
                   xticklabels=label_names, yticklabels=label_names)
        plt.title(f'Normalized Confusion Matrix - {model_type}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        # Save plot
        plot_file = os.path.join(self.metrics_dir, f'{model_type}_confusion_matrix_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        plt.savefig(plot_file)
        plt.close()
        
        logger.info(f"Saved confusion matrix plot to {plot_file}")
        return plot_file
    
    def get_latest_metrics(self):
        """
        Get the latest metrics for all models.
        
        Returns:
            pd.DataFrame: Latest metrics for each model
        """
        if not os.path.exists(self.model_metrics_file):
            logger.warning(f"Metrics file not found: {self.model_metrics_file}")
            return pd.DataFrame()
        
        # Read metrics file
        metrics_df = pd.read_csv(self.model_metrics_file)
        
        # Get latest entry for each model
        if 'timestamp' in metrics_df.columns:
            # Convert timestamp to datetime
            metrics_df['timestamp'] = pd.to_datetime(metrics_df['timestamp'])
            
            # Get latest entry for each model
            latest_metrics = metrics_df.sort_values('timestamp').groupby('model_type').last().reset_index()
        else:
            # If no timestamp, get the last entry for each model
            latest_metrics = metrics_df.groupby('model_type').last().reset_index()
        
        return latest_metrics
    
    def plot_model_comparison(self):
        """
        Create comparative bar chart of model performance.
        
        Returns:
            str: Path to saved plot
        """
        # Get latest metrics
        metrics_df = self.get_latest_metrics()
        
        if metrics_df.empty:
            logger.warning("No metrics data available for comparison")
            return None
        
        # Define metrics to compare
        comparison_metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
        
        # Create plot
        plt.figure(figsize=(12, 6))
        
        # Create grouped bar chart
        bar_width = 0.2
        index = np.arange(len(metrics_df['model_type']))
        
        for i, metric in enumerate(comparison_metrics):
            plt.bar(index + i*bar_width, metrics_df[metric], bar_width, 
                   label=metric.capitalize().replace('_', ' '))
        
        plt.xlabel('Model Type')
        plt.ylabel('Score')
        plt.title('Model Performance Comparison')
        plt.xticks(index + bar_width * (len(comparison_metrics)-1)/2, metrics_df['model_type'])
        plt.legend()
        plt.ylim(0, 1)
        plt.tight_layout()
        
        # Save plot
        plot_file = os.path.join(self.metrics_dir, f'model_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        plt.savefig(plot_file)
        plt.close()
        
        logger.info(f"Saved model comparison plot to {plot_file}")
        return plot_file