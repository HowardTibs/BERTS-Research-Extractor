import os
import json
import time
import logging
import psutil
import platform
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """Monitor and track runtime performance metrics."""
    
    def __init__(self, metrics_dir='./data/metrics'):
        """
        Initialize performance monitor.
        
        Args:
            metrics_dir (str): Directory to store metrics data
        """
        self.metrics_dir = metrics_dir
        os.makedirs(metrics_dir, exist_ok=True)
        self.performance_metrics_file = os.path.join(metrics_dir, 'performance_metrics.csv')
        
        # Initialize tracking variables
        self.start_time = None
        self.end_time = None
        self.stage_times = {}
        self.current_stage = None
        self.memory_samples = []
        self.cpu_samples = []
        self.sampling_interval = 0.5  # seconds
        
        # Get system info
        self.system_info = self._get_system_info()
        
        logger.info(f"Initialized PerformanceMonitor with metrics directory: {metrics_dir}")
        logger.info(f"System info: {platform.system()} {platform.release()}, {psutil.cpu_count()} CPUs")
    
    def _get_system_info(self):
        """
        Get system information.
        
        Returns:
            dict: System information
        """
        info = {
            'system': platform.system(),
            'release': platform.release(),
            'version': platform.version(),
            'machine': platform.machine(),
            'processor': platform.processor(),
            'cpu_count': psutil.cpu_count(),
            'cpu_freq': psutil.cpu_freq().current if psutil.cpu_freq() else 'Unknown',
            'total_memory': psutil.virtual_memory().total,
            'python_version': platform.python_version()
        }
        return info
    
    def start_monitoring(self, document_name=None):
        """
        Start monitoring performance.
        
        Args:
            document_name (str): Name of document being processed
        """
        self.document_name = document_name
        self.start_time = time.time()
        self.stage_times = {}
        self.memory_samples = []
        self.cpu_samples = []
        
        # Start background monitoring
        self._start_background_monitoring()
        
        logger.info(f"Started performance monitoring for document: {document_name}")
    
    def _start_background_monitoring(self):
        """Start background monitoring thread."""
        import threading
        
        def monitor_resources():
            while self.start_time and not self.end_time:
                # Sample CPU and memory
                self.cpu_samples.append(psutil.cpu_percent(interval=None))
                self.memory_samples.append(psutil.Process(os.getpid()).memory_info().rss)
                time.sleep(self.sampling_interval)
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=monitor_resources, daemon=True)
        self.monitor_thread.start()
    
    def start_stage(self, stage_name):
        """
        Start timing a specific processing stage.
        
        Args:
            stage_name (str): Name of the processing stage
        """
        if self.current_stage:
            # End the previous stage if there is one
            self.end_stage()
        
        self.current_stage = stage_name
        self.stage_times[stage_name] = {
            'start': time.time(),
            'end': None,
            'duration': None
        }
        
        logger.debug(f"Started stage: {stage_name}")
    
    def end_stage(self):
        """End the current processing stage."""
        if not self.current_stage:
            return
        
        # Record end time for the current stage
        stage = self.stage_times[self.current_stage]
        stage['end'] = time.time()
        stage['duration'] = stage['end'] - stage['start']
        
        logger.debug(f"Ended stage: {self.current_stage}, duration: {stage['duration']:.4f} seconds")
        self.current_stage = None
    
    def end_monitoring(self):
        """
        End monitoring and calculate overall metrics.
        
        Returns:
            dict: Performance metrics
        """
        # End any ongoing stage
        if self.current_stage:
            self.end_stage()
        
        # Record end time
        self.end_time = time.time()
        total_runtime = self.end_time - self.start_time
        
        # Calculate CPU and memory metrics
        avg_cpu = np.mean(self.cpu_samples) if self.cpu_samples else 0
        max_cpu = np.max(self.cpu_samples) if self.cpu_samples else 0
        avg_memory = np.mean(self.memory_samples) if self.memory_samples else 0
        max_memory = np.max(self.memory_samples) if self.memory_samples else 0
        
        # Get current process to measure other metrics
        process = psutil.Process(os.getpid())
        
        # Calculate power consumption estimation (very rough approximation)
        # Based on CPU usage and TDP with a simple linear model
        # Assume 100% CPU = TDP, and scale linearly (this is a simplification)
        tdp_estimate = 15  # Watts, typical for a laptop CPU at load
        power_consumption = (avg_cpu / 100) * tdp_estimate
        
        # Prepare metrics
        metrics = {
            'document_name': self.document_name or 'unknown',
            'total_runtime_seconds': total_runtime,
            'avg_cpu_percent': avg_cpu,
            'max_cpu_percent': max_cpu,
            'avg_memory_mb': avg_memory / (1024 * 1024),  # Convert bytes to MB
            'max_memory_mb': max_memory / (1024 * 1024),  # Convert bytes to MB
            'estimated_power_consumption_watts': power_consumption,
            'io_read_mb': process.io_counters().read_bytes / (1024 * 1024) if hasattr(process, 'io_counters') else 0,
            'io_write_mb': process.io_counters().write_bytes / (1024 * 1024) if hasattr(process, 'io_counters') else 0,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_type': 'unknown'  # Will be set by the caller
        }
        
        # Add stage durations
        for stage, timing in self.stage_times.items():
            metrics[f'{stage}_seconds'] = timing['duration']
        
        logger.info(f"Ended performance monitoring: total runtime = {total_runtime:.4f} seconds")
        return metrics
    
    def save_metrics(self, metrics, model_type=None):
        """
        Save performance metrics to CSV.
        
        Args:
            metrics (dict): Performance metrics
            model_type (str): Type of model used
        """
        # Update model type if provided
        if model_type:
            metrics['model_type'] = model_type
        
        # Create DataFrame from metrics
        metrics_df = pd.DataFrame([metrics])
        
        # Save to CSV
        if os.path.exists(self.performance_metrics_file):
            # Append to existing file
            existing_df = pd.read_csv(self.performance_metrics_file)
            updated_df = pd.concat([existing_df, metrics_df], ignore_index=True)
            updated_df.to_csv(self.performance_metrics_file, index=False)
        else:
            # Create new file
            metrics_df.to_csv(self.performance_metrics_file, index=False)
        
        logger.info(f"Saved performance metrics to {self.performance_metrics_file}")
    
    def plot_performance_comparison(self):
        """
        Create comparative charts of performance metrics.
        
        Returns:
            str: Path to saved plot
        """
        if not os.path.exists(self.performance_metrics_file):
            logger.warning(f"Performance metrics file not found: {self.performance_metrics_file}")
            return None
        
        # Read metrics file
        metrics_df = pd.read_csv(self.performance_metrics_file)
        
        if metrics_df.empty:
            logger.warning("No performance metrics available for comparison")
            return None
        
        # Group by model type
        grouped_metrics = metrics_df.groupby('model_type').mean().reset_index()
        
        # Define metrics to compare
        runtime_metrics = ['total_runtime_seconds']
        if 'document_processing_seconds' in metrics_df.columns:
            runtime_metrics.append('document_processing_seconds')
        if 'text_processing_seconds' in metrics_df.columns:
            runtime_metrics.append('text_processing_seconds')
        if 'classification_seconds' in metrics_df.columns:
            runtime_metrics.append('classification_seconds')
        
        resource_metrics = ['avg_cpu_percent', 'avg_memory_mb', 'estimated_power_consumption_watts']
        
        # Create plots (2x1 grid)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Runtime comparison
        grouped_metrics[runtime_metrics].set_index(grouped_metrics['model_type']).plot(kind='bar', ax=ax1)
        ax1.set_title('Processing Time Comparison by Model')
        ax1.set_ylabel('Seconds')
        ax1.set_xlabel('Model Type')
        ax1.legend(title='Stage')
        
        # Resource usage comparison
        grouped_metrics[resource_metrics].set_index(grouped_metrics['model_type']).plot(kind='bar', ax=ax2)
        ax2.set_title('Resource Usage Comparison by Model')
        ax2.set_ylabel('Value')
        ax2.set_xlabel('Model Type')
        ax2.legend(title='Metric')
        
        plt.tight_layout()
        
        # Save plot
        plot_file = os.path.join(self.metrics_dir, f'performance_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        plt.savefig(plot_file)
        plt.close()
        
        logger.info(f"Saved performance comparison plot to {plot_file}")
        return plot_file
    
    def create_performance_summary(self):
        """
        Create a summary of performance metrics.
        
        Returns:
            pd.DataFrame: Summary of performance metrics
        """
        if not os.path.exists(self.performance_metrics_file):
            logger.warning(f"Performance metrics file not found: {self.performance_metrics_file}")
            return pd.DataFrame()
        
        # Read metrics file
        metrics_df = pd.read_csv(self.performance_metrics_file)
        
        if metrics_df.empty:
            logger.warning("No performance metrics available for summary")
            return pd.DataFrame()
        
        # Group by model type
        summary = metrics_df.groupby('model_type').agg({
            'total_runtime_seconds': ['mean', 'min', 'max'],
            'avg_cpu_percent': ['mean', 'max'],
            'avg_memory_mb': ['mean', 'max'],
            'estimated_power_consumption_watts': 'mean'
        }).reset_index()
        
        # Flatten multi-level columns
        summary.columns = ['_'.join(col).strip('_') for col in summary.columns.values]
        
        return summary