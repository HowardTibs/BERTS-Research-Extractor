import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
import logging
from datetime import datetime
import threading
import time

# Add parent directory to path to import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from src.model_evaluator import ModelEvaluator
from src.performance_monitor import PerformanceMonitor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('metrics_gui.log')
    ]
)
logger = logging.getLogger(__name__)

class MetricsGUI:
    """GUI for displaying metrics and comparisons."""
    
    def __init__(self, root):
        """Initialize GUI."""
        self.root = root
        self.root.title("Research Paper Extractor - Metrics Dashboard")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)
        
        # Initialize metric components
        self.metrics_dir = tk.StringVar(value=os.path.join(os.getcwd(), "data/metrics"))
        self.model_evaluator = ModelEvaluator(metrics_dir=self.metrics_dir.get())
        self.performance_monitor = PerformanceMonitor(metrics_dir=self.metrics_dir.get())
        
        # Create GUI elements
        self._create_widgets()
        
        # Load initial data
        self._load_metrics_data()
    
    def _create_widgets(self):
        """Create GUI widgets."""
        # Create main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create tabs
        tab_control = ttk.Notebook(main_frame)
        model_tab = ttk.Frame(tab_control)
        performance_tab = ttk.Frame(tab_control)
        comparison_tab = ttk.Frame(tab_control)
        
        tab_control.add(model_tab, text="Model Metrics")
        tab_control.add(performance_tab, text="Performance Metrics")
        tab_control.add(comparison_tab, text="Model Comparison")
        tab_control.pack(fill=tk.BOTH, expand=True)
        
        # === Header Frame (common across tabs) ===
        header_frame = ttk.Frame(main_frame, padding="5")
        header_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(header_frame, text="Metrics Directory:").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Entry(header_frame, textvariable=self.metrics_dir, width=50).grid(row=0, column=1, sticky=tk.EW, padx=5, pady=5)
        ttk.Button(header_frame, text="Browse", command=self._browse_metrics_dir).grid(row=0, column=2, sticky=tk.W, pady=5)
        ttk.Button(header_frame, text="Refresh Data", command=self._load_metrics_data).grid(row=0, column=3, sticky=tk.E, pady=5)
        
        # === Model Metrics Tab ===
        self._create_model_metrics_tab(model_tab)
        
        # === Performance Metrics Tab ===
        self._create_performance_metrics_tab(performance_tab)
        
        # === Comparison Tab ===
        self._create_comparison_tab(comparison_tab)
    
    def _create_model_metrics_tab(self, parent):
        """Create model metrics tab."""
        # Model metrics frame
        metrics_frame = ttk.Frame(parent, padding="10")
        metrics_frame.pack(fill=tk.BOTH, expand=True)
        
        # Model selector
        selector_frame = ttk.Frame(metrics_frame)
        selector_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(selector_frame, text="Select Model:").pack(side=tk.LEFT, padx=5)
        self.model_selector = ttk.Combobox(selector_frame, state="readonly", width=20)
        self.model_selector.pack(side=tk.LEFT, padx=5)
        self.model_selector.bind("<<ComboboxSelected>>", self._update_model_metrics)
        
        # Model metrics table
        table_frame = ttk.LabelFrame(metrics_frame, text="Model Metrics", padding="10")
        table_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Create treeview for metrics
        self.metrics_tree = ttk.Treeview(table_frame, columns=("Metric", "Value"), show="headings")
        self.metrics_tree.heading("Metric", text="Metric")
        self.metrics_tree.heading("Value", text="Value")
        self.metrics_tree.column("Metric", width=200)
        self.metrics_tree.column("Value", width=100)
        self.metrics_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Add scrollbar
        tree_scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.metrics_tree.yview)
        tree_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.metrics_tree.config(yscrollcommand=tree_scrollbar.set)
        
        # Placeholder for confusion matrix
        self.confusion_matrix_frame = ttk.LabelFrame(metrics_frame, text="Confusion Matrix", padding="10")
        self.confusion_matrix_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Add placeholder text
        ttk.Label(self.confusion_matrix_frame, text="Select a model to display confusion matrix").pack(pady=20)
    
    def _create_performance_metrics_tab(self, parent):
        """Create performance metrics tab."""
        # Performance metrics frame
        performance_frame = ttk.Frame(parent, padding="10")
        performance_frame.pack(fill=tk.BOTH, expand=True)
        
        # Model selector
        selector_frame = ttk.Frame(performance_frame)
        selector_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(selector_frame, text="Select Model:").pack(side=tk.LEFT, padx=5)
        self.perf_model_selector = ttk.Combobox(selector_frame, state="readonly", width=20)
        self.perf_model_selector.pack(side=tk.LEFT, padx=5)
        self.perf_model_selector.bind("<<ComboboxSelected>>", self._update_performance_metrics)
        
        # Performance metrics tables
        tables_frame = ttk.Frame(performance_frame)
        tables_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Runtime metrics
        runtime_frame = ttk.LabelFrame(tables_frame, text="Runtime Metrics", padding="10")
        runtime_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        self.runtime_tree = ttk.Treeview(runtime_frame, columns=("Metric", "Value"), show="headings")
        self.runtime_tree.heading("Metric", text="Metric")
        self.runtime_tree.heading("Value", text="Value")
        self.runtime_tree.column("Metric", width=200)
        self.runtime_tree.column("Value", width=100)
        self.runtime_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        runtime_scrollbar = ttk.Scrollbar(runtime_frame, orient=tk.VERTICAL, command=self.runtime_tree.yview)
        runtime_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.runtime_tree.config(yscrollcommand=runtime_scrollbar.set)
        
        # Resource metrics
        resource_frame = ttk.LabelFrame(tables_frame, text="Resource Metrics", padding="10")
        resource_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        
        self.resource_tree = ttk.Treeview(resource_frame, columns=("Metric", "Value"), show="headings")
        self.resource_tree.heading("Metric", text="Metric")
        self.resource_tree.heading("Value", text="Value")
        self.resource_tree.column("Metric", width=200)
        self.resource_tree.column("Value", width=100)
        self.resource_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        resource_scrollbar = ttk.Scrollbar(resource_frame, orient=tk.VERTICAL, command=self.resource_tree.yview)
        resource_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.resource_tree.config(yscrollcommand=resource_scrollbar.set)
        
        # Placeholder for performance charts
        self.performance_chart_frame = ttk.LabelFrame(performance_frame, text="Performance Charts", padding="10")
        self.performance_chart_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Add placeholder text
        ttk.Label(self.performance_chart_frame, text="Select a model to display performance charts").pack(pady=20)
    
    def _create_comparison_tab(self, parent):
        """Create comparison tab."""
        # Comparison frame
        comparison_frame = ttk.Frame(parent, padding="10")
        comparison_frame.pack(fill=tk.BOTH, expand=True)
        
        # Buttons to generate comparison charts
        button_frame = ttk.Frame(comparison_frame)
        button_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(button_frame, text="Generate Model Metrics Comparison", command=self._show_model_comparison).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Generate Performance Comparison", command=self._show_performance_comparison).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Generate Comprehensive Comparison", command=self._show_comprehensive_comparison).pack(side=tk.LEFT, padx=5)
        
        # Comparison charts frame
        self.comparison_charts_frame = ttk.LabelFrame(comparison_frame, text="Comparison Charts", padding="10")
        self.comparison_charts_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Summary table
        summary_frame = ttk.LabelFrame(comparison_frame, text="Summary Table", padding="10")
        summary_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Create treeview for summary
        self.summary_tree = ttk.Treeview(summary_frame)
        self.summary_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Add scrollbar
        summary_scrollbar = ttk.Scrollbar(summary_frame, orient=tk.VERTICAL, command=self.summary_tree.yview)
        summary_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.summary_tree.config(yscrollcommand=summary_scrollbar.set)
        
        # Add horizontal scrollbar
        summary_h_scrollbar = ttk.Scrollbar(summary_frame, orient=tk.HORIZONTAL, command=self.summary_tree.xview)
        summary_h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.summary_tree.config(xscrollcommand=summary_h_scrollbar.set)
    
    def _browse_metrics_dir(self):
        """Browse for metrics directory."""
        directory = filedialog.askdirectory(title="Select Metrics Directory")
        if directory:
            self.metrics_dir.set(directory)
            # Reinitialize with new directory
            self.model_evaluator = ModelEvaluator(metrics_dir=self.metrics_dir.get())
            self.performance_monitor = PerformanceMonitor(metrics_dir=self.metrics_dir.get())
            # Reload data
            self._load_metrics_data()
    
    def _load_metrics_data(self):
        """Load metrics data from files."""
        try:
            # Load model metrics
            model_metrics_file = os.path.join(self.metrics_dir.get(), 'model_metrics.csv')
            if os.path.exists(model_metrics_file):
                model_df = pd.read_csv(model_metrics_file)
                
                # Get unique model types
                model_types = model_df['model_type'].unique().tolist()
                
                # Update model selector
                self.model_selector['values'] = model_types
                self.perf_model_selector['values'] = model_types
                
                if model_types:
                    self.model_selector.current(0)
                    self.perf_model_selector.current(0)
                    self._update_model_metrics(None)
                    self._update_performance_metrics(None)
            else:
                messagebox.showwarning("Warning", f"Model metrics file not found: {model_metrics_file}")
            
            # Load performance metrics
            performance_metrics_file = os.path.join(self.metrics_dir.get(), 'performance_metrics.csv')
            if os.path.exists(performance_metrics_file):
                # Create summary table
                self._create_summary_table()
            else:
                messagebox.showwarning("Warning", f"Performance metrics file not found: {performance_metrics_file}")
        
        except Exception as e:
            messagebox.showerror("Error", f"Error loading metrics data: {str(e)}")
            logger.error(f"Error loading metrics data: {str(e)}", exc_info=True)
    
    def _update_model_metrics(self, event):
        """Update model metrics display."""
        selected_model = self.model_selector.get()
        if not selected_model:
            return
        
        try:
            # Load metrics for selected model
            model_metrics_file = os.path.join(self.metrics_dir.get(), 'model_metrics.csv')
            if not os.path.exists(model_metrics_file):
                messagebox.showwarning("Warning", f"Model metrics file not found: {model_metrics_file}")
                return
            
            # Read metrics and filter for selected model
            metrics_df = pd.read_csv(model_metrics_file)
            model_data = metrics_df[metrics_df['model_type'] == selected_model].iloc[-1].to_dict()
            
            # Clear existing data
            for item in self.metrics_tree.get_children():
                self.metrics_tree.delete(item)
            
            # Add metrics to tree
            metrics_to_display = [
                ('Accuracy', 'accuracy'),
                ('Precision (Micro)', 'precision_micro'),
                ('Precision (Macro)', 'precision_macro'),
                ('Recall (Micro)', 'recall_micro'),
                ('Recall (Macro)', 'recall_macro'),
                ('F1 Score (Micro)', 'f1_micro'),
                ('F1 Score (Macro)', 'f1_macro'),
                ('Evaluation Time (s)', 'eval_time_seconds'),
                ('Number of Samples', 'num_samples'),
                ('Timestamp', 'timestamp')
            ]
            
            for display_name, metric_key in metrics_to_display:
                if metric_key in model_data:
                    value = model_data[metric_key]
                    # Format numeric values
                    if isinstance(value, (int, float)):
                        value = f"{value:.4f}" if metric_key != 'num_samples' else str(int(value))
                    self.metrics_tree.insert('', tk.END, values=(display_name, value))
            
            # Update confusion matrix if available
            self._update_confusion_matrix(selected_model)
        
        except Exception as e:
            messagebox.showerror("Error", f"Error updating model metrics: {str(e)}")
            logger.error(f"Error updating model metrics: {str(e)}", exc_info=True)
    
    def _update_confusion_matrix(self, model_type):
        """Update confusion matrix display."""
        # Clear existing frame content
        for widget in self.confusion_matrix_frame.winfo_children():
            widget.destroy()
        
        # Find confusion matrix file
        matrix_files = [f for f in os.listdir(self.metrics_dir.get()) 
                        if f.startswith(f"{model_type}_confusion_matrix") and f.endswith(".png")]
        
        if matrix_files:
            # Sort by timestamp to get the latest
            matrix_files.sort(reverse=True)
            matrix_file = os.path.join(self.metrics_dir.get(), matrix_files[0])
            
            try:
                # Display image
                from PIL import Image, ImageTk
                
                img = Image.open(matrix_file)
                # Resize to fit frame
                width, height = img.size
                max_size = 500
                if width > max_size or height > max_size:
                    ratio = min(max_size/width, max_size/height)
                    img = img.resize((int(width*ratio), int(height*ratio)), Image.LANCZOS)
                
                img = ImageTk.PhotoImage(img)
                
                # Create label to hold image
                img_label = ttk.Label(self.confusion_matrix_frame, image=img)
                img_label.image = img  # Keep a reference to prevent garbage collection
                img_label.pack(pady=10)
            
            except Exception as e:
                ttk.Label(self.confusion_matrix_frame, text=f"Error loading confusion matrix: {str(e)}").pack(pady=20)
        else:
            # Create new confusion matrix
            try:
                # Load test data results
                report_files = [f for f in os.listdir(self.metrics_dir.get()) 
                                if f.startswith(f"{model_type}_report") and f.endswith(".json")]
                
                if report_files:
                    # Display message
                    ttk.Label(self.confusion_matrix_frame, text="No confusion matrix image found, please generate one using model evaluator").pack(pady=20)
                else:
                    ttk.Label(self.confusion_matrix_frame, text="No confusion matrix or classification report available for this model").pack(pady=20)
            
            except Exception as e:
                ttk.Label(self.confusion_matrix_frame, text=f"Error checking for classification reports: {str(e)}").pack(pady=20)
    
    def _update_performance_metrics(self, event):
        """Update performance metrics display."""
        selected_model = self.perf_model_selector.get()
        if not selected_model:
            return
        
        try:
            # Load metrics for selected model
            perf_metrics_file = os.path.join(self.metrics_dir.get(), 'performance_metrics.csv')
            if not os.path.exists(perf_metrics_file):
                messagebox.showwarning("Warning", f"Performance metrics file not found: {perf_metrics_file}")
                return
            
            # Read metrics and filter for selected model
            metrics_df = pd.read_csv(perf_metrics_file)
            
            if 'model_type' not in metrics_df.columns:
                messagebox.showwarning("Warning", "Performance metrics file does not contain model type information")
                return
            
            model_data = metrics_df[metrics_df['model_type'] == selected_model]
            
            if model_data.empty:
                messagebox.showwarning("Warning", f"No performance data found for model type: {selected_model}")
                return
            
            # Use the latest entry
            model_data = model_data.iloc[-1].to_dict()
            
            # Clear existing data
            for item in self.runtime_tree.get_children():
                self.runtime_tree.delete(item)
            
            for item in self.resource_tree.get_children():
                self.resource_tree.delete(item)
            
            # Add runtime metrics to tree
            runtime_metrics = [
                ('Total Runtime (s)', 'total_runtime_seconds'),
                ('Document Processing (s)', 'document_processing_seconds'),
                ('Text Processing (s)', 'text_processing_seconds'),
                ('Classification (s)', 'classification_seconds'),
                ('Output Formatting (s)', 'output_formatting_seconds')
            ]
            
            for display_name, metric_key in runtime_metrics:
                if metric_key in model_data:
                    value = model_data[metric_key]
                    # Format numeric values
                    if isinstance(value, (int, float)):
                        value = f"{value:.4f}"
                    self.runtime_tree.insert('', tk.END, values=(display_name, value))
            
            # Add resource metrics to tree
            resource_metrics = [
                ('Avg CPU Usage (%)', 'avg_cpu_percent'),
                ('Max CPU Usage (%)', 'max_cpu_percent'),
                ('Avg Memory Usage (MB)', 'avg_memory_mb'),
                ('Max Memory Usage (MB)', 'max_memory_mb'),
                ('Est. Power Consumption (W)', 'estimated_power_consumption_watts'),
                ('I/O Read (MB)', 'io_read_mb'),
                ('I/O Write (MB)', 'io_write_mb')
            ]
            
            for display_name, metric_key in resource_metrics:
                if metric_key in model_data:
                    value = model_data[metric_key]
                    # Format numeric values
                    if isinstance(value, (int, float)):
                        value = f"{value:.4f}"
                    self.resource_tree.insert('', tk.END, values=(display_name, value))
            
            # Update performance charts
            self._update_performance_charts(selected_model)
        
        except Exception as e:
            messagebox.showerror("Error", f"Error updating performance metrics: {str(e)}")
            logger.error(f"Error updating performance metrics: {str(e)}", exc_info=True)
    
    def _update_performance_charts(self, model_type):
        """Update performance charts display."""
        # Clear existing frame content
        for widget in self.performance_chart_frame.winfo_children():
            widget.destroy()
        
        try:
            # Load metrics for selected model
            perf_metrics_file = os.path.join(self.metrics_dir.get(), 'performance_metrics.csv')
            if not os.path.exists(perf_metrics_file):
                ttk.Label(self.performance_chart_frame, text="Performance metrics file not found").pack(pady=20)
                return
            
            # Read metrics and filter for selected model
            metrics_df = pd.read_csv(perf_metrics_file)
            
            if 'model_type' not in metrics_df.columns:
                ttk.Label(self.performance_chart_frame, text="Performance metrics file does not contain model type information").pack(pady=20)
                return
            
            model_data = metrics_df[metrics_df['model_type'] == model_type]
            
            if model_data.empty:
                ttk.Label(self.performance_chart_frame, text=f"No performance data found for model type: {model_type}").pack(pady=20)
                return
            
            # Create figure with two subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
            
            # Runtime vs. Document Size
            if 'document_name' in model_data.columns:
                ax1.scatter(model_data['total_runtime_seconds'], model_data['avg_memory_mb'], alpha=0.7)
                ax1.set_xlabel('Runtime (seconds)')
                ax1.set_ylabel('Memory Usage (MB)')
                ax1.set_title('Runtime vs. Memory Usage')
                ax1.grid(True, linestyle='--', alpha=0.7)
            
            # CPU vs. Memory Usage
            ax2.scatter(model_data['avg_cpu_percent'], model_data['avg_memory_mb'], alpha=0.7)
            ax2.set_xlabel('CPU Usage (%)')
            ax2.set_ylabel('Memory Usage (MB)')
            ax2.set_title('CPU vs. Memory Usage')
            ax2.grid(True, linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            
            # Create canvas and add to frame
            canvas = FigureCanvasTkAgg(fig, master=self.performance_chart_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Add toolbar
            from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
            toolbar_frame = ttk.Frame(self.performance_chart_frame)
            toolbar_frame.pack(fill=tk.X)
            toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
            toolbar.update()
        
        except Exception as e:
            ttk.Label(self.performance_chart_frame, text=f"Error creating performance charts: {str(e)}").pack(pady=20)
            logger.error(f"Error creating performance charts: {str(e)}", exc_info=True)
    
    def _show_model_comparison(self):
        """Show model comparison charts."""
        # Clear existing frame content
        for widget in self.comparison_charts_frame.winfo_children():
            widget.destroy()
        
        try:
            # Generate model comparison plot
            plot_file = self.model_evaluator.plot_model_comparison()
            
            if plot_file and os.path.exists(plot_file):
                # Display image
                from PIL import Image, ImageTk
                
                img = Image.open(plot_file)
                # Resize to fit frame
                width, height = img.size
                max_width = 800
                if width > max_width:
                    ratio = max_width/width
                    img = img.resize((int(width*ratio), int(height*ratio)), Image.LANCZOS)
                
                img = ImageTk.PhotoImage(img)
                
                # Create label to hold image
                img_label = ttk.Label(self.comparison_charts_frame, image=img)
                img_label.image = img  # Keep a reference to prevent garbage collection
                img_label.pack(pady=10)
            else:
                ttk.Label(self.comparison_charts_frame, text="Failed to generate model comparison chart").pack(pady=20)
        
        except Exception as e:
            ttk.Label(self.comparison_charts_frame, text=f"Error creating model comparison: {str(e)}").pack(pady=20)
            logger.error(f"Error creating model comparison: {str(e)}", exc_info=True)
    
    def _show_performance_comparison(self):
        """Show performance comparison charts."""
        # Clear existing frame content
        for widget in self.comparison_charts_frame.winfo_children():
            widget.destroy()
        
        try:
            # Generate performance comparison plot
            plot_file = self.performance_monitor.plot_performance_comparison()
            
            if plot_file and os.path.exists(plot_file):
                # Display image
                from PIL import Image, ImageTk
                
                img = Image.open(plot_file)
                # Resize to fit frame
                width, height = img.size
                max_width = 800
                if width > max_width:
                    ratio = max_width/width
                    img = img.resize((int(width*ratio), int(height*ratio)), Image.LANCZOS)
                
                img = ImageTk.PhotoImage(img)
                
                # Create label to hold image
                img_label = ttk.Label(self.comparison_charts_frame, image=img)
                img_label.image = img  # Keep a reference to prevent garbage collection
                img_label.pack(pady=10)
            else:
                ttk.Label(self.comparison_charts_frame, text="Failed to generate performance comparison chart").pack(pady=20)
        
        except Exception as e:
            ttk.Label(self.comparison_charts_frame, text=f"Error creating performance comparison: {str(e)}").pack(pady=20)
            logger.error(f"Error creating performance comparison: {str(e)}", exc_info=True)
    
    def _show_comprehensive_comparison(self):
        """Show comprehensive comparison of models."""
        # Clear existing frame content
        for widget in self.comparison_charts_frame.winfo_children():
            widget.destroy()
        
        try:
            # Load model metrics
            model_metrics_file = os.path.join(self.metrics_dir.get(), 'model_metrics.csv')
            perf_metrics_file = os.path.join(self.metrics_dir.get(), 'performance_metrics.csv')
            
            if not os.path.exists(model_metrics_file) or not os.path.exists(perf_metrics_file):
                ttk.Label(self.comparison_charts_frame, text="Model or performance metrics file not found").pack(pady=20)
                return
            
            # Read metrics
            model_df = pd.read_csv(model_metrics_file)
            perf_df = pd.read_csv(perf_metrics_file)
            
            # Get latest metrics for each model
            if 'timestamp' in model_df.columns:
                model_df['timestamp'] = pd.to_datetime(model_df['timestamp'])
                latest_model_metrics = model_df.sort_values('timestamp').groupby('model_type').last().reset_index()
            else:
                latest_model_metrics = model_df.groupby('model_type').last().reset_index()
            
            if 'timestamp' in perf_df.columns:
                perf_df['timestamp'] = pd.to_datetime(perf_df['timestamp'])
                latest_perf_metrics = perf_df.sort_values('timestamp').groupby('model_type').last().reset_index()
            else:
                latest_perf_metrics = perf_df.groupby('model_type').last().reset_index()
            
            # Merge metrics
            merged_metrics = pd.merge(latest_model_metrics, latest_perf_metrics, on='model_type', how='outer')
            
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Accuracy metrics
            ax1 = axes[0, 0]
            metrics_to_plot = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']
            merged_metrics[metrics_to_plot].set_index(merged_metrics['model_type']).plot(kind='bar', ax=ax1)
            ax1.set_title('Model Accuracy Metrics')
            ax1.set_ylim(0, 1)
            ax1.set_ylabel('Score')
            ax1.set_xlabel('')
            ax1.legend(title='Metric')
            
            # Runtime metrics
            ax2 = axes[0, 1]
            runtime_cols = [col for col in merged_metrics.columns if col.endswith('_seconds') and col != 'eval_time_seconds']
            
            if runtime_cols:
                merged_metrics[runtime_cols].set_index(merged_metrics['model_type']).plot(kind='bar', ax=ax2)
                ax2.set_title('Processing Time Breakdown')
                ax2.set_ylabel('Seconds')
                ax2.set_xlabel('')
                ax2.legend(title='Stage')
            else:
                ax2.text(0.5, 0.5, 'No runtime breakdown data available', ha='center', va='center')
                ax2.set_title('Processing Time Breakdown')
                ax2.axis('off')
            
            # Resource usage
            ax3 = axes[1, 0]
            resource_cols = ['avg_cpu_percent', 'avg_memory_mb', 'estimated_power_consumption_watts']
            
            if all(col in merged_metrics.columns for col in resource_cols):
                # Normalize for better visualization
                normalized_metrics = merged_metrics.copy()
                for col in resource_cols:
                    max_val = normalized_metrics[col].max()
                    if max_val > 0:
                        normalized_metrics[col] = normalized_metrics[col] / max_val
                
                normalized_metrics[resource_cols].set_index(normalized_metrics['model_type']).plot(kind='bar', ax=ax3)
                ax3.set_title('Resource Usage (Normalized)')
                ax3.set_ylabel('Normalized Value')
                ax3.set_xlabel('')
                ax3.legend(title='Resource')
            else:
                ax3.text(0.5, 0.5, 'No resource usage data available', ha='center', va='center')
                ax3.set_title('Resource Usage')
                ax3.axis('off')
            
            # Tradeoff: Accuracy vs Speed
            ax4 = axes[1, 1]
            
            if 'accuracy' in merged_metrics.columns and 'total_runtime_seconds' in merged_metrics.columns:
                # Create scatter plot
                ax4.scatter(
                    merged_metrics['total_runtime_seconds'], 
                    merged_metrics['accuracy'],
                    s=100, alpha=0.7
                )
                
                # Add labels for each point
                for i, model in enumerate(merged_metrics['model_type']):
                    ax4.annotate(
                        model,
                        (merged_metrics['total_runtime_seconds'].iloc[i], merged_metrics['accuracy'].iloc[i]),
                        xytext=(5, 5), textcoords='offset points'
                    )
                
                ax4.set_title('Accuracy vs. Speed Tradeoff')
                ax4.set_ylabel('Accuracy')
                ax4.set_xlabel('Runtime (seconds)')
                ax4.grid(True, linestyle='--', alpha=0.7)
            else:
                ax4.text(0.5, 0.5, 'Accuracy or runtime data not available', ha='center', va='center')
                ax4.set_title('Accuracy vs. Speed Tradeoff')
                ax4.axis('off')
            
            plt.tight_layout()
            
            # Create canvas and add to frame
            canvas = FigureCanvasTkAgg(fig, master=self.comparison_charts_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Add toolbar
            from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
            toolbar_frame = ttk.Frame(self.comparison_charts_frame)
            toolbar_frame.pack(fill=tk.X)
            toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
            toolbar.update()
            
            # Update summary table
            self._create_summary_table()
        
        except Exception as e:
            ttk.Label(self.comparison_charts_frame, text=f"Error creating comprehensive comparison: {str(e)}").pack(pady=20)
            logger.error(f"Error creating comprehensive comparison: {str(e)}", exc_info=True)
    
    def _create_summary_table(self):
        """Create summary table of all metrics."""
        try:
            # Load model metrics
            model_metrics_file = os.path.join(self.metrics_dir.get(), 'model_metrics.csv')
            perf_metrics_file = os.path.join(self.metrics_dir.get(), 'performance_metrics.csv')
            
            if not os.path.exists(model_metrics_file) or not os.path.exists(perf_metrics_file):
                return
            
            # Read metrics
            model_df = pd.read_csv(model_metrics_file)
            perf_df = pd.read_csv(perf_metrics_file)
            
            # Get latest metrics for each model
            if 'timestamp' in model_df.columns:
                model_df['timestamp'] = pd.to_datetime(model_df['timestamp'])
                latest_model_metrics = model_df.sort_values('timestamp').groupby('model_type').last().reset_index()
            else:
                latest_model_metrics = model_df.groupby('model_type').last().reset_index()
            
            if 'timestamp' in perf_df.columns:
                perf_df['timestamp'] = pd.to_datetime(perf_df['timestamp'])
                latest_perf_metrics = perf_df.sort_values('timestamp').groupby('model_type').last().reset_index()
            else:
                latest_perf_metrics = perf_df.groupby('model_type').last().reset_index()
            
            # Merge metrics
            merged_metrics = pd.merge(latest_model_metrics, latest_perf_metrics, on='model_type', how='outer')
            
            # Select columns for summary
            summary_columns = [
                'model_type', 'accuracy', 'f1_macro', 'precision_macro', 'recall_macro',
                'total_runtime_seconds', 'avg_cpu_percent', 'avg_memory_mb', 
                'estimated_power_consumption_watts'
            ]
            
            # Filter to available columns
            summary_columns = [col for col in summary_columns if col in merged_metrics.columns]
            
            # Create summary dataframe
            summary_df = merged_metrics[summary_columns].copy()
            
            # Rename columns for display
            column_mapping = {
                'model_type': 'Model',
                'accuracy': 'Accuracy',
                'f1_macro': 'F1 Score',
                'precision_macro': 'Precision',
                'recall_macro': 'Recall',
                'total_runtime_seconds': 'Runtime (s)',
                'avg_cpu_percent': 'CPU (%)',
                'avg_memory_mb': 'Memory (MB)',
                'estimated_power_consumption_watts': 'Power (W)'
            }
            
            summary_df = summary_df.rename(columns={col: column_mapping.get(col, col) for col in summary_df.columns})
            
            # Clear existing treeview
            for item in self.summary_tree.get_children():
                self.summary_tree.delete(item)
            
            # Configure columns
            self.summary_tree['columns'] = list(summary_df.columns)
            for col in summary_df.columns:
                self.summary_tree.heading(col, text=col)
                self.summary_tree.column(col, width=100)
            
            # Add data
            for _, row in summary_df.iterrows():
                values = []
                for col in summary_df.columns:
                    val = row[col]
                    if isinstance(val, (int, float)):
                        if col in ['Accuracy', 'F1 Score', 'Precision', 'Recall']:
                            val = f"{val:.4f}"
                        elif col in ['Runtime (s)', 'CPU (%)', 'Memory (MB)', 'Power (W)']:
                            val = f"{val:.2f}"
                        else:
                            val = f"{val}"
                    values.append(val)
                
                self.summary_tree.insert('', tk.END, values=values)
        
        except Exception as e:
            logger.error(f"Error creating summary table: {str(e)}", exc_info=True)

def main():
    """Main entry point for metrics GUI application."""
    root = tk.Tk()
    app = MetricsGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()