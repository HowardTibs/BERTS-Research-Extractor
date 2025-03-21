import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import traceback
import logging
from pathlib import Path

# Add parent directory to path to import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from main_extractor import ResearchExtractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('extraction_gui.log')
    ]
)
logger = logging.getLogger(__name__)

class RedirectText:
    """Redirect print statements to tkinter text widget."""
    
    def __init__(self, text_widget):
        self.text_widget = text_widget
        self.buffer = ""
    
    def write(self, string):
        self.buffer += string
        self.text_widget.configure(state="normal")
        self.text_widget.insert(tk.END, string)
        self.text_widget.see(tk.END)
        self.text_widget.configure(state="disabled")
    
    def flush(self):
        self.text_widget.configure(state="normal")
        self.text_widget.insert(tk.END, self.buffer)
        self.buffer = ""
        self.text_widget.see(tk.END)
        self.text_widget.configure(state="disabled")

class ExtractorGUI:
    """GUI for Research Extractor application."""
    
    def __init__(self, root):
        """Initialize GUI."""
        self.root = root
        self.root.title("Research Paper Extractor")
        self.root.geometry("1000x700")
        self.root.minsize(800, 600)
        
        # Initialize extractor
        self.models_dir = tk.StringVar(value=os.path.join(os.getcwd(), "models"))
        self.output_dir = tk.StringVar(value=os.path.join(os.getcwd(), "output"))
        self.input_path = tk.StringVar()
        self.output_format = tk.StringVar(value="csv")
        self.recursive = tk.BooleanVar(value=False)
        
        self.extractor = None
        self.available_sections = []
        self.selected_sections = []
        
        # Create GUI elements
        self._create_widgets()
        
        # Initialize extractor in a separate thread
        self._initialize_extractor()
    
    def _create_widgets(self):
        """Create GUI widgets."""
        # Create main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create tabs
        tab_control = ttk.Notebook(main_frame)
        extraction_tab = ttk.Frame(tab_control)
        settings_tab = ttk.Frame(tab_control)
        help_tab = ttk.Frame(tab_control)
        
        tab_control.add(extraction_tab, text="Extraction")
        tab_control.add(settings_tab, text="Settings")
        tab_control.add(help_tab, text="Help")
        tab_control.pack(fill=tk.BOTH, expand=True)
        
        # === Extraction Tab ===
        extraction_frame = ttk.Frame(extraction_tab, padding="10")
        extraction_frame.pack(fill=tk.BOTH, expand=True)
        
        # Input section
        input_frame = ttk.LabelFrame(extraction_frame, text="Input", padding="10")
        input_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(input_frame, text="File or Directory:").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Entry(input_frame, textvariable=self.input_path, width=50).grid(row=0, column=1, sticky=tk.EW, padx=5, pady=5)
        ttk.Button(input_frame, text="Browse", command=self._browse_input).grid(row=0, column=2, sticky=tk.E, pady=5)
        
        ttk.Checkbutton(input_frame, text="Process subdirectories recursively", variable=self.recursive).grid(row=1, column=0, columnspan=3, sticky=tk.W, pady=5)
        
        # Output section
        output_frame = ttk.LabelFrame(extraction_frame, text="Output", padding="10")
        output_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(output_frame, text="Output Format:").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Radiobutton(output_frame, text="CSV", variable=self.output_format, value="csv").grid(row=0, column=1, sticky=tk.W, pady=5)
        ttk.Radiobutton(output_frame, text="JSON", variable=self.output_format, value="json").grid(row=0, column=2, sticky=tk.W, pady=5)
        
        ttk.Label(output_frame, text="Output Directory:").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Entry(output_frame, textvariable=self.output_dir, width=50).grid(row=1, column=1, columnspan=2, sticky=tk.EW, padx=5, pady=5)
        ttk.Button(output_frame, text="Browse", command=self._browse_output_dir).grid(row=1, column=3, sticky=tk.E, pady=5)
        
        # Sections selection
        sections_frame = ttk.LabelFrame(extraction_frame, text="Extract Sections", padding="10")
        sections_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.sections_listbox = tk.Listbox(sections_frame, selectmode=tk.MULTIPLE, height=8)
        self.sections_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        sections_scrollbar = ttk.Scrollbar(sections_frame, orient=tk.VERTICAL, command=self.sections_listbox.yview)
        sections_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.sections_listbox.config(yscrollcommand=sections_scrollbar.set)
        
        # Buttons
        buttons_frame = ttk.Frame(extraction_frame)
        buttons_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(buttons_frame, text="Select All", command=self._select_all_sections).pack(side=tk.LEFT, padx=5)
        ttk.Button(buttons_frame, text="Clear Selection", command=self._clear_section_selection).pack(side=tk.LEFT, padx=5)
        ttk.Button(buttons_frame, text="Extract", command=self._start_extraction).pack(side=tk.RIGHT, padx=5)
        
        # Output log
        log_frame = ttk.LabelFrame(extraction_frame, text="Log", padding="10")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, state="disabled", height=10)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # Redirect stdout and stderr to the log text widget
        self.stdout_redirect = RedirectText(self.log_text)
        sys.stdout = self.stdout_redirect
        sys.stderr = self.stdout_redirect
        
        # === Settings Tab ===
        settings_frame = ttk.Frame(settings_tab, padding="10")
        settings_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(settings_frame, text="Models Directory:").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Entry(settings_frame, textvariable=self.models_dir, width=50).grid(row=0, column=1, sticky=tk.EW, padx=5, pady=5)
        ttk.Button(settings_frame, text="Browse", command=self._browse_models_dir).grid(row=0, column=2, sticky=tk.E, pady=5)
        
        ttk.Button(settings_frame, text="Reload Models", command=self._initialize_extractor).grid(row=1, column=1, pady=10)
        
        # Model info frame
        model_info_frame = ttk.LabelFrame(settings_frame, text="Model Information", padding="10")
        model_info_frame.grid(row=2, column=0, columnspan=3, sticky=tk.NSEW, pady=10)
        settings_frame.rowconfigure(2, weight=1)
        settings_frame.columnconfigure(1, weight=1)
        
        self.model_info_text = scrolledtext.ScrolledText(model_info_frame, state="disabled", height=10)
        self.model_info_text.pack(fill=tk.BOTH, expand=True)
        
        # === Help Tab ===
        help_frame = ttk.Frame(help_tab, padding="10")
        help_frame.pack(fill=tk.BOTH, expand=True)
        
        help_text = scrolledtext.ScrolledText(help_frame, height=20, wrap=tk.WORD)
        help_text.pack(fill=tk.BOTH, expand=True)
        help_text.insert(tk.END, """
Research Paper Extractor Help

This application extracts structured information from research papers in various formats.

How to use:
1. Input: Select a file or directory of research papers to process.
2. Output: Choose between CSV or JSON output format and select an output directory.
3. Sections: Select which sections to extract. Leave unselected to extract all available sections.
4. Click "Extract" to start the extraction process.

Supported File Formats:
- PDF (both digital and scanned with OCR)
- Word documents (.doc, .docx)
- Text files (.txt)
- Images of documents (.jpg, .png, .tiff)

Extraction Capabilities:
The system can extract various sections from research papers, including title, authors, abstract, keywords, introduction, methodology, results, discussion, conclusion, and references.

Models:
The extraction uses three lightweight models:
- DistilBERT: A distilled version of BERT providing a good balance of performance and speed.
- TinyBERT: An even smaller model focused on efficiency.
- ALBERT: A lite BERT architecture with parameter reduction techniques.

The models are combined in an ensemble for improved accuracy.

Troubleshooting:
- Ensure the models directory contains the trained models.
- For scanned documents, extraction may be less accurate due to OCR limitations.
- If specific sections are not being extracted correctly, try processing the document without section filtering.
        """)
        help_text.configure(state="disabled")
    
    def _browse_input(self):
        """Browse for input file or directory."""
        path = filedialog.askopenfilename(
            title="Select Research Paper",
            filetypes=[
                ("All Supported Files", "*.pdf *.doc *.docx *.txt *.jpg *.jpeg *.png *.tiff *.tif"),
                ("PDF Files", "*.pdf"),
                ("Word Documents", "*.doc *.docx"),
                ("Text Files", "*.txt"),
                ("Image Files", "*.jpg *.jpeg *.png *.tiff *.tif"),
                ("All Files", "*.*")
            ]
        )
        if not path:
            # Try selecting a directory instead
            path = filedialog.askdirectory(title="Select Directory with Research Papers")
        
        if path:
            self.input_path.set(path)
    
    def _browse_output_dir(self):
        """Browse for output directory."""
        directory = filedialog.askdirectory(title="Select Output Directory")
        if directory:
            self.output_dir.set(directory)
    
    def _browse_models_dir(self):
        """Browse for models directory."""
        directory = filedialog.askdirectory(title="Select Models Directory")
        if directory:
            self.models_dir.set(directory)
    
    def _initialize_extractor(self):
        """Initialize the extractor in a separate thread."""
        self.log_message("Initializing research extractor...")
        
        # Disable UI elements during initialization
        self._set_ui_state(False)
        
        def init_thread():
            try:
                # Create output directory if it doesn't exist
                os.makedirs(self.output_dir.get(), exist_ok=True)
                
                # Initialize extractor
                self.extractor = ResearchExtractor(
                    models_dir=self.models_dir.get(),
                    output_dir=self.output_dir.get()
                )
                
                # Get available section types
                self.available_sections = self.extractor.get_available_section_types()
                
                # Update UI on main thread
                self.root.after(0, self._update_ui_after_init)
                
            except Exception as e:
                # Log error and update UI on main thread
                error_msg = f"Error initializing extractor: {str(e)}\n{traceback.format_exc()}"
                logger.error(error_msg)
                self.root.after(0, lambda: self._set_ui_state(True))
                self.root.after(0, lambda: self.log_message(error_msg))
        
        # Start initialization thread
        threading.Thread(target=init_thread, daemon=True).start()
    
    def _update_ui_after_init(self):
        """Update UI after extractor is initialized."""
        # Enable UI elements
        self._set_ui_state(True)
        
        # Update available sections in listbox
        self.sections_listbox.delete(0, tk.END)
        for section in self.available_sections:
            self.sections_listbox.insert(tk.END, section)
        
        # Update model info
        if self.extractor and self.extractor.models:
            model_info = f"Loaded {len(self.extractor.models)} models:\n"
            for i, model in enumerate(self.extractor.models):
                model_info += f"- Model {i+1}: {model.model_type}\n"
            
            self.model_info_text.configure(state="normal")
            self.model_info_text.delete(1.0, tk.END)
            self.model_info_text.insert(tk.END, model_info)
            self.model_info_text.configure(state="disabled")
            
            self.log_message(f"Extractor initialized with {len(self.extractor.models)} models")
        else:
            self.model_info_text.configure(state="normal")
            self.model_info_text.delete(1.0, tk.END)
            self.model_info_text.insert(tk.END, "No models loaded. Extraction will use basic heuristics.")
            self.model_info_text.configure(state="disabled")
            
            self.log_message("Extractor initialized with no models. Using fallback extraction.")
    
    def _set_ui_state(self, enabled):
        """Enable or disable UI elements."""
        state = "normal" if enabled else "disabled"
        for widget in self.root.winfo_children():
            if isinstance(widget, ttk.Frame):
                for child in widget.winfo_children():
                    if isinstance(child, (ttk.Button, ttk.Entry, ttk.Radiobutton, ttk.Checkbutton)):
                        child.configure(state=state)
    
    def _select_all_sections(self):
        """Select all sections in the listbox."""
        self.sections_listbox.select_set(0, tk.END)
    
    def _clear_section_selection(self):
        """Clear all section selections."""
        self.sections_listbox.selection_clear(0, tk.END)
    
    def _get_selected_sections(self):
        """Get list of selected sections."""
        selected_indices = self.sections_listbox.curselection()
        return [self.sections_listbox.get(i) for i in selected_indices]
    
    def _start_extraction(self):
        """Start extraction process in a separate thread."""
        # Get input path
        input_path = self.input_path.get()
        if not input_path:
            messagebox.showerror("Error", "Please select an input file or directory")
            return
        
        if not os.path.exists(input_path):
            messagebox.showerror("Error", f"Input path does not exist: {input_path}")
            return
        
        # Get selected sections
        selected_sections = self._get_selected_sections()
        
        # Disable UI during extraction
        self._set_ui_state(False)
        
        self.log_message(f"Starting extraction from: {input_path}")
        self.log_message(f"Output format: {self.output_format.get()}")
        if selected_sections:
            self.log_message(f"Extracting sections: {', '.join(selected_sections)}")
        else:
            self.log_message("Extracting all available sections")
        
        def extraction_thread():
            try:
                # Process input
                if os.path.isfile(input_path):
                    # Process single file
                    sections, output_path = self.extractor.process_file(
                        input_path, 
                        self.output_format.get(), 
                        selected_sections if selected_sections else None
                    )
                    
                    # Update UI on main thread
                    if sections:
                        self.root.after(0, lambda: self.log_message(f"Extracted sections: {', '.join(sections.keys())}"))
                        self.root.after(0, lambda: self.log_message(f"Output saved to: {output_path}"))
                        self.root.after(0, lambda: messagebox.showinfo("Extraction Complete", f"Extracted {len(sections)} sections and saved to:\n{output_path}"))
                    else:
                        self.root.after(0, lambda: self.log_message("No sections were extracted"))
                        self.root.after(0, lambda: messagebox.showwarning("Extraction Complete", "No sections were extracted from the file"))
                
                elif os.path.isdir(input_path):
                    # Process directory
                    processed_files = self.extractor.process_directory(
                        input_path, 
                        self.output_format.get(), 
                        selected_sections if selected_sections else None,
                        self.recursive.get()
                    )
                    
                    # Update UI on main thread
                    self.root.after(0, lambda: self.log_message(f"Processed {len(processed_files)} files"))
                    self.root.after(0, lambda: messagebox.showinfo("Extraction Complete", f"Processed {len(processed_files)} files\nOutput saved to: {self.output_dir.get()}"))
            
            except Exception as e:
                # Log error and update UI on main thread
                error_msg = f"Error during extraction: {str(e)}\n{traceback.format_exc()}"
                logger.error(error_msg)
                self.root.after(0, lambda: self.log_message(error_msg))
                self.root.after(0, lambda: messagebox.showerror("Error", f"Error during extraction: {str(e)}"))
            
            finally:
                # Re-enable UI on main thread
                self.root.after(0, lambda: self._set_ui_state(True))
        
        # Start extraction thread
        threading.Thread(target=extraction_thread, daemon=True).start()
    
    def log_message(self, message):
        """Add message to log text widget."""
        self.log_text.configure(state="normal")
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state="disabled")
        logger.info(message)

def main():
    """Main entry point for GUI application."""
    root = tk.Tk()
    app = ExtractorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()