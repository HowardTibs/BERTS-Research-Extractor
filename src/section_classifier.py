import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from transformers import (
    DistilBertTokenizer, DistilBertForSequenceClassification,
    BertTokenizer, BertForSequenceClassification,
    AlbertTokenizer, AlbertForSequenceClassification,
    AdamW, get_linear_schedule_with_warmup
)
from sklearn.metrics import classification_report, confusion_matrix
import logging
from tqdm import tqdm
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SectionClassificationDataset(Dataset):
    """Dataset for section classification."""
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        """
        Initialize dataset.
        
        Args:
            texts (list): List of text strings
            labels (list): List of label indices
            tokenizer: Transformer tokenizer
            max_length (int): Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class SectionClassifier:
    """Base class for section classification models."""
    
    def __init__(self, model_type, num_labels=7, model_path=None):
        """
        Initialize section classifier.
        
        Args:
            model_type (str): Type of model ('distilbert', 'tinybert', or 'albert')
            num_labels (int): Number of section labels
            model_path (str): Path to pre-trained model, if available
        """
        self.model_type = model_type.lower()
        self.num_labels = num_labels
        
        # Define label mapping
        self.id2label = {
            0: "title",
            1: "authors",
            2: "abstract",
            3: "keywords",
            4: "introduction",
            5: "methodology",
            6: "other"
        }
        self.label2id = {v: k for k, v in self.id2label.items()}
        
        # For custom label sets
        if num_labels != len(self.id2label):
            logger.warning(f"Custom label set with {num_labels} labels. Default mappings will be overridden.")
        
        # Set up device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize model and tokenizer
        self._initialize_model(model_path)
    
    def _initialize_model(self, model_path):
        """Initialize the appropriate model and tokenizer."""
        raise NotImplementedError("Subclasses must implement this method")
    
    def train(self, train_dataloader, val_dataloader=None, epochs=3, lr=5e-5, warmup_steps=0):
        """
        Fine-tune the model.
        
        Args:
            train_dataloader: DataLoader for training data
            val_dataloader: DataLoader for validation data
            epochs (int): Number of training epochs
            lr (float): Learning rate
            warmup_steps (int): Number of warmup steps for scheduler
        
        Returns:
            dict: Training statistics
        """
        logger.info(f"Starting training of {self.model_type} model")
        
        # Prepare optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=lr)
        
        # Set up learning rate scheduler
        total_steps = len(train_dataloader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Training loop
        train_stats = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        # Global training progress bar
        global_progress = tqdm(range(epochs), desc="Training progress")
        
        for epoch in range(epochs):
            # Set model to training mode
            self.model.train()
            total_train_loss = 0
            
            # Progress bar for batches
            batch_progress = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}")
            
            for batch in batch_progress:
                # Clear gradients
                self.model.zero_grad()
                
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_train_loss += loss.item()
                
                # Backward pass
                loss.backward()
                
                # Update parameters
                optimizer.step()
                
                # Update learning rate
                scheduler.step()
                
                # Update progress bar
                batch_progress.set_postfix({'loss': loss.item()})
            
            # Calculate average training loss
            avg_train_loss = total_train_loss / len(train_dataloader)
            train_stats['train_loss'].append(avg_train_loss)
            
            logger.info(f"Epoch {epoch+1}/{epochs} - Training loss: {avg_train_loss:.4f}")
            
            # Validation
            if val_dataloader:
                val_loss, val_accuracy = self.evaluate(val_dataloader)
                train_stats['val_loss'].append(val_loss)
                train_stats['val_accuracy'].append(val_accuracy)
                
                logger.info(f"Validation loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")
            
            # Update global progress bar
            global_progress.update(1)
        
        logger.info(f"Training of {self.model_type} model completed")
        return train_stats
    
    def evaluate(self, dataloader):
        """
        Evaluate model on validation data.
        
        Args:
            dataloader: DataLoader for validation data
        
        Returns:
            tuple: (average loss, accuracy)
        """
        logger.info(f"Evaluating {self.model_type} model")
        
        # Set model to evaluation mode
        self.model.eval()
        
        total_loss = 0
        all_labels = []
        all_preds = []
        
        # Evaluation loop
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass (no gradient calculation needed)
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
            
            # Add batch loss
            total_loss += outputs.loss.item()
            
            # Get predictions
            preds = torch.argmax(outputs.logits, dim=1)
            
            # Add to lists for metrics calculation
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(dataloader)
        accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
        
        # Detailed metrics
        logger.info("\nClassification Report:")
        logger.info(classification_report(
            all_labels, all_preds, 
            target_names=[self.id2label[i] for i in range(self.num_labels)]
        ))
        
        return avg_loss, accuracy
    
    def predict(self, texts, batch_size=16):
        """
        Predict section labels for texts.
        
        Args:
            texts (list): List of text strings
            batch_size (int): Batch size for prediction
        
        Returns:
            list: Predicted section labels
        """
        logger.info(f"Making predictions with {self.model_type} model")
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Create a simple dataloader for prediction
        inputs = []
        for text in texts:
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=512,
                return_tensors='pt'
            )
            inputs.append({
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten()
            })
        
        # Prediction loop
        all_preds = []
        
        for i in range(0, len(inputs), batch_size):
            batch = inputs[i:i+batch_size]
            
            # Combine batch elements
            input_ids = torch.stack([item['input_ids'] for item in batch]).to(self.device)
            attention_mask = torch.stack([item['attention_mask'] for item in batch]).to(self.device)
            
            # Make predictions
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Get class predictions
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
        
        # Convert indices to labels
        return [self.id2label[pred] for pred in all_preds]
    
    def save_model(self, output_dir):
        """
        Save model and tokenizer.
        
        Args:
            output_dir (str): Directory to save model
        """
        logger.info(f"Saving {self.model_type} model to {output_dir}")
        
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model and tokenizer
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # Save label mapping
        with open(os.path.join(output_dir, 'label_mapping.json'), 'w') as f:
            json.dump({
                'id2label': self.id2label,
                'label2id': self.label2id
            }, f)
        
        logger.info(f"Model saved successfully")
    
    def load_model(self, model_path):
        """
        Load pre-trained model and tokenizer.
        
        Args:
            model_path (str): Path to model directory
        """
        logger.info(f"Loading {self.model_type} model from {model_path}")
        
        # Load label mapping if available
        label_map_path = os.path.join(model_path, 'label_mapping.json')
        if os.path.exists(label_map_path):
            with open(label_map_path, 'r') as f:
                label_mapping = json.load(f)
                self.id2label = label_mapping['id2label']
                self.label2id = label_mapping['label2id']
                # Convert string keys to integers for id2label
                self.id2label = {int(k): v for k, v in self.id2label.items()}
                self.num_labels = len(self.id2label)
                logger.info(f"Loaded label mapping with {self.num_labels} labels")

class DistilBERTClassifier(SectionClassifier):
    """DistilBERT model for section classification."""
    
    def _initialize_model(self, model_path):
        """Initialize DistilBERT model and tokenizer."""
        logger.info("Initializing DistilBERT model")
        
        if model_path and os.path.exists(model_path):
            logger.info(f"Loading pre-trained DistilBERT model from {model_path}")
            self.tokenizer = DistilBertTokenizer.from_pretrained(model_path)
            self.model = DistilBertForSequenceClassification.from_pretrained(
                model_path,
                num_labels=self.num_labels,
                id2label=self.id2label,
                label2id=self.label2id
            )
        else:
            logger.info("Initializing DistilBERT from pre-trained weights")
            self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
            self.model = DistilBertForSequenceClassification.from_pretrained(
                'distilbert-base-uncased',
                num_labels=self.num_labels,
                id2label=self.id2label,
                label2id=self.label2id
            )
        
        self.model.to(self.device)

class TinyBERTClassifier(SectionClassifier):
    """TinyBERT model for section classification."""
    
    def _initialize_model(self, model_path):
        """Initialize TinyBERT model and tokenizer."""
        logger.info("Initializing TinyBERT model")
        
        if model_path and os.path.exists(model_path):
            logger.info(f"Loading pre-trained TinyBERT model from {model_path}")
            self.tokenizer = BertTokenizer.from_pretrained(model_path)
            self.model = BertForSequenceClassification.from_pretrained(
                model_path,
                num_labels=self.num_labels,
                id2label=self.id2label,
                label2id=self.label2id
            )
        else:
            logger.info("Initializing TinyBERT from pre-trained weights")
            # TinyBERT uses BERT tokenizer
            self.tokenizer = BertTokenizer.from_pretrained('huawei-noah/TinyBERT_General_4L_312D')
            self.model = BertForSequenceClassification.from_pretrained(
                'huawei-noah/TinyBERT_General_4L_312D',
                num_labels=self.num_labels,
                id2label=self.id2label,
                label2id=self.label2id
            )
        
        self.model.to(self.device)

class ALBERTClassifier(SectionClassifier):
    """ALBERT model for section classification."""
    
    def _initialize_model(self, model_path):
        """Initialize ALBERT model and tokenizer."""
        logger.info("Initializing ALBERT model")
        
        if model_path and os.path.exists(model_path):
            logger.info(f"Loading pre-trained ALBERT model from {model_path}")
            self.tokenizer = AlbertTokenizer.from_pretrained(model_path)
            self.model = AlbertForSequenceClassification.from_pretrained(
                model_path,
                num_labels=self.num_labels,
                id2label=self.id2label,
                label2id=self.label2id
            )
        else:
            logger.info("Initializing ALBERT from pre-trained weights")
            self.tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
            self.model = AlbertForSequenceClassification.from_pretrained(
                'albert-base-v2',
                num_labels=self.num_labels,
                id2label=self.id2label,
                label2id=self.label2id
            )
        
        self.model.to(self.device)

class EnsembleClassifier:
    """Ensemble of models for section classification."""
    
    def __init__(self, models):
        """
        Initialize ensemble classifier.
        
        Args:
            models (list): List of SectionClassifier instances
        """
        self.models = models
        logger.info(f"Initialized ensemble with {len(models)} models")
        
        # Use the first model's label mapping
        self.id2label = models[0].id2label
        self.label2id = models[0].label2id
    
    def predict(self, texts, batch_size=16):
        """
        Make predictions using ensemble voting.
        
        Args:
            texts (list): List of text strings
            batch_size (int): Batch size for prediction
        
        Returns:
            list: Predicted section labels
        """
        logger.info(f"Making predictions with ensemble of {len(self.models)} models")
        
        # Get predictions from each model
        all_model_preds = []
        for model in self.models:
            preds = model.predict(texts, batch_size)
            all_model_preds.append(preds)
        
        # Transpose to get predictions per text
        per_text_preds = list(zip(*all_model_preds))
        
        # Get majority vote for each text
        from collections import Counter
        
        final_preds = []
        for preds in per_text_preds:
            counter = Counter(preds)
            majority_label = counter.most_common(1)[0][0]
            final_preds.append(majority_label)
        
        return final_preds