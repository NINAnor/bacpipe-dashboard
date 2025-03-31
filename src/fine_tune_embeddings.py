import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback


class ProgressBarCallback(Callback):
    """Custom callback to report progress to Streamlit"""
    
    def __init__(self, progress_callback=None):
        super().__init__()
        self.progress_callback = progress_callback
        self.total_epochs = 0
    
    def on_fit_start(self, trainer, pl_module):
        self.total_epochs = trainer.max_epochs
    
    def on_train_epoch_start(self, trainer, pl_module):
        if self.progress_callback:
            self.progress_callback(trainer.current_epoch + 1, self.total_epochs)


class EmbeddingTransformerModule(pl.LightningModule):

    def __init__(self, input_dim, hidden_dim, num_classes, learning_rate=0.001):
        super().__init__()
        self.save_hyperparameters()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
        self.learning_rate = learning_rate
    
    def forward(self, x):
        hidden = self.fc1(x)
        activated = self.relu(hidden)
        output = self.fc2(activated)
        return output
    
    def get_embedding(self, x):
        return self.relu(self.fc1(x))
    
    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = nn.functional.cross_entropy(outputs, targets)
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = nn.functional.cross_entropy(outputs, targets)
        _, predicted = outputs.max(1)
        correct = predicted.eq(targets).sum().item()
        total = targets.size(0)
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', correct / total, prog_bar=True)
        
        return {'val_loss': loss, 'correct': correct, 'total': total}
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)


def train_embedding_model(embeddings, labels, hidden_dim=64, epochs=30, batch_size=32, learning_rate=0.001, progress_callback=None):
    """
    Train a simple neural network on embeddings to improve class separation using PyTorch Lightning
    
    Args:
        embeddings: Original embeddings (numpy array)
        labels: String labels for each embedding
        hidden_dim: Size of the hidden layer
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for Adam optimizer
        progress_callback: Optional callback function to report progress (epoch, total)
    
    Returns:
        model: Trained PyTorch Lightning model
        label_to_id: Dictionary mapping label strings to numeric IDs
    """
    # Convert string labels to numeric IDs
    unique_labels = sorted(set(labels))
    label_to_id = {label: i for i, label in enumerate(unique_labels)}
    numeric_labels = np.array([label_to_id[label] for label in labels])
    
    # Convert to PyTorch tensors
    X_tensor = torch.FloatTensor(embeddings)
    y_tensor = torch.LongTensor(numeric_labels)
    
    # Create train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X_tensor, y_tensor, test_size=0.2, stratify=y_tensor, random_state=42
    )
    
    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Create model
    input_dim = embeddings.shape[1]
    num_classes = len(unique_labels)
    model = EmbeddingTransformerModule(input_dim, hidden_dim, num_classes, learning_rate)
    
    # Create progress callback
    callbacks = [ProgressBarCallback(progress_callback)]
    
    # Create trainer and train
    trainer = pl.Trainer(
        max_epochs=epochs,
        enable_progress_bar=False,  # We'll use our own progress reporting
        enable_checkpointing=False,  # No model saving needed
        logger=False,  # No logging needed
        callbacks=callbacks
    )
    
    # Train the model
    trainer.fit(model, train_loader, val_loader)
    
    return model, label_to_id


def get_transformed_embeddings(model, embeddings):

    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(embeddings)
        transformed = model.get_embedding(X_tensor)
    return transformed.numpy()