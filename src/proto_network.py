import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from torchmetrics import Accuracy, F1Score
from sklearn.model_selection import train_test_split
import numpy as np
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

# Import easyfsl components
from easyfsl.methods import PrototypicalNetworks
from easyfsl.samplers import TaskSampler

class SimpleNN(nn.Module):
    """Simple linear projection network for embedding transformation"""
    def __init__(self, in_dim=1024, out_dim=512):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.layers = nn.Sequential(
            nn.Linear(in_features=in_dim, out_features=out_dim),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.layers(x)

class EmbeddingsDataset(Dataset):
    """Dataset for in-memory embeddings"""
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels
        
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]
    
    def get_labels(self):
        """Return all labels - required by easyfsl TaskSampler"""
        return self.labels.tolist()

class ProtoNetworkModel(pl.LightningModule):
    """PyTorch Lightning module for Prototypical Network"""
    def __init__(
        self,
        backbone_model,
        n_way=5,
        embedding_dim=1024,
        projection_dim=512,
        lr=1e-4,
        **kwargs
    ):
        super().__init__()
        self.backbone_model = backbone_model
        self.n_way = n_way
        self.lr = lr
        self.embedding_dim = embedding_dim
        self.projection_dim = projection_dim
        
        # Build the prototypical network with the provided backbone
        self.model = PrototypicalNetworks(self.backbone_model)
        
        self.save_hyperparameters(ignore=['backbone_model'])
        
        # Metrics
        self.train_acc = Accuracy(task="multiclass", num_classes=self.n_way)
        self.valid_acc = Accuracy(task="multiclass", num_classes=self.n_way)
        self.train_f1 = F1Score(task="multiclass", num_classes=self.n_way)
        self.valid_f1 = F1Score(task="multiclass", num_classes=self.n_way)
        
        # Store prototypes
        self.prototypes = None
    
    def forward(self, support_images, support_labels, query_images):
        self.model.process_support_set(support_images, support_labels)
        classification_scores = self.model(query_images)
        return classification_scores
    
    def training_step(self, batch, batch_idx):
        support_images, support_labels, query_images, query_labels, _ = batch
        
        # Ensure tensors are on the correct device
        support_images = support_images.to(self.device)
        support_labels = support_labels.to(self.device)
        query_images = query_images.to(self.device)
        query_labels = query_labels.to(self.device)
        
        classification_scores = self.forward(support_images, support_labels, query_images)
        train_loss = nn.functional.cross_entropy(classification_scores, query_labels)
        
        # Metrics
        predicted_labels = torch.max(classification_scores, 1)[1]
        self.log("train_loss", train_loss, prog_bar=True)
        self.log("train_acc", self.train_acc(predicted_labels, query_labels), prog_bar=True)
        self.log("train_f1", self.train_f1(predicted_labels, query_labels), prog_bar=True)
        
        return train_loss
    
    def validation_step(self, batch, batch_idx):
        support_images, support_labels, query_images, query_labels, _ = batch
        
        support_images = support_images.to(self.device)
        support_labels = support_labels.to(self.device)
        query_images = query_images.to(self.device)
        query_labels = query_labels.to(self.device)
        
        classification_scores = self.forward(support_images, support_labels, query_images)
        val_loss = nn.functional.cross_entropy(classification_scores, query_labels)
        
        # Metrics
        predicted_labels = torch.max(classification_scores, 1)[1]
        self.log("val_loss", val_loss, prog_bar=True)
        self.log("val_acc", self.valid_acc(predicted_labels, query_labels), prog_bar=True)
        self.log("val_f1", self.valid_f1(predicted_labels, query_labels), prog_bar=True)
        
        # Store the prototypes
        self.prototypes = self.model.prototypes
        
        return val_loss
    
    def configure_optimizers(self):
        return optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            betas=(0.9, 0.98),
            weight_decay=0.01
        )
    
    def get_all_prototypes(self, embeddings, labels):
        self.model.process_support_set(embeddings, labels)
        return self.model.prototypes

def train_proto_network(embeddings, labels, embedding_dim=512, n_way=5, n_shot=3, n_query=2, epochs=30, learning_rate=0.001, progress_callback=None):

    # Convert string labels to numeric IDs
    unique_labels = sorted(set(labels))
    label_to_id = {label: i for i, label in enumerate(unique_labels)}
    numeric_labels = np.array([label_to_id[label] for label in labels])
    
    # Convert to PyTorch tensors
    X_tensor = torch.FloatTensor(embeddings)
    y_tensor = torch.LongTensor(numeric_labels)
    
    # Create train/val split - stratified to ensure all classes in both sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_tensor, y_tensor, test_size=0.2, stratify=y_tensor, random_state=42
    )
    
    # Create datasets
    train_dataset = EmbeddingsDataset(X_train, y_train)
    val_dataset = EmbeddingsDataset(X_val, y_val)
    
    # Create few-shot task samplers
    n_train_tasks = 100
    n_val_tasks = 20
    
    # Check we have enough samples per class
    class_counts = {i.item(): (y_train == i).sum().item() for i in torch.unique(y_train)}
    min_samples = n_shot + n_query
    valid_classes = len([c for c in class_counts.values() if c >= min_samples])
    
    if valid_classes < n_way:
        print(f"Warning: Not enough classes with {min_samples} samples. Using {valid_classes} classes.")
        n_way = max(2, valid_classes)
    
    train_sampler = TaskSampler(
        train_dataset, 
        n_way=n_way, 
        n_shot=n_shot, 
        n_query=n_query, 
        n_tasks=n_train_tasks
    )
    
    val_sampler = TaskSampler(
        val_dataset, 
        n_way=n_way, 
        n_shot=n_shot, 
        n_query=n_query, 
        n_tasks=n_val_tasks
    )
    
    # Create episodic data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=2,
        pin_memory=False,
        collate_fn=train_sampler.episodic_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_sampler=val_sampler,
        num_workers=2,
        pin_memory=False,
        collate_fn=val_sampler.episodic_collate_fn
    )
    
    # Create backbone model and prototypical network
    input_dim = embeddings.shape[1]
    backbone = SimpleNN(in_dim=input_dim, out_dim=embedding_dim)
    model = ProtoNetworkModel(
        backbone_model=backbone,
        n_way=n_way,
        embedding_dim=input_dim,
        projection_dim=embedding_dim,
        lr=learning_rate
    )
    
    # Create progress callback
    class ProgressCallback(pl.Callback):
        def on_train_epoch_start(self, trainer, pl_module):
            if progress_callback:
                progress_callback(trainer.current_epoch + 1, epochs)
    

    callbacks = [ProgressCallback(),
                 EarlyStopping(monitor="val_loss", min_delta=0.00, patience=3, verbose=False, mode="min")]

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=epochs,
        enable_progress_bar=False,
        enable_checkpointing=False,
        logger=False,
        callbacks=callbacks,
        accelerator="cuda" if torch.cuda.is_available() else "cpu",
    )
    
    # Train the model
    trainer.fit(model, train_loader, val_loader)
    
    # Get transformed embeddings and prototypes for all data
    dataset = EmbeddingsDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=128)
    
    # Transform all embeddings
    model.eval()
    transformed_embeddings = []
    with torch.no_grad():
        for batch_x, _ in loader:
            batch_transformed = backbone(batch_x)
            transformed_embeddings.append(batch_transformed)
    
    transformed_embeddings = torch.cat(transformed_embeddings)
    prototypes = model.get_all_prototypes(X_tensor, y_tensor)
    
    return model, label_to_id, prototypes

def get_proto_transformed_embeddings(model, embeddings):
    """Transform embeddings using the prototypical network backbone"""
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(embeddings)
        # Use the backbone model which is inside the prototypical network
        transformed = model.backbone_model(X_tensor)
    return transformed.numpy()