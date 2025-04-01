import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


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
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = nn.functional.cross_entropy(outputs, targets)
        _, predicted = outputs.max(1)
        correct = predicted.eq(targets).sum().item()
        total = targets.size(0)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", correct / total, prog_bar=True)

        return {"val_loss": loss, "correct": correct, "total": total}

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)


def train_embedding_model(
    train_X,
    train_y,
    val_X,
    val_y,
    hidden_dim=64,
    epochs=30,
    batch_size=32,
    learning_rate=0.001,
    progress_callback=None,
    num_classes=None
):

    train_dataset = TensorDataset(train_X, train_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = TensorDataset(val_X, val_y)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    if num_classes is None:
        num_classes = len(torch.unique(train_y))
    
    # Get the shape of the embeddings and create the model
    input_dim = train_X.shape[1]
    model = EmbeddingTransformerModule(
        input_dim, hidden_dim, num_classes, learning_rate
    )
    
    callbacks = [EarlyStopping(monitor="val_loss", min_delta=0.00, patience=3, verbose=False, mode="min")
    ]
    
    trainer = pl.Trainer(
        max_epochs=epochs,
        enable_progress_bar=False,
        enable_checkpointing=False,
        logger=False,
        callbacks=callbacks,
        accelerator="cpu",
        devices=1
    )
    
    trainer.fit(model, train_loader, val_loader)
    
    return model


def get_transformed_embeddings(model, embeddings):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(embeddings)
        transformed = model.get_embedding(X_tensor)
    return transformed.numpy()
