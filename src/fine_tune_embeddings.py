import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


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


def prepare_embedding_data(embeddings, labels, test_size=0.2):

    unique_labels = sorted(set(labels))
    label_to_id = {label: i for i, label in enumerate(unique_labels)}
    numeric_labels = np.array([label_to_id[label] for label in labels])

    X_tensor = torch.FloatTensor(embeddings)
    y_tensor = torch.LongTensor(numeric_labels)

    X_train, X_val, y_train, y_val = train_test_split(
        X_tensor, y_tensor, test_size=test_size, stratify=y_tensor, random_state=42
    )

    return {
        "X_train": X_train,
        "X_val": X_val, 
        "y_train": y_train,
        "y_val": y_val,
        "label_to_id": label_to_id,
        "unique_labels": unique_labels,
        "input_dim": embeddings.shape[1]
    }


def train_embedding_model(
    embeddings,
    labels,
    test_size=0.2,
    hidden_dim=64,
    epochs=30,
    batch_size=32,
    learning_rate=0.001,
    progress_callback=None,
):
    
    data = prepare_embedding_data(embeddings, labels, test_size)
    
    train_dataset = TensorDataset(data["X_train"], data["y_train"])
    val_dataset = TensorDataset(data["X_val"], data["y_val"])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    input_dim = data["input_dim"]
    num_classes = len(data["unique_labels"])
    model = EmbeddingTransformerModule(
        input_dim, hidden_dim, num_classes, learning_rate
    )
    
    callbacks = [ProgressBarCallback(progress_callback),
                 EarlyStopping(monitor="val_loss", min_delta=0.00, patience=3, verbose=False, mode="min")]

    trainer = pl.Trainer(
        max_epochs=epochs,
        enable_progress_bar=False,
        enable_checkpointing=False,
        logger=False,
        callbacks=callbacks,
    )

    trainer.fit(model, train_loader, val_loader)

    return model, data["label_to_id"]


def get_transformed_embeddings(model, embeddings):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(embeddings)
        transformed = model.get_embedding(X_tensor)
    return transformed.numpy()
