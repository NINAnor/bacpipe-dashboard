import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from easyfsl.methods import PrototypicalNetworks
from easyfsl.samplers import TaskSampler
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.utils.data import DataLoader, Dataset
from torchmetrics import Accuracy, F1Score


class SimpleNN(nn.Module):
    """Simple linear projection network for embedding transformation"""

    def __init__(self, in_dim=1024, out_dim=512):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.layers = nn.Sequential(
            nn.Linear(in_features=in_dim, out_features=out_dim), nn.ReLU()
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
        **kwargs,
    ):
        super().__init__()
        self.backbone_model = backbone_model
        self.n_way = n_way
        self.lr = lr
        self.embedding_dim = embedding_dim
        self.projection_dim = projection_dim

        self.model = PrototypicalNetworks(self.backbone_model)

        self.save_hyperparameters(ignore=["backbone_model"])

        self.train_acc = Accuracy(task="multiclass", num_classes=self.n_way)
        self.valid_acc = Accuracy(task="multiclass", num_classes=self.n_way)
        self.train_f1 = F1Score(task="multiclass", num_classes=self.n_way)
        self.valid_f1 = F1Score(task="multiclass", num_classes=self.n_way)

        self.prototypes = None

    def forward(self, support_images, support_labels, query_images):
        self.model.process_support_set(support_images, support_labels)
        classification_scores = self.model(query_images)
        return classification_scores

    def training_step(self, batch, batch_idx):
        support_images, support_labels, query_images, query_labels, _ = batch

        support_images = support_images.to(self.device)
        support_labels = support_labels.to(self.device)
        query_images = query_images.to(self.device)
        query_labels = query_labels.to(self.device)

        classification_scores = self.forward(
            support_images, support_labels, query_images
        )
        train_loss = nn.functional.cross_entropy(classification_scores, query_labels)

        # Metrics
        predicted_labels = torch.max(classification_scores, 1)[1]
        self.log("train_loss", train_loss, prog_bar=True)
        self.log(
            "train_acc", self.train_acc(predicted_labels, query_labels), prog_bar=True
        )
        self.log(
            "train_f1", self.train_f1(predicted_labels, query_labels), prog_bar=True
        )

        return train_loss

    def validation_step(self, batch, batch_idx):
        support_images, support_labels, query_images, query_labels, _ = batch

        support_images = support_images.to(self.device)
        support_labels = support_labels.to(self.device)
        query_images = query_images.to(self.device)
        query_labels = query_labels.to(self.device)

        classification_scores = self.forward(
            support_images, support_labels, query_images
        )
        val_loss = nn.functional.cross_entropy(classification_scores, query_labels)

        predicted_labels = torch.max(classification_scores, 1)[1]
        self.log("val_loss", val_loss, prog_bar=True)
        self.log(
            "val_acc", self.valid_acc(predicted_labels, query_labels), prog_bar=True
        )
        self.log("val_f1", self.valid_f1(predicted_labels, query_labels), prog_bar=True)

        self.prototypes = self.model.prototypes

        return val_loss

    def configure_optimizers(self):
        return optim.AdamW(
            self.model.parameters(), lr=self.lr, betas=(0.9, 0.98), weight_decay=0.01
        )

    def get_all_prototypes(self, embeddings, labels):
        self.model.process_support_set(embeddings, labels)
        return self.model.prototypes


def train_proto_network(
    train_X,
    train_y,
    val_X,
    val_y,
    embedding_dim=512,
    n_way=5,
    n_shot=3,
    n_query=2,
    epochs=30,
    learning_rate=0.001,
    n_train_tasks=100,
    n_val_tasks=20,
):
    train_dataset = EmbeddingsDataset(train_X, train_y)
    val_dataset = EmbeddingsDataset(val_X, val_y)

    # Check we have enough samples per class for episodic training
    class_counts = {
        i.item(): (train_y == i).sum().item() for i in torch.unique(train_y)
    }
    min_samples = n_shot + n_query
    valid_classes = len([c for c in class_counts.values() if c >= min_samples])

    if valid_classes < n_way:
        print(
            f"Warning: Not enough classes with {min_samples} samples."
            f"Using {valid_classes} classes."
        )
        n_way = max(2, valid_classes)

    train_sampler = TaskSampler(
        train_dataset,
        n_way=n_way,
        n_shot=n_shot,
        n_query=n_query,
        n_tasks=n_train_tasks,
    )

    val_sampler = TaskSampler(
        val_dataset, n_way=n_way, n_shot=n_shot, n_query=n_query, n_tasks=n_val_tasks
    )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=2,
        pin_memory=False,
        collate_fn=train_sampler.episodic_collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_sampler=val_sampler,
        num_workers=2,
        pin_memory=False,
        collate_fn=val_sampler.episodic_collate_fn,
    )

    # Create backbone model and prototypical network
    input_dim = train_X.shape[1]
    backbone = SimpleNN(in_dim=input_dim, out_dim=embedding_dim)
    model = ProtoNetworkModel(
        backbone_model=backbone,
        n_way=n_way,
        embedding_dim=input_dim,
        projection_dim=embedding_dim,
        lr=learning_rate,
    )

    callbacks = [
        EarlyStopping(
            monitor="val_loss", min_delta=0.00, patience=3, verbose=False, mode="min"
        )
    ]

    trainer = pl.Trainer(
        max_epochs=epochs,
        enable_progress_bar=False,
        enable_checkpointing=False,
        logger=False,
        callbacks=callbacks,
        accelerator="cuda" if torch.cuda.is_available() else "cpu",
        devices=1,
    )
    trainer.fit(model, train_loader, val_loader)

    # get prototypes for all classes
    all_data = torch.cat([train_X, val_X])
    all_labels = torch.cat([train_y, val_y])

    # Create a mapping from numeric IDs to strings (will be provided by main.py)
    label_map = {i: i for i in range(len(torch.unique(all_labels)))}

    # Get prototypes using all data
    prototypes = model.get_all_prototypes(all_data, all_labels)

    return model, label_map, prototypes


def get_proto_transformed_embeddings(model, embeddings):
    """Transform embeddings using the prototypical network backbone"""
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(embeddings)
        transformed = model.backbone_model(X_tensor)
    return transformed.numpy()
