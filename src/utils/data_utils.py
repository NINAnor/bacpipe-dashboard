import numpy as np
import torch
from sklearn.model_selection import train_test_split


def split_data(embeddings, labels, files, test_size=0.2):
    """Split data into training and validation sets"""
    # Convert string labels to numeric IDs for stratification
    unique_labels = sorted(set(labels))
    label_to_id = {label: i for i, label in enumerate(unique_labels)}
    numeric_labels = np.array([label_to_id[label] for label in labels])

    # Create stratified train/val split
    X_train, X_val, y_train_idx, y_val_idx, train_files, val_files = train_test_split(
        embeddings,
        numeric_labels,
        files,
        test_size=test_size,
        stratify=numeric_labels,
        random_state=42,
    )

    # Convert numeric labels back to original strings
    y_train = [list(unique_labels)[idx] for idx in y_train_idx]
    y_val = [list(unique_labels)[idx] for idx in y_val_idx]

    return X_train, X_val, y_train, y_val, train_files, val_files


def prepare_embedding_data(embeddings, labels):
    unique_labels = sorted(set(labels))
    label_to_id = {label: i for i, label in enumerate(unique_labels)}
    numeric_labels = np.array([label_to_id[label] for label in labels])

    X_tensor = torch.FloatTensor(embeddings)
    y_tensor = torch.LongTensor(numeric_labels)

    return {
        "X": X_tensor,
        "y": y_tensor,
        "label_to_id": label_to_id,
        "unique_labels": unique_labels,
        "input_dim": embeddings.shape[1],
    }
