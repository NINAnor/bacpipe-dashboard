import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score,
    adjusted_mutual_info_score,
    adjusted_rand_score,
    confusion_matrix,
    f1_score,
)


def calculate_clustering_metrics(embeddings, labels):
    unique_labels = sorted(set(labels))
    label_to_id = {label: i for i, label in enumerate(unique_labels)}
    numeric_labels = [label_to_id[label] for label in labels]

    n_clusters = len(unique_labels)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)

    ari = adjusted_rand_score(numeric_labels, cluster_labels)
    ami = adjusted_mutual_info_score(numeric_labels, cluster_labels)

    return {"ari": ari, "ami": ami}


def calculate_classification_metrics(embeddings, true_labels):
    """
    This function tells how clusterable embeddings are — i.e.,
    how well they preserve class information without supervision.
    """
    if isinstance(true_labels[0], str):
        label_to_id = {label: i for i, label in enumerate(sorted(set(true_labels)))}
        true_numeric = np.array([label_to_id[label] for label in true_labels])
        class_names = sorted(set(true_labels))
    elif torch.is_tensor(true_labels):
        true_numeric = true_labels.detach().cpu().numpy()
        unique_vals = sorted(np.unique(true_numeric).tolist())
        class_names = [str(i) for i in unique_vals]
    else:
        true_numeric = np.array(true_labels)
        unique_vals = sorted(np.unique(true_numeric).tolist())
        class_names = [str(i) for i in unique_vals]

    # K-means cluster
    n_clusters = len(set(true_numeric))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)

    # Create confusion matrix - don't specify labels parameter here
    cm = confusion_matrix(true_numeric, cluster_labels)

    # Use Hungarian algorithm to find optimal mapping
    row_ind, col_ind = linear_sum_assignment(-cm)

    # Remap cluster labels to match true labels
    remapped_labels = np.zeros_like(cluster_labels)
    for i, j in zip(col_ind, row_ind, strict=False):
        remapped_labels[cluster_labels == i] = j

    # Calculate metrics
    accuracy = accuracy_score(true_numeric, remapped_labels)
    f1 = f1_score(true_numeric, remapped_labels, average="weighted")

    # Get all unique labels actually present in the data
    all_labels = sorted(np.union1d(np.unique(true_numeric), np.unique(remapped_labels)))

    # Create remapped confusion matrix with explicitly calculated labels
    cm_remapped = confusion_matrix(true_numeric, remapped_labels, labels=all_labels)

    return {
        "accuracy": accuracy,
        "f1": f1,
        "cm": cm_remapped,  # Use the key 'cm' to match what's in summary_dashboard
        "class_names": class_names,
    }
