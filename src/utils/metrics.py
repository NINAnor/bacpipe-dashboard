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

    We create K-means clusters with the same number of clusters as classes
    so we can construct a confusion matrix and use the Hungarian algorithm to find
    the best mapping between cluster labels and true labels.
    We then calculate accuracy and F1 score based on this mapping.
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

    # Create confusion matrix
    cm = confusion_matrix(true_numeric, cluster_labels, labels=range(n_clusters))

    # Use Hungarian algorithm to find optimal mapping
    row_ind, col_ind = linear_sum_assignment(-cm)

    # Remap cluster labels to match true labels
    remapped_labels = np.zeros_like(cluster_labels)
    for i, j in zip(col_ind, row_ind, strict=False):
        remapped_labels[cluster_labels == i] = j

    # re-mapped confusion matrix
    cm_remapped = confusion_matrix(
        true_labels, remapped_labels, labels=range(n_clusters)
    )

    # Calculate metrics
    accuracy = accuracy_score(true_numeric, remapped_labels)
    f1 = f1_score(true_numeric, remapped_labels, average="weighted")

    return {
        "accuracy": accuracy,
        "f1": f1,
        "cm": cm_remapped,
        "class_names": class_names,
    }
