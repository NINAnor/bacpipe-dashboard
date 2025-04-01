from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
from sklearn.metrics import accuracy_score, f1_score

import torch
import numpy as np


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


# This function tells how clusterable embeddings are — i.e., how well they preserve class information without supervision.

def calculate_classification_metrics(embeddings, true_labels):

    if isinstance(true_labels[0], str):
        label_to_id = {label: i for i, label in enumerate(sorted(set(true_labels)))}
        true_numeric = np.array([label_to_id[label] for label in true_labels])
    elif torch.is_tensor(true_labels):
        true_numeric = true_labels.detach().cpu().numpy()
    else:
        true_numeric = np.array(true_labels)
    
    # Create K-means cluster with same number of clusters as classes
    n_clusters = len(set(true_numeric))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    # We need to map cluster IDs to class IDs for meaningful evaluation
    # Find the best mapping between cluster labels and true labels
    from sklearn.metrics import confusion_matrix
    from scipy.optimize import linear_sum_assignment
    
    # Create confusion matrix
    cm = confusion_matrix(true_numeric, cluster_labels, labels=range(n_clusters))
    
    # Use Hungarian algorithm to find optimal mapping
    row_ind, col_ind = linear_sum_assignment(-cm)
    
    # Remap cluster labels to match true labels
    remapped_labels = np.zeros_like(cluster_labels)
    for i, j in zip(col_ind, row_ind):
        remapped_labels[cluster_labels == i] = j
    
    # Calculate metrics
    accuracy = accuracy_score(true_numeric, remapped_labels)
    f1 = f1_score(true_numeric, remapped_labels, average='weighted')
    
    return {"accuracy": accuracy, "f1": f1}
