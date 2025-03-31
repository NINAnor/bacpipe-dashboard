from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score


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
