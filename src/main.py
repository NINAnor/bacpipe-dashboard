import pathlib

import hydra
import streamlit as st

import numpy as np

from fine_tune_embeddings import get_transformed_embeddings, train_embedding_model
from utils.figures_utils import get_2d_features, get_figure
from utils.metrics import calculate_clustering_metrics
from utils.path_utils import generate_embeddings, load_embeddings_with_labels
from proto_network import train_proto_network, get_proto_transformed_embeddings


def display_embedding_info(embeddings, labels):
    """Display basic information about the embeddings"""
    st.write(f"Embedding shape: {embeddings.shape}")
    unique_labels = set(labels)
    st.write(f"Number of classes: {len(unique_labels)}")
    st.write(f"Classes: {', '.join(sorted(unique_labels))}")


def display_original_embeddings(embeddings, labels, perplexity):
    with st.status("Computing 2D embeddings with t-SNE...") as status:
        original_features_2d = get_2d_features(embeddings, perplexity)
        status.update(
            label="t-SNE computation complete", state="complete", expanded=False
        )

    st.header("Original Embeddings")

    # Calculate and display clustering metrics
    with st.status("Calculating clustering metrics..."):
        metrics = calculate_clustering_metrics(embeddings, labels)
        st.info(f"Metrics - ARI: {metrics['ari']:.4f}, AMI: {metrics['ami']:.2f}")

    # Explanation of metrics
    with st.expander("What do these metrics mean?"):
        st.markdown("""
        **ARI (Adjusted Rand Index)**: Measures the similarity between the clustering
                    from embeddings and the true labels.
        - Ranges from -0.5 to 1.0
        - 1.0 means perfect agreement between clusters and labels
        - 0 means random clustering
        - Negative values indicate worse than random clustering

        **AMI (Adjusted Mutual Information)**: Measures the information shared
                    between clusters and true labels.
        - Ranges from 0 to 1.0
        - Higher values indicate better correspondence between clusters and true classes
        """)

    fig_original = get_figure(original_features_2d, labels)
    st.plotly_chart(fig_original, use_container_width=True)

    return original_features_2d


def display_transformed_embeddings(embeddings, labels, perplexity, hidden_dim, epochs, test_size):
    with st.status(
        "Training neural network for better cluster separation..."
    ) as status:
        progress_bar = st.progress(0)

        def update_progress(epoch, total):
            progress_bar.progress(epoch / total)

        model, label_to_id = train_embedding_model(
            embeddings,
            labels,
            test_size=test_size,
            hidden_dim=hidden_dim,
            epochs=epochs,
            progress_callback=update_progress,
        )

        t_embeddings = get_transformed_embeddings(model, embeddings)
        progress_bar.empty()
        status.update(
            label=f"Neural network trained! New embedding shape: {t_embeddings.shape}",
            state="complete",
            expanded=False,
        )

    with st.status("Computing t-SNE for transformed embeddings...") as status:
        transformed_features_2d = get_2d_features(t_embeddings, perplexity)
        status.update(
            label="t-SNE computation complete", state="complete", expanded=False
        )

    st.header("Transformed Embeddings")

    with st.status("Calculating clustering metrics..."):
        metrics = calculate_clustering_metrics(t_embeddings, labels)
        st.info(f"Metrics - ARI: {metrics['ari']:.2f},AMI: {metrics['ami']:.4f}")

    fig_transformed = get_figure(transformed_features_2d, labels)
    st.plotly_chart(fig_transformed, use_container_width=True)

    return t_embeddings, transformed_features_2d

def display_proto_embeddings(embeddings, labels, perplexity, hidden_dim, epochs):
    """Process and display embeddings transformed with a prototypical network"""
    with st.status("Training prototypical network for better separation...") as status:
        progress_bar = st.progress(0)
        
        def update_progress(epoch, total):
            progress_bar.progress(epoch / total)
        
        model, label_to_id, prototypes = train_proto_network(
            embeddings, 
            labels,
            embedding_dim=hidden_dim,
            epochs=epochs,
            progress_callback=update_progress
        )
        
        proto_embeddings = get_proto_transformed_embeddings(model, embeddings)
        progress_bar.empty()
        status.update(
            label=f"Prototypical network trained! New embedding shape: {proto_embeddings.shape}",
            state="complete",
            expanded=False,
        )
    
    with st.status("Computing t-SNE for prototypical embeddings...") as status:
        transformed_features_2d = get_2d_features(proto_embeddings, perplexity)
        status.update(
            label="t-SNE computation complete", 
            state="complete", 
            expanded=False
        )
    
    st.header("Prototypical Network Embeddings")
    
    # Calculate and display clustering metrics
    with st.status("Calculating clustering metrics..."):
        metrics = calculate_clustering_metrics(proto_embeddings, labels)
        st.info(f"Embedding Quality Metrics - ARI: {metrics['ari']:.4f}, AMI: {metrics['ami']:.4f}")
    
    fig_transformed = get_figure(transformed_features_2d, labels)
    st.plotly_chart(fig_transformed, use_container_width=True)
    
    # Optionally visualize the prototypes
    if st.checkbox("Show Class Prototypes"):
        st.write("Class Prototypes Visualization")
        # Get prototype embeddings for visualization
        unique_labels = sorted(set(labels))
        label_to_id = {label: i for i, label in enumerate(unique_labels)}
        prototype_embeddings = np.array([prototypes[label_to_id[label]].detach().numpy() for label in unique_labels])
        
        # Create t-SNE for prototypes
        if prototype_embeddings.shape[0] > 1:  # Only if we have multiple prototypes
            proto_2d = get_2d_features(prototype_embeddings, perplexity)
            proto_fig = get_figure(proto_2d, unique_labels, "Class Prototypes")
            st.plotly_chart(proto_fig, use_container_width=True)
    
    return proto_embeddings, transformed_features_2d


def setup_sidebar():
    AVAILABLE_MODELS = [
        "birdnet",
        "perch_bird",
        "insect66",
        "rcl_fs_bsed",
        "aves_especies",
        "biolingual",
        "protoclr",
        "surfperch",
        "animal2vec_xc",
        "avesecho_passt",
        "birdaves_especies",
        "vggish",
        "google_whale",
        "audiomae",
        "hbdet",
        "mix2",
    ]

    st.sidebar.header("Embedding Model")
    selected_model = st.sidebar.selectbox(
        "Select Audio Embedding Model", AVAILABLE_MODELS, index=0
    )

    model_info = {
        "birdnet": "BirdNET 2.4 - Bird sound classification model made by Cornell",
        "perch_bird": "PERCH Bird - Bird sound classification model made by Naturalis",
        "vggish": "VGGish - Google's audio classification model",
        "audiomae": "AudioMAE - Self-supervised audio model",
        # TODO: add more model information
    }

    if selected_model in model_info:
        st.sidebar.info(model_info[selected_model])

    # Test size settings
    st.sidebar.header("Data parameters")
    test_size = st.sidebar.slider("Validation Split", 0.0, 1.0, 0.2, 0.05)


    # Neural network settings
    st.sidebar.header("Neural Network Settings")
    hidden_dim = st.sidebar.slider("Hidden Layer Dimension", 2, 512, 256)
    epochs = st.sidebar.slider("Training Epochs", 10, 100, 10)

    return selected_model, test_size, hidden_dim, epochs


@hydra.main(version_base=None, config_path="../", config_name="config")
def main(cfg):
    st.title("Audio Embedding Visualization Dashboard")

    # SIDEBAR CONTROLS
    selected_model, test_size, hidden_dim, epochs = setup_sidebar()

    # PATH SETTINGS
    data_dir = pathlib.Path(cfg.DATA_DIR)
    metadata_path = "/home/benjamin.cretois/data/esc50/ESC-50-master/meta/esc50.csv"

    # GENERATE THE EMBEDDINGS
    with st.status(
        f"Generating embeddings using {selected_model.upper()}..."
    ) as status:
        loader = generate_embeddings(
            selected_model, data_dir, check_if_primary_combination_exists=True
        )
        embed_dir = loader.embed_dir
        status.update(
            label=f"Embeddings generated at {embed_dir}",
            state="complete",
            expanded=False,
        )

    # LOAD THE EMBEDDINGS
    with st.status("Loading embeddings and matching labels...") as status:
        embeddings, labels, file_paths = load_embeddings_with_labels(
            embed_dir, metadata_path, model_name=selected_model
        )
        status.update(
            label=f"Loaded {len(labels)} embeddings from {selected_model}",
            state="complete",
            expanded=False,
        )

    display_embedding_info(embeddings, labels)
    display_original_embeddings(embeddings, labels, perplexity=8)
    display_transformed_embeddings(embeddings, labels, 8, hidden_dim, epochs, test_size)
    display_proto_embeddings(embeddings, labels, 8, hidden_dim, epochs)


if __name__ == "__main__":
    main()
