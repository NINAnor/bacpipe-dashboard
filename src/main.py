import os
import sys
import streamlit as st
import pathlib
import hydra
import pandas as pd
import numpy as np

from utils.path_utils import generate_embeddings, load_embeddings_with_labels
from utils.figures_utils import get_2d_features, get_figure
from fine_tune_embeddings import train_embedding_model, get_transformed_embeddings
from utils.metrics import calculate_clustering_metrics


def display_embedding_info(embeddings, labels):
    """Display basic information about the embeddings"""
    st.write(f"Embedding shape: {embeddings.shape}")
    unique_labels = set(labels)
    st.write(f"Number of classes: {len(unique_labels)}")
    st.write(f"Classes: {', '.join(sorted(unique_labels))}")


def display_original_embeddings(embeddings, labels, perplexity):
    """Process and display the original embeddings with quality metrics"""
    with st.status("Computing 2D embeddings with t-SNE...") as status:
        original_features_2d = get_2d_features(embeddings, perplexity)
        status.update(label="t-SNE computation complete", state="complete", expanded=False)
    
    st.header("Original Embeddings")
    
    # Calculate and display clustering metrics
    with st.status("Calculating clustering metrics..."):
        metrics = calculate_clustering_metrics(embeddings, labels)
        st.info(f"Embedding Quality Metrics - ARI: {metrics['ari']:.4f}, AMI: {metrics['ami']:.4f}")
    
    # Explanation of metrics
    with st.expander("What do these metrics mean?"):
        st.markdown("""
        **ARI (Adjusted Rand Index)**: Measures the similarity between the clustering from embeddings and the true labels.
        - Ranges from -0.5 to 1.0
        - 1.0 means perfect agreement between clusters and labels
        - 0 means random clustering
        - Negative values indicate worse than random clustering
        
        **AMI (Adjusted Mutual Information)**: Measures the information shared between clusters and true labels.
        - Ranges from 0 to 1.0
        - Higher values indicate better correspondence between clusters and true classes
        """)
    
    fig_original = get_figure(original_features_2d, labels)
    st.plotly_chart(fig_original, use_container_width=True)
    
    return original_features_2d


def display_transformed_embeddings(embeddings, labels, perplexity, hidden_dim, epochs):

    with st.status("Training neural network for better cluster separation...") as status:
        progress_bar = st.progress(0)
        
        def update_progress(epoch, total):
            progress_bar.progress(epoch / total)
        
        model, label_to_id = train_embedding_model(
            embeddings, labels,
            hidden_dim=hidden_dim,
            epochs=epochs,
            progress_callback=update_progress
        )
        
        transformed_embeddings = get_transformed_embeddings(model, embeddings)
        progress_bar.empty() 
        status.update(label=f"Neural network trained! New embedding shape: {transformed_embeddings.shape}", 
                     state="complete", expanded=False)
    
    with st.status("Computing t-SNE for transformed embeddings...") as status:
        transformed_features_2d = get_2d_features(transformed_embeddings, perplexity)
        status.update(label="t-SNE computation complete", state="complete", expanded=False)
    
    st.header("Transformed Embeddings")
    
    with st.status("Calculating clustering metrics..."):
        metrics = calculate_clustering_metrics(transformed_embeddings, labels)
        st.info(f"Embedding Quality Metrics - ARI: {metrics['ari']:.4f}, AMI: {metrics['ami']:.4f}")
    
    fig_transformed = get_figure(transformed_features_2d, labels)
    st.plotly_chart(fig_transformed, use_container_width=True)
    
    return transformed_embeddings, transformed_features_2d


def setup_sidebar():

    AVAILABLE_MODELS = [
        "birdnet", "perch_bird", "insect66", "rcl_fs_bsed", 
        "aves_especies", "biolingual", "protoclr", "surfperch",
        "animal2vec_xc", "avesecho_passt", "birdaves_especies", 
        "vggish", "google_whale", "audiomae", "hbdet", "mix2"
    ]
    
    st.sidebar.header("Embedding Model")
    selected_model = st.sidebar.selectbox(
        "Select Audio Embedding Model", 
        AVAILABLE_MODELS, 
        index=0
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
    
    # TSNE settings - not sure if really useful
    st.sidebar.header("Visualization Settings")
    perplexity = st.sidebar.slider("TSNE Perplexity", 5, 50, 8)
    
    # Neural network settings
    st.sidebar.header("Neural Network Settings")
    hidden_dim = st.sidebar.slider("Hidden Layer Dimension", 2, 512, 256)
    epochs = st.sidebar.slider("Training Epochs", 10, 100, 10)
    
    return selected_model, perplexity, hidden_dim, epochs


@hydra.main(version_base=None, config_path="../", config_name="config")
def main(cfg):
    st.title("Audio Embedding Visualization Dashboard")
    
    # SIDEBAR CONTROLS
    selected_model, perplexity, hidden_dim, epochs = setup_sidebar()
    
    # PATH SETTINGS
    data_dir = pathlib.Path(cfg.DATA_DIR)
    metadata_path = "/home/benjamin.cretois/data/esc50/ESC-50-master/meta/esc50.csv"

    # GENERATE THE EMBEDDINGS
    with st.status(f"Generating embeddings using {selected_model.upper()}...") as status:
        loader = generate_embeddings(selected_model, data_dir, check_if_primary_combination_exists=True)
        embed_dir = loader.embed_dir
        status.update(label=f"Embeddings generated at {embed_dir}", state="complete", expanded=False)
    
    # LOAD THE EMBEDDINGS
    with st.status("Loading embeddings and matching labels...") as status:
        embeddings, labels, file_paths = load_embeddings_with_labels(embed_dir, metadata_path, model_name=selected_model)
        status.update(label=f"Loaded {len(labels)} embeddings from {selected_model}", 
                     state="complete", expanded=False)
    
    display_embedding_info(embeddings, labels)
    display_original_embeddings(embeddings, labels, perplexity)
    display_transformed_embeddings(embeddings, labels, perplexity, hidden_dim, epochs)


if __name__ == "__main__":
    main()