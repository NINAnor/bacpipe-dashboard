import streamlit as st
import pandas as pd
import plotly.express as px

from components.caching import cached_train_embedding_model, cached_tsne_features
from fine_tune_embeddings import get_transformed_embeddings
from utils.metrics import calculate_clustering_metrics
from utils.figures_utils import get_figure

def render(model_changed, test_size_changed, hidden_dim_changed, epochs_changed):
    """Render the Fine-tuned Embeddings tab"""
    if 'train_val_split' not in st.session_state:
        st.warning("Please load data first by clicking the 'Data Loading' tab")
        return
    
    # Access data from session state
    split_data = st.session_state.train_val_split
    train_data = split_data['train_data']
    val_data = split_data['val_data']
    
    # Determine if recomputation is needed
    relevant_changed = model_changed or test_size_changed or hidden_dim_changed or epochs_changed
    hidden_dim = st.session_state.last_params['hidden_dim']
    epochs = st.session_state.last_params['epochs']
    
    if 'finetune_results' not in st.session_state or relevant_changed:
        # First train model
        with st.status("Training neural network..."):
            model = cached_train_embedding_model(
                train_data["X"], train_data["y"], 
                val_data["X"], val_data["y"], 
                hidden_dim, epochs
            )
        
        # Transform embeddings
        t_embeddings = get_transformed_embeddings(model, val_data["X"])
        
        st.write(f"Validation embedding shape: {t_embeddings.shape}")
        
        # Display transformed embeddings
        st.header("Transformed Embeddings (Validation Set)")

        id_to_label = {v: k for k, v in val_data["label_to_id"].items()}
        plot_labels = [id_to_label[val.item()] for val in val_data["y"]]

        # Calculate metrics
        metrics = calculate_clustering_metrics(t_embeddings, plot_labels)
        st.info(f"Metrics - ARI: {metrics['ari']:.2f}, AMI: {metrics['ami']:.4f}")
        
        # Create visualization
        transformed_features_2d = cached_tsne_features(t_embeddings, 8)
        fig_transformed = get_figure(transformed_features_2d, plot_labels)
        st.plotly_chart(fig_transformed, use_container_width=True)
        
        # Store results in session state
        st.session_state.finetune_results = {
            'embeddings': t_embeddings,
            'metrics': metrics,
            'model': model
        }
    else:
        # Use cached results
        st.info("Using cached fine-tuned embeddings")
        
        # Get data from session state
        t_embeddings = st.session_state.finetune_results['embeddings']
        metrics = st.session_state.finetune_results['metrics']
        
        # Display header and metrics
        st.write(f"Validation embedding shape: {t_embeddings.shape}")
        st.header("Transformed Embeddings (Validation Set)")
        st.info(f"Metrics - ARI: {metrics['ari']:.2f}, AMI: {metrics['ami']:.4f}")
        
        # Get labels for plotting
        id_to_label = {v: k for k, v in val_data["label_to_id"].items()}
        plot_labels = [id_to_label[val.item()] for val in val_data["y"]]
        
        # Recalculate visualization
        transformed_features_2d = cached_tsne_features(t_embeddings, 8)
        fig_transformed = get_figure(transformed_features_2d, plot_labels)
        st.plotly_chart(fig_transformed, use_container_width=True)