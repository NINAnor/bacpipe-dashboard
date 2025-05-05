import streamlit as st
import pandas as pd
import plotly.express as px

from components.caching import cached_train_proto_network, cached_tsne_features
from proto_network import get_proto_transformed_embeddings
from utils.metrics import calculate_clustering_metrics
from utils.figures_utils import get_figure

def render(model_changed, test_size_changed, hidden_dim_changed, epochs_changed):
    """Render the Prototypical Network tab"""
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
    
    if 'proto_results' not in st.session_state or relevant_changed:
        # First train model
        with st.status("Training prototypical network..."):
            model, _, prototypes = cached_train_proto_network(
                train_data["X"], train_data["y"], 
                val_data["X"], val_data["y"], 
                hidden_dim, epochs
            )
        
        # Transform embeddings
        proto_embeddings = get_proto_transformed_embeddings(model, val_data["X"])
        
        # Display shape info
        st.write(f"Validation embedding shape: {proto_embeddings.shape}")
        
        # Display transformed embeddings
        st.header("Prototypical Network Embeddings (Validation Set)")

        id_to_label = {v: k for k, v in val_data["label_to_id"].items()}
        plot_labels = [id_to_label[val.item()] for val in val_data["y"]]

        # Calculate metrics
        metrics = calculate_clustering_metrics(proto_embeddings, plot_labels)
        st.info(f"Embedding Metrics - ARI: {metrics['ari']:.4f}, AMI: {metrics['ami']:.4f}")
        
        # Create visualization
        transformed_features_2d = cached_tsne_features(proto_embeddings, 8)
        fig_transformed = get_figure(transformed_features_2d, plot_labels)
        # Add unique key to plotly chart
        st.plotly_chart(fig_transformed, use_container_width=True, key="proto_new_plot")
        
        # Store results in session state
        st.session_state.proto_results = {
            'embeddings': proto_embeddings,
            'metrics': metrics,
            'model': model,
            'prototypes': prototypes
        }
    else:
        # Use cached results
        st.info("Using cached prototypical network embeddings")
        
        # Get data from session state
        proto_embeddings = st.session_state.proto_results['embeddings']
        metrics = st.session_state.proto_results['metrics']
        
        # Display header and metrics
        st.write(f"Validation embedding shape: {proto_embeddings.shape}")
        st.header("Prototypical Network Embeddings (Validation Set)")
        st.info(f"Embedding Metrics - ARI: {metrics['ari']:.4f}, AMI: {metrics['ami']:.4f}")
        
        # Get labels for plotting
        id_to_label = {v: k for k, v in val_data["label_to_id"].items()}
        plot_labels = [id_to_label[val.item()] for val in val_data["y"]]
        
        # Recalculate visualization
        transformed_features_2d = cached_tsne_features(proto_embeddings, 8)
        fig_transformed = get_figure(transformed_features_2d, plot_labels)
        # Add unique key to plotly chart - different from the one above
        st.plotly_chart(fig_transformed, use_container_width=True, key="proto_cached_plot")