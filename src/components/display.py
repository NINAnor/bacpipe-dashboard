import os
import streamlit as st
import pandas as pd
import plotly.express as px
from streamlit_plotly_events import plotly_events

from fine_tune_embeddings import get_transformed_embeddings, train_embedding_model
from proto_network import get_proto_transformed_embeddings, train_proto_network
from utils.figures_utils import get_2d_features, get_figure
from utils.metrics import calculate_clustering_metrics
from components.caching import cached_tsne_features

def display_embedding_info(embeddings, labels, is_validation=False):
    """Display basic information about the embeddings"""
    dataset_type = "Validation" if is_validation else "Full"
    st.write(f"{dataset_type} Embedding shape: {embeddings.shape}")
    unique_labels = set(labels)
    st.write(f"Number of classes: {len(unique_labels)}")
    st.write(f"Classes: {', '.join(sorted(unique_labels))}")

def display_original_embeddings(embeddings, labels, file_paths, perplexity):
    if "audio_files" not in st.session_state:
        st.session_state.audio_files = file_paths
        st.session_state.current_audio = None
        st.session_state.selected_point = None
        
    with st.status("Computing 2D embeddings with t-SNE...") as status:
        original_features_2d = get_2d_features(embeddings, perplexity)
        status.update(label="t-SNE computation complete", state="complete", expanded=False)

    st.header("Original Embeddings (Validation Set)")
    
    # Calculate metrics
    metrics = calculate_clustering_metrics(embeddings, labels)
    st.info(f"Metrics - ARI: {metrics['ari']:.4f}, AMI: {metrics['ami']:.2f}")

    # Create a DataFrame with all the data
    df = pd.DataFrame({
        "x": original_features_2d[:, 0],
        "y": original_features_2d[:, 1], 
        "label": labels,
        "file_path": file_paths
    })
    
    # Create interactive plot
    fig = px.scatter(
        df, x="x", y="y", color="label", 
        hover_data=["label", "file_path"],
        title="Click on a point to play the audio"
    )
    
    # Display the plot and capture clicks
    selected_point = plotly_events(fig, click_event=True, override_height=600)
    
    # Handle audio playback when a point is clicked
    if selected_point:
        handle_audio_playback(selected_point, df)

    return embeddings, metrics

def handle_audio_playback(selected_point, df):
    """Handle audio playback when a point is clicked"""
    # Get the index of the clicked point
    x, y = selected_point[0]["x"], selected_point[0]["y"]
    closest_point = ((df["x"] - x)**2 + (df["y"] - y)**2).idxmin()
    
    # Get the file path and play the audio
    audio_file = df.loc[closest_point, "file_path"]
    
    # Update session state
    if "current_audio" not in st.session_state or st.session_state.current_audio != audio_file:
        st.session_state.current_audio = audio_file
        st.session_state.selected_point = closest_point
        
        # Create columns for audio player and info
        col1, col2 = st.columns([2, 3])
        
        with col1:
            st.audio(audio_file, format="audio/wav")
            
        with col2:
            st.markdown(f"**File:** `{os.path.basename(audio_file)}`")
            st.markdown(f"**Category:** {df.loc[closest_point, 'label']}")

def display_transformed_embeddings(
    train_embeddings,
    train_labels,
    val_embeddings,
    val_labels,
    label_to_id,
    perplexity,
    hidden_dim,
    epochs,
    model=None,  # Add option to pass pre-trained model
):
    # Don't create another status if model is already provided
    if model is None:
        with st.status("Training neural network for better cluster separation...") as status:
            model = train_embedding_model(
                train_embeddings,
                train_labels,
                val_embeddings,
                val_labels,
                hidden_dim=hidden_dim,
                epochs=epochs,
            )
            status.update(label="Neural network trained!", state="complete", expanded=False)
    
    # Transform embeddings
    t_embeddings = get_transformed_embeddings(model, val_embeddings)
    
    st.write(f"Validation embedding shape: {t_embeddings.shape}")
    
    # Compute t-SNE separately without nesting status containers
    transformed_features_2d = get_2d_features(t_embeddings, perplexity)
    
    st.header("Transformed Embeddings (Validation Set)")

    id_to_label = {v: k for k, v in label_to_id.items()}
    plot_labels = [id_to_label[val.item()] for val in val_labels]

    # Calculate metrics without nested status
    metrics = calculate_clustering_metrics(t_embeddings, plot_labels)
    st.info(f"Metrics - ARI: {metrics['ari']:.2f}, AMI: {metrics['ami']:.4f}")

    fig_transformed = get_figure(transformed_features_2d, plot_labels)
    st.plotly_chart(fig_transformed, use_container_width=True)

    return t_embeddings, metrics


def display_proto_embeddings(
    train_embeddings,
    train_labels,
    val_embeddings,
    val_labels,
    label_to_id,
    perplexity,
    hidden_dim,
    epochs,
    model=None,  # Add option to pass pre-trained model
    prototypes=None  # Add option to pass pre-computed prototypes
):
    # Don't create another status if model is already provided
    if model is None:
        with st.status("Training prototypical network for better separation...") as status:
            model, _, prototypes = train_proto_network(
                train_embeddings,
                train_labels,
                val_embeddings,
                val_labels,
                embedding_dim=hidden_dim,
                epochs=epochs,
            )
            status.update(label="Prototypical network trained!", state="complete", expanded=False)
    
    # Transform embeddings
    proto_embeddings = get_proto_transformed_embeddings(model, val_embeddings)
    
    # Display shape info
    st.write(f"Validation embedding shape: {proto_embeddings.shape}")
    
    # Compute t-SNE (without nesting status)
    transformed_features_2d = get_2d_features(proto_embeddings, perplexity)
    
    st.header("Prototypical Network Embeddings (Validation Set)")

    id_to_label = {v: k for k, v in label_to_id.items()}
    plot_labels = [id_to_label[val.item()] for val in val_labels]

    # Calculate metrics (without nesting status)
    metrics = calculate_clustering_metrics(proto_embeddings, plot_labels)
    st.info(f"Embedding Metrics - ARI: {metrics['ari']:.4f}, AMI: {metrics['ami']:.4f}")

    # Create and display the figure
    fig_transformed = get_figure(transformed_features_2d, plot_labels)
    st.plotly_chart(fig_transformed, use_container_width=True)

    return proto_embeddings, metrics
