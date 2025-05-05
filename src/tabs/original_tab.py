import streamlit as st
import pandas as pd
import os
from streamlit_plotly_events import plotly_events
import plotly.express as px

from components.display import display_embedding_info
from components.caching import cached_tsne_features
from utils.metrics import calculate_clustering_metrics
from utils.figures_utils import get_2d_features, get_figure

def render(model_changed, test_size_changed):
    """Render the Original Embeddings tab"""
    if 'train_val_split' not in st.session_state:
        st.warning("Please load data first by clicking the 'Data Loading' tab")
        return
    
    # Access data from session state
    split_data = st.session_state.train_val_split
    X_val, y_val = split_data['X_val'], split_data['y_val']
    val_files = split_data['val_files']
    
    # Only recompute if model or test_size changed
    if 'original_results' not in st.session_state or model_changed or test_size_changed:
        print(y_val)
        
        # Calculate metrics
        with st.status("Computing t-SNE and metrics...") as status:
            original_features_2d = get_2d_features(X_val, perplexity=8)
            metrics = calculate_clustering_metrics(X_val, y_val)
            status.update(label="t-SNE computation complete", state="complete", expanded=False)
        
        st.header("Original Embeddings (Validation Set)")
        st.info(f"Metrics - ARI: {metrics['ari']:.4f}, AMI: {metrics['ami']:.2f}")
        
        # Create interactive plot
        create_interactive_plot(original_features_2d, y_val, val_files)
        
        # Store for reuse
        st.session_state.original_results = {
            'embeddings': X_val,
            'metrics': metrics
        }
    else:
        st.info("Using cached original embeddings visualization")
        
        # Get data from session state
        metrics = st.session_state.original_results['metrics']
        
        # Redisplay metrics and visualization
        st.header("Original Embeddings (Validation Set)")
        st.info(f"Metrics - ARI: {metrics['ari']:.4f}, AMI: {metrics['ami']:.2f}")
        
        # Recalculate visualization (quick operation)
        original_features_2d = cached_tsne_features(X_val, 8)
        
        # Create interactive plot
        create_interactive_plot(original_features_2d, y_val, val_files)

def create_interactive_plot(features_2d, labels, file_paths):
    """Create an interactive plot with audio playback functionality"""
    # Create a DataFrame for the plot
    df = pd.DataFrame({
        "x": features_2d[:, 0],
        "y": features_2d[:, 1], 
        "label": labels,
        "file_path": file_paths
    })
    
    # Interactive plot
    fig = px.scatter(
        df, x="x", y="y", color="label", 
        hover_data=["label", "file_path"],
        title="Click on a point to play the audio"
    )
    
    # Display plot and handle clicks
    selected_point = plotly_events(fig, click_event=True, override_height=600)
    
    # Audio playback
    if selected_point:
        x, y = selected_point[0]["x"], selected_point[0]["y"]
        closest_point = ((df["x"] - x)**2 + (df["y"] - y)**2).idxmin()
        audio_file = df.loc[closest_point, "file_path"]
        
        # Create columns for audio player and info
        col1, col2 = st.columns([2, 3])
        with col1:
            st.audio(audio_file, format="audio/wav")
        with col2:
            st.markdown(f"**File:** `{os.path.basename(audio_file)}`")
            st.markdown(f"**Category:** {df.loc[closest_point, 'label']}")