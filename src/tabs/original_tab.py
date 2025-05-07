import streamlit as st

from components.caching import cached_tsne_features
from components.display import render_interactive_scatterplot
from utils.figures_utils import get_2d_features
from utils.metrics import calculate_clustering_metrics


def render(model_changed, test_size_changed):
    if "train_val_split" not in st.session_state:
        st.warning("Please load data first by clicking the 'Data Loading' tab")
        return

    # Access data from session state
    split_data = st.session_state.train_val_split
    X_val, y_val = split_data["X_val"], split_data["y_val"]
    val_files = split_data["val_files"]

    # Only recompute if model or test_size changed
    if "original_results" not in st.session_state or model_changed or test_size_changed:
        # Calculate metrics
        with st.status("Computing t-SNE and metrics...") as status:
            original_features_2d = get_2d_features(X_val, perplexity=8)
            metrics = calculate_clustering_metrics(X_val, y_val)
            status.update(
                label="t-SNE computation complete", state="complete", expanded=False
            )

        st.header("Original Embeddings (Validation Set)")
        st.info(f"Metrics - ARI: {metrics['ari']:.4f}, AMI: {metrics['ami']:.2f}")

        # Create interactive plot
        render_interactive_scatterplot(
            points=original_features_2d,
            labels=y_val,
            audio_paths=val_files,
            title="Original Embeddings (Validation Set)",
        )

        # Store for reuse
        st.session_state.original_results = {"embeddings": X_val, "metrics": metrics}
    else:
        # Get data from session state
        metrics = st.session_state.original_results["metrics"]

        # Redisplay metrics and visualization
        st.header("Original Embeddings (Validation Set)")
        st.info(f"Metrics - ARI: {metrics['ari']:.4f}, AMI: {metrics['ami']:.2f}")

        # Recalculate visualization (quick operation)
        original_features_2d = cached_tsne_features(X_val, 8)

        # Create interactive plot
        render_interactive_scatterplot(
            points=original_features_2d,
            labels=y_val,
            audio_paths=val_files,
            title="Original Embeddings (Validation Set)",
        )
