import streamlit as st

from components.caching import cached_train_embedding_model, cached_tsne_features
from components.display import render_interactive_scatterplot
from fine_tune_embeddings import get_transformed_embeddings
from utils.metrics import calculate_clustering_metrics


def render(model_changed, test_size_changed, hidden_dim_changed, epochs_changed):
    if "train_val_split" not in st.session_state:
        st.warning("Please load data first by clicking the 'Data Loading' tab")
        return

    # Access data from session state
    split_data = st.session_state.train_val_split
    train_data = split_data["train_data"]
    val_data = split_data["val_data"]
    val_files = split_data["val_files"]

    # Determine if recomputation is needed
    relevant_changed = (
        model_changed or test_size_changed or hidden_dim_changed or epochs_changed
    )
    hidden_dim = st.session_state.last_params["hidden_dim"]
    epochs = st.session_state.last_params["epochs"]

    if "finetune_results" not in st.session_state or relevant_changed:
        # First train model
        with st.status("Training neural network..."):
            model = cached_train_embedding_model(
                train_data["X"],
                train_data["y"],
                val_data["X"],
                val_data["y"],
                hidden_dim,
                epochs,
            )

        # Transform embeddings
        t_embeddings = get_transformed_embeddings(model, val_data["X"])
        id_to_label = {v: k for k, v in val_data["label_to_id"].items()}
        plot_labels = [id_to_label[val.item()] for val in val_data["y"]]

        # Calculate metrics
        metrics = calculate_clustering_metrics(t_embeddings, plot_labels)
        st.info(f"Metrics - ARI: {metrics['ari']:.2f}, AMI: {metrics['ami']:.4f}")

        # Create visualization with t-SNE
        transformed_features_2d = cached_tsne_features(t_embeddings, 8)

        # Store results in session state
        st.session_state.finetune_results = {
            "embeddings": t_embeddings,
            "metrics": metrics,
            "model": model,
            "features_2d": transformed_features_2d,
            "plot_labels": plot_labels,
        }
        render_interactive_scatterplot(
            points=transformed_features_2d,
            labels=plot_labels,
            audio_paths=val_files,
            height=600,
            title="Fine-tuned Embeddings Visualization",
        )

    else:
        # Get data from session state
        t_embeddings = st.session_state.finetune_results["embeddings"]
        metrics = st.session_state.finetune_results["metrics"]
        transformed_features_2d = st.session_state.finetune_results["features_2d"]
        plot_labels = st.session_state.finetune_results["plot_labels"]

        st.header("Transformed Embeddings (Validation Set)")
        st.info(f"Metrics - ARI: {metrics['ari']:.2f}, AMI: {metrics['ami']:.4f}")

        render_interactive_scatterplot(
            points=transformed_features_2d,
            labels=plot_labels,
            audio_paths=val_files,
            height=600,
            title="Fine-tuned Embeddings Visualization",
        )
