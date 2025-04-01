import pathlib

import hydra
from hydra.core.global_hydra import GlobalHydra
import streamlit as st
import numpy as np
import pandas as pd

import plotly.express as px

from fine_tune_embeddings import get_transformed_embeddings, train_embedding_model
from utils.figures_utils import get_2d_features, get_figure
from utils.metrics import calculate_clustering_metrics, calculate_classification_metrics
from utils.path_utils import generate_embeddings, load_embeddings_with_labels
from proto_network import train_proto_network, get_proto_transformed_embeddings
from utils.data_utils import split_data, prepare_embedding_data




def display_embedding_info(embeddings, labels, is_validation=False):
    """Display basic information about the embeddings"""
    dataset_type = "Validation" if is_validation else "Full"
    st.write(f"{dataset_type} Embedding shape: {embeddings.shape}")
    unique_labels = set(labels)
    st.write(f"Number of classes: {len(unique_labels)}")
    st.write(f"Classes: {', '.join(sorted(unique_labels))}")

def display_original_embeddings(embeddings, labels, perplexity):
    with st.status("Computing 2D embeddings with t-SNE...") as status:
        original_features_2d = get_2d_features(embeddings, perplexity)
        status.update(
            label="t-SNE computation complete", state="complete", expanded=False
        )

    st.header("Original Embeddings (Validation Set)")

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

    return original_features_2d, metrics

def display_transformed_embeddings(train_embeddings, train_labels, val_embeddings, val_labels, label_to_id, perplexity, hidden_dim, epochs):
    with st.status(
        "Training neural network for better cluster separation..."
    ) as status:

        model = train_embedding_model(
            train_embeddings,
            train_labels,
            val_embeddings,
            val_labels,
            hidden_dim=hidden_dim,
            epochs=epochs,
        )

        # Transform only validation embeddings
        t_embeddings = get_transformed_embeddings(model, val_embeddings)

        status.update(
            label=f"Neural network trained! Validation embedding shape: {t_embeddings.shape}",
            state="complete",
            expanded=False,
        )

    with st.status("Computing t-SNE for transformed validation embeddings...") as status:
        transformed_features_2d = get_2d_features(t_embeddings, perplexity)
        status.update(
            label="t-SNE computation complete", state="complete", expanded=False
        )

    st.header("Transformed Embeddings (Validation Set)")

    id_to_label = {v: k for k, v in label_to_id.items()}
    plot_labels = [id_to_label[val.item()] for val in val_labels]

    with st.status("Calculating clustering metrics..."):
        metrics = calculate_clustering_metrics(t_embeddings, plot_labels)
        st.info(f"Metrics - ARI: {metrics['ari']:.2f}, AMI: {metrics['ami']:.4f}")



    fig_transformed = get_figure(transformed_features_2d, plot_labels)
    st.plotly_chart(fig_transformed, use_container_width=True)

    return t_embeddings, metrics

def display_proto_embeddings(train_embeddings, train_labels, val_embeddings, val_labels, label_to_id, perplexity, hidden_dim, epochs):
    """Process and display embeddings transformed with a prototypical network (validation set only)"""
    with st.status("Training prototypical network for better separation...") as status:

        # Train model with prepared tensors
        model, _, prototypes = train_proto_network(
            train_embeddings, 
            train_labels,
            val_embeddings,
            val_labels,
            embedding_dim=hidden_dim,
            epochs=epochs,
        )
        
        # Transform only validation embeddings
        proto_embeddings = get_proto_transformed_embeddings(model, val_embeddings)

        status.update(
            label=f"Prototypical network trained! Validation embedding shape: {proto_embeddings.shape}",
            state="complete",
            expanded=False,
        )
    
    with st.status("Computing t-SNE for prototypical validation embeddings...") as status:
        transformed_features_2d = get_2d_features(proto_embeddings, perplexity)
        status.update(
            label="t-SNE computation complete", 
            state="complete", 
            expanded=False
        )
    
    st.header("Prototypical Network Embeddings (Validation Set)")
    

    id_to_label = {v: k for k, v in label_to_id.items()}
    plot_labels = [id_to_label[val.item()] for val in val_labels]

    with st.status("Calculating clustering metrics..."):
        metrics = calculate_clustering_metrics(proto_embeddings, plot_labels)
        st.info(f"Embedding Quality Metrics - ARI: {metrics['ari']:.4f}, AMI: {metrics['ami']:.4f}")
    

    fig_transformed = get_figure(transformed_features_2d, plot_labels)
    st.plotly_chart(fig_transformed, use_container_width=True)
    
    return proto_embeddings, metrics

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
    }

    if selected_model in model_info:
        st.sidebar.info(model_info[selected_model])

    # Test size settings
    st.sidebar.header("Data parameters")
    test_size = st.sidebar.slider("Validation Split", 0.1, 0.5, 0.2, 0.05)

    # Neural network settings
    st.sidebar.header("Neural Network Settings")
    hidden_dim = st.sidebar.slider("Hidden Layer Dimension", 2, 512, 256)
    epochs = st.sidebar.slider("Training Epochs", 10, 100, 10)

    return selected_model, test_size, hidden_dim, epochs


def summary_dashboard(transf_embeddings, transf_metrics, proto_embeddings, proto_metrics, original_embeddings, original_metrics, y_val):

    original_class_metrics = calculate_classification_metrics(original_embeddings, y_val)
    transf_class_metrics = calculate_classification_metrics(transf_embeddings, y_val)
    proto_class_metrics = calculate_classification_metrics(proto_embeddings, y_val)
    
    # Combine metrics
    original_metrics.update(original_class_metrics)
    transf_metrics.update(transf_class_metrics)
    proto_metrics.update(proto_class_metrics)
    
    # Display metrics comparison
    st.header("Embedding Methods Comparison")
    
    metrics_df = pd.DataFrame({
        'Metric': ['ARI', 'AMI', 'Accuracy', 'F1 Score'],
        'Original Embeddings': [
            f"{original_metrics['ari']:.4f}", 
            f"{original_metrics['ami']:.4f}",
            f"{original_metrics['accuracy']:.4f}",
            f"{original_metrics['f1']:.4f}"
        ],
        'Fine-tuned Embeddings': [
            f"{transf_metrics['ari']:.4f}", 
            f"{transf_metrics['ami']:.4f}",
            f"{transf_metrics['accuracy']:.4f}",
            f"{transf_metrics['f1']:.4f}"
        ],
        'Prototypical Networks': [
            f"{proto_metrics['ari']:.4f}", 
            f"{proto_metrics['ami']:.4f}",
            f"{proto_metrics['accuracy']:.4f}",
            f"{proto_metrics['f1']:.4f}"
        ]
    })
    
    st.table(metrics_df)
    
    # Create a bar chart comparing the methods
    st.subheader("Visual Comparison")
    
    chart_data = pd.DataFrame({
        'Metric': ['ARI', 'AMI', 'Accuracy', 'F1 Score'] * 3,
        'Value': [
            original_metrics['ari'], original_metrics['ami'], 
            original_metrics['accuracy'], original_metrics['f1'],
            transf_metrics['ari'], transf_metrics['ami'], 
            transf_metrics['accuracy'], transf_metrics['f1'],
            proto_metrics['ari'], proto_metrics['ami'], 
            proto_metrics['accuracy'], proto_metrics['f1']
        ],
        'Method': ['Original'] * 4 + ['Fine-tuning'] * 4 + ['Prototypical'] * 4
    })
    
    fig = px.bar(
        chart_data,
        x='Metric',
        y='Value',
        color='Method',
        barmode='group',
        title='Comparison of Embedding Methods',
        height=500,
        color_discrete_map={
            'Original': '#636EFA',     # Blue
            'Fine-tuning': '#EF553B',  # Red/orange
            'Prototypical': '#00CC96'  # Green
        }
    )
    
    fig.update_layout(
        yaxis_title='Score',
        legend=dict(
            orientation='h', 
            yanchor='bottom', 
            y=1.02,        
            xanchor='right', 
            x=1
        )
    )

    st.plotly_chart(fig, use_container_width=True)


def run_dashboard(cfg):
    st.title("Audio Embedding Visualization Dashboard")

    # SIDEBAR CONTROLS
    selected_model, test_size, hidden_dim, epochs = setup_sidebar()

    # PATH SETTINGS
    data_dir = cfg["DATA_DIR"]
    metadata_path = cfg["METADATA_PATH"]

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

    # SPLIT THE DATASET
    with st.status(f"Splitting data into train ({1-test_size:.0%}) and validation ({test_size:.0%}) sets..."):
        X_train, X_val, y_train, y_val = split_data(embeddings, labels, test_size)
        st.success(f"Train set: {len(y_train)} samples, Validation set: {len(y_val)} samples")
    
    train_data = prepare_embedding_data(X_train, y_train)
    val_data = prepare_embedding_data(X_val, y_val)

    # Display info about the validation set
    display_embedding_info(X_val, y_val, is_validation=True)
    original_embeddings, original_metrics = display_original_embeddings(X_val, y_val, perplexity=8)
    
    # Train the model using the "vanilla" and "prototypical" pipeline
    transf_embeddings, transf_metrics = display_transformed_embeddings(train_data["X"], train_data["y"], val_data["X"], val_data["y"], val_data["label_to_id"], 8, hidden_dim, epochs)
    proto_embeddings, proto_metrics = display_proto_embeddings(train_data["X"], train_data["y"], val_data["X"], val_data["y"], val_data["label_to_id"], 8, hidden_dim, epochs)

    summary_dashboard(transf_embeddings, transf_metrics, proto_embeddings, proto_metrics, original_embeddings, original_metrics, val_data["y"])

def reset_hydra_config():
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

# Use this decorator pattern to safely work with Hydra
@hydra.main(version_base=None, config_path="../", config_name="config")
def _main(cfg):
    # Store config in session state for persistence
    if 'config' not in st.session_state:
        st.session_state['config'] = {
            'DATA_DIR': cfg.get('DATA_DIR', ''),
            'METADATA_PATH': cfg.get('METADATA_PATH', '')
        }
    
    # Call your actual main function with the config
    run_dashboard(st.session_state['config'])

if __name__ == "__main__":
    reset_hydra_config()
    _main()