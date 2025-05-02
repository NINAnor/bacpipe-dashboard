import hydra
import pandas as pd
import plotly.express as px
import streamlit as st
from hydra.core.global_hydra import GlobalHydra

from streamlit_plotly_events import plotly_events

import os

from fine_tune_embeddings import get_transformed_embeddings, train_embedding_model
from proto_network import get_proto_transformed_embeddings, train_proto_network
from utils.data_utils import prepare_embedding_data, split_data
from utils.figures_utils import get_2d_features, get_figure, plot_confusion_matrix
from utils.metrics import calculate_classification_metrics, calculate_clustering_metrics
from utils.path_utils import generate_embeddings, load_embeddings_with_labels


# Update all caching functions that handle PyTorch tensors
@st.cache_data
def cached_tsne_features(_features, perplexity):
    """Cache t-SNE computation"""
    return get_2d_features(_features, perplexity)

@st.cache_resource
def cached_train_embedding_model(_train_embeddings, _train_labels, _val_embeddings, _val_labels, hidden_dim, epochs):
    """Cache trained embedding model"""
    return train_embedding_model(_train_embeddings, _train_labels, _val_embeddings, _val_labels, 
                                 hidden_dim=hidden_dim, epochs=epochs)

@st.cache_resource
def cached_train_proto_network(_train_embeddings, _train_labels, _val_embeddings, _val_labels, hidden_dim, epochs):
    """Cache trained prototypical network"""
    return train_proto_network(_train_embeddings, _train_labels, _val_embeddings, _val_labels, 
                              embedding_dim=hidden_dim, epochs=epochs)

@st.cache_data
def cached_split_data(_embeddings, _labels, _files, test_size):
    """Cache data splitting"""
    return split_data(_embeddings, _labels, _files, test_size)

@st.cache_data(ttl=3600)
def cached_load_embeddings(embed_dir, metadata_path, model_name):
    """Cache embedding loading - no tensors in arguments so no changes needed"""
    return load_embeddings_with_labels(embed_dir, metadata_path, model_name=model_name)

def display_embedding_info(embeddings, labels, is_validation=False):
    """Display basic information about the embeddings"""
    dataset_type = "Validation" if is_validation else "Full"
    st.write(f"{dataset_type} Embedding shape: {embeddings.shape}")
    unique_labels = set(labels)
    st.write(f"Number of classes: {len(unique_labels)}")
    st.write(f"Classes: {', '.join(sorted(unique_labels))}")


def display_original_embeddings(embeddings, labels, file_paths, perplexity):
    # Initialize session state for audio playback
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
        # Get the index of the clicked point
        x, y = selected_point[0]["x"], selected_point[0]["y"]
        closest_point = ((df["x"] - x)**2 + (df["y"] - y)**2).idxmin()
        
        # Get the file path and play the audio
        audio_file = df.loc[closest_point, "file_path"]
        
        # Update session state
        if st.session_state.current_audio != audio_file:
            st.session_state.current_audio = audio_file
            st.session_state.selected_point = closest_point
            
            # Create columns for audio player and info
            col1, col2 = st.columns([2, 3])
            
            with col1:
                st.audio(audio_file, format="audio/wav")
                
            with col2:
                st.markdown(f"**File:** `{os.path.basename(audio_file)}`")
                st.markdown(f"**Category:** {df.loc[closest_point, 'label']}")

    return embeddings, metrics


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


def summary_dashboard(
    transf_embeddings,
    transf_metrics,
    proto_embeddings,
    proto_metrics,
    original_embeddings,
    original_metrics,
    y_val,
):
    original_class_metrics = calculate_classification_metrics(
        original_embeddings, y_val
    )
    transf_class_metrics = calculate_classification_metrics(transf_embeddings, y_val)
    proto_class_metrics = calculate_classification_metrics(proto_embeddings, y_val)

    # Combine metrics
    original_metrics.update(original_class_metrics)
    transf_metrics.update(transf_class_metrics)
    proto_metrics.update(proto_class_metrics)

    # Display metrics comparison
    st.header("Embedding Methods Comparison")

    metrics_df = pd.DataFrame(
        {
            "Metric": ["ARI", "AMI", "Accuracy", "F1 Score"],
            "Original Embeddings": [
                f"{original_metrics['ari']:.4f}",
                f"{original_metrics['ami']:.4f}",
                f"{original_metrics['accuracy']:.4f}",
                f"{original_metrics['f1']:.4f}",
            ],
            "Fine-tuned Embeddings": [
                f"{transf_metrics['ari']:.4f}",
                f"{transf_metrics['ami']:.4f}",
                f"{transf_metrics['accuracy']:.4f}",
                f"{transf_metrics['f1']:.4f}",
            ],
            "Prototypical Networks": [
                f"{proto_metrics['ari']:.4f}",
                f"{proto_metrics['ami']:.4f}",
                f"{proto_metrics['accuracy']:.4f}",
                f"{proto_metrics['f1']:.4f}",
            ],
        }
    )

    st.table(metrics_df)

    # Create a bar chart comparing the methods
    st.subheader("Visual Comparison")

    chart_data = pd.DataFrame(
        {
            "Metric": ["ARI", "AMI", "Accuracy", "F1 Score"] * 3,
            "Value": [
                original_metrics["ari"],
                original_metrics["ami"],
                original_metrics["accuracy"],
                original_metrics["f1"],
                transf_metrics["ari"],
                transf_metrics["ami"],
                transf_metrics["accuracy"],
                transf_metrics["f1"],
                proto_metrics["ari"],
                proto_metrics["ami"],
                proto_metrics["accuracy"],
                proto_metrics["f1"],
            ],
            "Method": ["Original"] * 4 + ["Fine-tuning"] * 4 + ["Prototypical"] * 4,
        }
    )

    fig = px.bar(
        chart_data,
        x="Metric",
        y="Value",
        color="Method",
        barmode="group",
        title="Comparison of Embedding Methods",
        height=500,
        color_discrete_map={
            "Original": "#636EFA",  # Blue
            "Fine-tuning": "#EF553B",  # Red/orange
            "Prototypical": "#00CC96",  # Green
        },
    )

    fig.update_layout(
        yaxis_title="Score",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    st.plotly_chart(fig, use_container_width=True)

    # Add confusion matrices section
    st.header("Confusion Matrices")

    cm_tabs = st.tabs(["Original", "Fine-tuned", "Prototypical"])

    with cm_tabs[0]:
        st.write("### Original Embeddings Confusion Matrix")
        cm_fig_original = plot_confusion_matrix(
            original_metrics["cm"],
            original_metrics["class_names"],
            title="Original Embeddings",
        )
        st.plotly_chart(cm_fig_original, use_container_width=True)

    with cm_tabs[1]:
        st.write("### Fine-tuned Embeddings Confusion Matrix")
        cm_fig_t = plot_confusion_matrix(
            transf_metrics["cm"],
            transf_metrics["class_names"],
            title="Fine-tuned Embeddings",
        )
        st.plotly_chart(cm_fig_t, use_container_width=True)

    with cm_tabs[2]:
        st.write("### Prototypical Network Confusion Matrix")
        cm_fig = plot_confusion_matrix(
            proto_metrics["cm"],
            proto_metrics["class_names"],
            title="Prototypical Network",
        )
        st.plotly_chart(cm_fig, use_container_width=True)


def run_dashboard(cfg):
    st.title("Audio Embedding Visualization Dashboard")

    # SIDEBAR CONTROLS
    selected_model, test_size, hidden_dim, epochs = setup_sidebar()

    # Initialize parameter tracking
    if 'last_params' not in st.session_state:
        st.session_state.last_params = {
            'model': selected_model,
            'test_size': test_size,
            'hidden_dim': hidden_dim,
            'epochs': epochs
        }

    # Check which parameters changed
    model_changed = selected_model != st.session_state.last_params['model']
    test_size_changed = test_size != st.session_state.last_params['test_size']
    hidden_dim_changed = hidden_dim != st.session_state.last_params['hidden_dim']
    epochs_changed = epochs != st.session_state.last_params['epochs']

    # Create tabs for different sections
    data_tab, original_tab, finetune_tab, proto_tab, compare_tab = st.tabs([
        "Data Loading", "Original Embeddings", "Fine-tuned", "Prototypical", "Comparison"
    ])

    # PATH SETTINGS
    data_dir = cfg["DATA_DIR"]
    metadata_path = cfg["METADATA_PATH"]

    # Update state with current parameters
    st.session_state.last_params = {
        'model': selected_model,
        'test_size': test_size,
        'hidden_dim': hidden_dim,
        'epochs': epochs
    }

    # DATA LOADING TAB - Always runs when model or test_size changes
    with data_tab:
        # GENERATE THE EMBEDDINGS - Cache with model
        if 'embeddings' not in st.session_state or model_changed:
            with st.status(f"Generating embeddings using {selected_model.upper()}...") as status:
                loader = generate_embeddings(selected_model, data_dir, check_if_primary_combination_exists=True)
                embed_dir = loader.embed_dir
                
                # Load with caching
                embeddings, labels, embedding_paths = cached_load_embeddings(embed_dir, metadata_path, selected_model)
                
                # Map to audio files
                metadata_df = pd.read_csv(metadata_path)
                audio_files = []
                for path in embedding_paths:
                    filename = os.path.basename(path).split('.')[0]
                    matches = metadata_df[metadata_df['filename'].str.contains(filename, case=False, na=False)]
                    
                    if len(matches) > 0:
                        audio_path = os.path.join(data_dir, 'audio', f"{matches['filename'].iloc[0]}.wav")
                        audio_files.append(audio_path)
                    else:
                        audio_files.append(path)
                
                st.session_state.embeddings = embeddings
                st.session_state.labels = labels
                st.session_state.audio_files = audio_files
                
                status.update(label=f"Loaded {len(labels)} embeddings", state="complete", expanded=False)
        else:
            embeddings = st.session_state.embeddings
            labels = st.session_state.labels
            audio_files = st.session_state.audio_files
            st.success(f"Using cached embeddings for {selected_model} ({len(labels)} samples)")

        # SPLIT DATA - Recompute when test_size changes
        if 'train_val_split' not in st.session_state or model_changed or test_size_changed:
            with st.status(f"Splitting data with {test_size:.0%} validation set..."):
                X_train, X_val, y_train, y_val, train_files, val_files = cached_split_data(
                    embeddings, labels, audio_files, test_size
                )
                
                train_data = prepare_embedding_data(X_train, y_train)
                val_data = prepare_embedding_data(X_val, y_val)
                
                st.session_state.train_val_split = {
                    'X_train': X_train, 'X_val': X_val, 
                    'y_train': y_train, 'y_val': y_val,
                    'train_files': train_files, 'val_files': val_files,
                    'train_data': train_data, 'val_data': val_data
                }
                
                st.success(f"Train: {len(y_train)} samples, Validation: {len(y_val)} samples")
        else:
            split_data = st.session_state.train_val_split
            X_train, X_val = split_data['X_train'], split_data['X_val']
            y_train, y_val = split_data['y_train'], split_data['y_val']
            train_files, val_files = split_data['train_files'], split_data['val_files']
            train_data, val_data = split_data['train_data'], split_data['val_data']
            st.success(f"Using cached data split (Train: {len(y_train)}, Val: {len(y_val)})")

    with original_tab:
        if 'train_val_split' not in st.session_state:
            st.warning("Please load data first by clicking the 'Data Loading' tab")
        else:
            # Access data from session state
            split_data = st.session_state.train_val_split
            X_val, y_val = split_data['X_val'], split_data['y_val']
            val_files = split_data['val_files']
            
            # Only recompute if model or test_size changed
            if 'original_results' not in st.session_state or model_changed or test_size_changed:
                display_embedding_info(X_val, y_val, is_validation=True)
                original_embeddings, original_metrics = display_original_embeddings(
                    X_val, y_val, val_files, perplexity=8
                )
                st.session_state.original_results = {
                    'embeddings': original_embeddings,
                    'metrics': original_metrics
                }
            else:
                # Just redisplay without recomputing
                display_embedding_info(X_val, y_val, is_validation=True)
                st.info("Using cached original embeddings visualization")
                
                # Get data from session state
                original_embeddings = st.session_state.original_results['embeddings']
                metrics = st.session_state.original_results['metrics']
                
                # Redisplay metrics
                st.info(f"Metrics - ARI: {metrics['ari']:.4f}, AMI: {metrics['ami']:.2f}")
                
                # Recalculate visualization (quick operation)
                original_features_2d = cached_tsne_features(original_embeddings, 8)
                
                # Create a DataFrame for the plot
                df = pd.DataFrame({
                    "x": original_features_2d[:, 0],
                    "y": original_features_2d[:, 1], 
                    "label": y_val,
                    "file_path": val_files
                })
                
                # Interactive plot
                fig = px.scatter(
                    df, x="x", y="y", color="label", 
                    hover_data=["label", "file_path"],
                    title="Click on a point to play the audio"
                )
                
                # Display plot and handle clicks
                selected_point = plotly_events(fig, click_event=True, override_height=600)
                
                # Audio playback (same as in display_original_embeddings)
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

    # FINE-TUNED EMBEDDINGS TAB
    with finetune_tab:
        if 'train_val_split' not in st.session_state:
            st.warning("Please load data first by clicking the 'Data Loading' tab")
        else:
            # Access data from session state
            split_data = st.session_state.train_val_split
            train_data = split_data['train_data']
            val_data = split_data['val_data']
            
            # Determine if recomputation is needed
            relevant_changed = model_changed or test_size_changed or hidden_dim_changed or epochs_changed
            
            if 'finetune_results' not in st.session_state or relevant_changed:
                # First train model without nested status
                with st.status("Training neural network..."):
                    model = cached_train_embedding_model(
                        train_data["X"], train_data["y"], 
                        val_data["X"], val_data["y"], 
                        hidden_dim, epochs
                    )
                
                # Then display embeddings without nested status
                transf_embeddings, transf_metrics = display_transformed_embeddings(
                    train_data["X"], train_data["y"], 
                    val_data["X"], val_data["y"],
                    val_data["label_to_id"], 8, hidden_dim, epochs,
                    model=model  # Pass the pre-trained model
                )
                
                # Store results in session state
                st.session_state.finetune_results = {
                    'embeddings': transf_embeddings,
                    'metrics': transf_metrics,
                    'model': model
                }
            else:
                # Use cached results
                st.info("Using cached fine-tuned embeddings")
                
                # Get data from session state
                transf_embeddings = st.session_state.finetune_results['embeddings']
                metrics = st.session_state.finetune_results['metrics']
                
                # Display header and metrics
                st.header("Transformed Embeddings (Validation Set)")
                st.info(f"Metrics - ARI: {metrics['ari']:.2f}, AMI: {metrics['ami']:.4f}")
                
                # Get labels for plotting
                id_to_label = {v: k for k, v in val_data["label_to_id"].items()}
                plot_labels = [id_to_label[val.item()] for val in val_data["y"]]
                
                # Recalculate visualization
                transformed_features_2d = cached_tsne_features(transf_embeddings, 8)
                fig_transformed = get_figure(transformed_features_2d, plot_labels)
                st.plotly_chart(fig_transformed, use_container_width=True)

    # PROTOTYPICAL NETWORK TAB
    with proto_tab:
        if 'train_val_split' not in st.session_state:
            st.warning("Please load data first by clicking the 'Data Loading' tab")
        else:
            # Access data from session state
            split_data = st.session_state.train_val_split
            train_data = split_data['train_data']
            val_data = split_data['val_data']
            
            # Determine if recomputation is needed
            relevant_changed = model_changed or test_size_changed or hidden_dim_changed or epochs_changed
            
            if 'proto_results' not in st.session_state or relevant_changed:
                # First train model without nested status
                with st.status("Training prototypical network..."):
                    model, _, prototypes = cached_train_proto_network(
                        train_data["X"], train_data["y"], 
                        val_data["X"], val_data["y"], 
                        hidden_dim, epochs
                    )
                
                # Then display embeddings without nested status
                proto_embeddings, proto_metrics = display_proto_embeddings(
                    train_data["X"], train_data["y"], 
                    val_data["X"], val_data["y"],
                    val_data["label_to_id"], 8, hidden_dim, epochs,
                    model=model,  # Pass the pre-trained model
                    prototypes=prototypes  # Pass the pre-computed prototypes
                )
                
                # Store results in session state
                st.session_state.proto_results = {
                    'embeddings': proto_embeddings,
                    'metrics': proto_metrics,
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
                st.header("Prototypical Network Embeddings (Validation Set)")
                st.info(f"Embedding Metrics - ARI: {metrics['ari']:.4f}, AMI: {metrics['ami']:.4f}")
                
                # Get labels for plotting
                id_to_label = {v: k for k, v in val_data["label_to_id"].items()}
                plot_labels = [id_to_label[val.item()] for val in val_data["y"]]
                
                # Recalculate visualization
                transformed_features_2d = cached_tsne_features(proto_embeddings, 8)
                fig_transformed = get_figure(transformed_features_2d, plot_labels)
                st.plotly_chart(fig_transformed, use_container_width=True)

    # COMPARISON TAB
    with compare_tab:
        if ('original_results' not in st.session_state or 
            'finetune_results' not in st.session_state or 
            'proto_results' not in st.session_state):
            st.warning("Please complete all previous tabs before viewing comparison")
        else:
            # Get data from session state
            original_embeddings = st.session_state.original_results['embeddings']
            original_metrics = st.session_state.original_results['metrics']
            
            transf_embeddings = st.session_state.finetune_results['embeddings']
            transf_metrics = st.session_state.finetune_results['metrics']
            
            proto_embeddings = st.session_state.proto_results['embeddings']
            proto_metrics = st.session_state.proto_results['metrics']
            
            # Get labels
            y_val = st.session_state.train_val_split['y_val']
            
            # Generate comparison
            summary_dashboard(
                transf_embeddings, transf_metrics,
                proto_embeddings, proto_metrics,
                original_embeddings, original_metrics,
                y_val
            )


def reset_hydra_config():
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()


# Use this decorator pattern to safely work with Hydra
@hydra.main(version_base=None, config_path="../", config_name="config")
def _main(cfg):
    # Store config in session state for persistence
    if "config" not in st.session_state:
        st.session_state["config"] = {
            "DATA_DIR": cfg.get("DATA_DIR", ""),
            "METADATA_PATH": cfg.get("METADATA_PATH", ""),
        }

    # Call your actual main function with the config
    run_dashboard(st.session_state["config"])


if __name__ == "__main__":
    reset_hydra_config()
    _main()
