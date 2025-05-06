import streamlit as st

from components.caching import cached_train_proto_network, cached_tsne_features
from proto_network import get_proto_transformed_embeddings
from utils.metrics import calculate_clustering_metrics
from components.display import render_interactive_scatterplot

def render(model_changed, test_size_changed, hidden_dim_changed, epochs_changed):
    """Render the Prototypical Network tab"""
    if 'train_val_split' not in st.session_state:
        st.warning("Please load data first by clicking the 'Data Loading' tab")
        return
    
    # Access data from session state
    split_data = st.session_state.train_val_split
    train_data = split_data['train_data']
    val_data = split_data['val_data']
    val_files = split_data['val_files']
    
    # Determine if recomputation is needed
    relevant_changed = model_changed or test_size_changed or hidden_dim_changed or epochs_changed
    hidden_dim = st.session_state.last_params['hidden_dim']
    epochs = st.session_state.last_params['epochs']
    
    if 'proto_results' not in st.session_state or relevant_changed:
        # train the model
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
        
        # Create visualization with t-SNE
        transformed_features_2d = cached_tsne_features(proto_embeddings, 8)
        
        # Store results in session state
        st.session_state.proto_results = {
            'embeddings': proto_embeddings,
            'metrics': metrics,
            'model': model,
            'prototypes': prototypes,
            'features_2d': transformed_features_2d,  # Store 2D features for reuse
            'plot_labels': plot_labels  # Store labels for reuse
        }

        # Viz the embeddings
        render_interactive_scatterplot(
            points=transformed_features_2d,
            labels=plot_labels,
            audio_paths=val_files,
            height=600,
            title="Prototypical Network Embeddings"
        )

    else:        
        # Get data from session state
        proto_embeddings = st.session_state.proto_results['embeddings']
        metrics = st.session_state.proto_results['metrics']
        
        # Get or recalculate 2D features and labels
        if 'features_2d' in st.session_state.proto_results:
            transformed_features_2d = st.session_state.proto_results['features_2d']
            plot_labels = st.session_state.proto_results['plot_labels']
        else:
            # Get labels for plotting if not stored
            id_to_label = {v: k for k, v in val_data["label_to_id"].items()}
            plot_labels = [id_to_label[val.item()] for val in val_data["y"]]
            
            # Recalculate visualization if not stored
            transformed_features_2d = cached_tsne_features(proto_embeddings, 8)
        
        # Display header and metrics
        st.header("Prototypical Network Embeddings (Validation Set)")
        st.info(f"Embedding Metrics - ARI: {metrics['ari']:.4f}, AMI: {metrics['ami']:.4f}")

        # Viz the embeddings
        render_interactive_scatterplot(
            points=transformed_features_2d,
            labels=plot_labels,
            audio_paths=val_files,
            height=600,
            title="Prototypical Network Embeddings"
        )