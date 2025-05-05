import streamlit as st
from fine_tune_embeddings import train_embedding_model
from proto_network import train_proto_network
from utils.data_utils import split_data
from utils.figures_utils import get_2d_features
from utils.path_utils import load_embeddings_with_labels

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