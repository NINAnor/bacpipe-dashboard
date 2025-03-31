import os
import sys
import numpy as np
import pandas as pd

# Run from the bacpipe repository root to fix relative imports
def run_in_bacpipe_context(func):
    """A decorator that runs a function in the context of the bacpipe repository"""
    def wrapper(*args, **kwargs):
        original_dir = os.getcwd()
        original_path = sys.path.copy()
        
        # Change to bacpipe directory
        bacpipe_repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "bacpipe"))
        os.chdir(bacpipe_repo_dir)
        
        # Add bacpipe repo root to sys.path
        if bacpipe_repo_dir not in sys.path:
            sys.path.insert(0, bacpipe_repo_dir)
        
        try:
            return func(*args, **kwargs)
        finally:
            # Restore original directory and path
            os.chdir(original_dir)
            sys.path = original_path
    
    return wrapper

@run_in_bacpipe_context
def generate_embeddings(model_name, data_dir, check_if_primary_combination_exists=True):

    from bacpipe.main import get_embeddings
    
    # Get the embeddings loader from bacpipe
    loader = get_embeddings(
        model_name=model_name,
        audio_dir=data_dir, 
        dim_reduction_model="None",
        check_if_primary_combination_exists=check_if_primary_combination_exists
    )
    
    if not os.path.isabs(loader.embed_dir):
        bacpipe_repo_dir = os.getcwd()
        embed_dir_path = os.path.abspath(os.path.join(bacpipe_repo_dir, loader.embed_dir))
        audio_dir_path = os.path.join(embed_dir_path, "audio")
        
        if os.path.exists(audio_dir_path):
            loader.embed_dir = audio_dir_path
        else:
            loader.embed_dir = embed_dir_path
    
    return loader


def load_embeddings_with_labels(embed_dir, metadata_path, model_name="birdnet"):
    """Load embeddings and match them with labels from the ESC-50 metadata"""
    metadata = pd.read_csv(metadata_path)
    
    embeddings = []
    labels = []
    file_paths = []
    
    for file in os.listdir(embed_dir):
        if file.endswith(f'_{model_name}.npy'): 
            original_filename = file.replace(f'_{model_name}.npy', '.wav')
            
            row = metadata[metadata['filename'] == original_filename]
            
            if not row.empty:
                embedding_path = os.path.join(embed_dir, file)
                embedding = np.load(embedding_path)
                
                label = row['category'].values[0]
                
                if len(embedding.shape) > 1 and embedding.shape[0] > 1:
                    for i in range(embedding.shape[0]):
                        embeddings.append(embedding[i])
                        labels.append(label) 
                        file_paths.append(f"{embedding_path}:segment{i}")
                else:
                    embeddings.append(embedding.squeeze()) 
                    labels.append(label)
                    file_paths.append(embedding_path)
    
    if embeddings:
        embeddings = np.vstack(embeddings)
    else:
        embeddings = np.array([])
    
    print(f"Embeddings shape: {embeddings.shape}, Labels length: {len(labels)}")
    
    return embeddings, labels, file_paths