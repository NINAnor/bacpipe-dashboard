import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# Run from the bacpipe repository root to fix relative imports
def run_in_bacpipe_context(func):
    """A decorator that runs a function in the context of the bacpipe repository"""

    def wrapper(*args, **kwargs):
        original_dir = Path.cwd()
        original_path = sys.path.copy()

        # Change to bacpipe directory
        bacpipe_repo_dir = Path(__file__).parent.parent.parent / "bacpipe"
        bacpipe_repo_dir = bacpipe_repo_dir.resolve()  # Get absolute path
        os.chdir(bacpipe_repo_dir)  # Still need os.chdir as Path has no equivalent

        # Add bacpipe repo root to sys.path
        if str(bacpipe_repo_dir) not in sys.path:
            sys.path.insert(0, str(bacpipe_repo_dir))

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
        check_if_primary_combination_exists=check_if_primary_combination_exists,
    )

    # Convert loader.embed_dir to Path object if it's a string
    embed_dir = Path(loader.embed_dir)

    if not embed_dir.is_absolute():
        bacpipe_repo_dir = Path.cwd()
        embed_dir_path = (bacpipe_repo_dir / loader.embed_dir).resolve()
        audio_dir_path = embed_dir_path / "audio"

        if audio_dir_path.exists():
            loader.embed_dir = str(audio_dir_path)
        else:
            loader.embed_dir = str(embed_dir_path)

    return loader


def load_embeddings_with_labels(embed_dir, metadata_path, model_name="birdnet"):
    """Load embeddings and match them with labels from the ESC-50 metadata"""
    metadata = pd.read_csv(metadata_path)
    embed_dir_path = Path(embed_dir)

    embeddings = []
    labels = []
    file_paths = []

    for file_path in embed_dir_path.iterdir():
        if file_path.name.endswith(f"_{model_name}.npy"):
            original_filename = file_path.name.replace(f"_{model_name}.npy", ".wav")

            row = metadata[metadata["filename"] == original_filename]

            if not row.empty:
                embedding = np.load(file_path)
                label = row["category"].values[0]

                if len(embedding.shape) > 1 and embedding.shape[0] > 1:
                    for i in range(embedding.shape[0]):
                        embeddings.append(embedding[i])
                        labels.append(label)
                        file_paths.append(f"{file_path}:segment{i}")
                else:
                    embeddings.append(embedding.squeeze())
                    labels.append(label)
                    file_paths.append(str(file_path))

    if embeddings:
        embeddings = np.vstack(embeddings)
    else:
        embeddings = np.array([])

    print(f"Embeddings shape: {embeddings.shape}, Labels length: {len(labels)}")

    return embeddings, labels, file_paths
