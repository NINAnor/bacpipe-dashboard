from pathlib import Path

import pandas as pd
import streamlit as st

from components.caching import cached_load_embeddings, cached_split_data
from components.display import display_embedding_info
from utils.data_utils import prepare_embedding_data
from utils.path_utils import generate_embeddings


def render(cfg, model_changed, test_size_changed):
    data_dir = cfg["DATA_DIR"]
    metadata_path = cfg["METADATA_PATH"]

    # GENERATE THE EMBEDDINGS - Cache with model
    if "embeddings" not in st.session_state or model_changed:
        with st.status("Generating embeddings...") as status:
            selected_model = st.session_state.last_params["model"]
            loader = generate_embeddings(
                selected_model, data_dir, check_if_primary_combination_exists=True
            )
            embed_dir = loader.embed_dir

            embeddings, labels, embedding_paths = cached_load_embeddings(
                embed_dir, metadata_path, selected_model
            )

            # Map to audio files
            metadata_df = pd.read_csv(metadata_path)
            audio_files = []
            for path in embedding_paths:
                filename = Path(path).name.split(".")[0]
                matches = metadata_df[
                    metadata_df["filename"].str.contains(filename, case=False, na=False)
                ]

                if len(matches) > 0:
                    audio_path = (
                        Path(data_dir) / "audio" / f"{matches['filename'].iloc[0]}.wav"
                    )
                    audio_files.append(str(audio_path))
                else:
                    audio_files.append(path)

            st.session_state.embeddings = embeddings
            st.session_state.labels = labels
            st.session_state.audio_files = audio_files

            status.update(
                label=f"Loaded {len(labels)} embeddings",
                state="complete",
                expanded=False,
            )

    else:
        embeddings = st.session_state.embeddings
        labels = st.session_state.labels
        audio_files = st.session_state.audio_files

    with st.status("Splitting data..."):
        test_size = st.session_state.last_params["test_size"]
        X_train, X_val, y_train, y_val, train_files, val_files = cached_split_data(
            embeddings, labels, audio_files, test_size
        )

        train_data = prepare_embedding_data(X_train, y_train)
        val_data = prepare_embedding_data(X_val, y_val)

        st.session_state.train_val_split = {
            "X_train": X_train,
            "X_val": X_val,
            "y_train": y_train,
            "y_val": y_val,
            "train_files": train_files,
            "val_files": val_files,
            "train_data": train_data,
            "val_data": val_data,
        }

        st.success(f"Train: {len(y_train)} samples, Validation: {len(y_val)} samples")
    display_embedding_info(X_val, y_val, is_validation=True)
