import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def display_embedding_info(embeddings, labels, is_validation=False):
    dataset_type = "Validation" if is_validation else "Full"
    st.write(f"{dataset_type} Embedding shape: {embeddings.shape}")
    unique_labels = set(labels)
    st.write(f"Number of classes: {len(unique_labels)}")
    st.write(f"Classes: {', '.join(sorted(unique_labels))}")


def get_correct_audio_path(embedding_path, data_dir):
    try:
        embedding_path_obj = Path(embedding_path)
        base_name = embedding_path_obj.name

        # Try to extract ESC-50 identifier pattern
        pattern = r"(\d+-\d+-[A-Z]-\d+)"
        match = re.search(pattern, base_name)

        if match:
            esc_id = match.group(1)
            audio_path = Path(data_dir) / "audio" / f"{esc_id}.wav"
            if audio_path.exists():
                return str(audio_path)

        # Fallback to searching by prefix
        audio_dir = Path(data_dir) / "audio"
        if audio_dir.exists():
            parts = base_name.split("-")
            if len(parts) >= 2:
                prefix = f"{parts[0]}-{parts[1]}"
                for file in audio_dir.iterdir():
                    if file.name.startswith(prefix) and file.name.endswith(".wav"):
                        return str(file)

        return embedding_path
    except Exception:
        return embedding_path


def render_interactive_scatterplot(
    points,
    labels,
    audio_paths=None,
    height=600,
    title="Embedding Visualization",
    form_key_prefix=None,
):
    data_dir = st.session_state.config["DATA_DIR"]

    # Create a unique form key based on the provided prefix or title
    if form_key_prefix is None:
        # Create a key from the title by removing spaces and special chars
        form_key_prefix = "".join(c for c in title if c.isalnum()).lower()

    # Create DataFrame
    df = pd.DataFrame(
        {
            "x": points[:, 0].astype(float),
            "y": points[:, 1].astype(float),
            "label": labels,
            "point_id": range(len(labels)),
            "file_path": audio_paths if audio_paths else [""] * len(labels),
        }
    )

    fig = go.Figure()

    # Get unique labels and assign colors
    unique_labels = sorted(list(set(labels)))

    # Add traces for each label group
    for label in unique_labels:
        group = df[df["label"] == label]

        # Create audio file paths for hover info
        audio_paths_list = []
        embedding_filenames = []

        for _, row in group.iterrows():
            embedding_path = row["file_path"]
            embedding_filenames.append(Path(embedding_path).name)

            # Get corresponding audio file path
            audio_path = get_correct_audio_path(embedding_path, data_dir)
            audio_paths_list.append(Path(audio_path).name)

        # Create customdata with more information
        customdata = np.stack(
            (group["label"], group["point_id"], embedding_filenames, audio_paths_list),
            axis=-1,
        )

        fig.add_trace(
            go.Scatter(
                x=group["x"],
                y=group["y"],
                mode="markers",
                name=str(label),
                marker=dict(
                    size=12, opacity=0.8, line=dict(width=1.5, color="DarkSlateGrey")
                ),
                hovertemplate=(
                    "<b>Label:</b> %{customdata[0]}<br>"
                    + "<b>Point ID:</b> %{customdata[1]}<br>"
                    + "<b>Embedding:</b> %{customdata[2]}<br>"
                    + "<b>Audio:</b> %{customdata[3]}<extra></extra>"
                ),
                customdata=customdata,
            )
        )

    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="t-SNE dimension 1",
        yaxis_title="t-SNE dimension 2",
        height=height,
        hovermode="closest",
        legend_title="Category",
    )

    # Display the figure (without interactivity)
    st.plotly_chart(fig, use_container_width=True)

    ###############################
    # AUDIO PART OF THE DASHBOARD #
    ###############################

    st.subheader("Audio Player")

    # Use unique form keys for each instance
    with st.form(f"{form_key_prefix}_point_id_form"):
        point_id = st.number_input(
            "Enter Point ID", min_value=0, max_value=len(df) - 1, value=0
        )
        play_button = st.form_submit_button("Play Audio")

        if play_button:
            point_row = df[df["point_id"] == point_id].iloc[0]
            embedding_path = point_row["file_path"]
            audio_path = get_correct_audio_path(embedding_path, data_dir)

            # Display information
            st.subheader(f"Point {point_id}")
            st.write(f"**Category:** {point_row['label']}")
            st.write(f"**File:** `{Path(audio_path).name}`")

            # Play audio
            if Path(audio_path).exists() and audio_path.endswith(".wav"):
                st.audio(audio_path, format="audio/wav")
            else:
                st.error(f"Audio file not found: {audio_path}")
