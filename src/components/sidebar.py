import streamlit as st

def setup_sidebar():
    AVAILABLE_MODELS = [
        "birdnet", "perch_bird", "insect66", "rcl_fs_bsed", "aves_especies",
        "biolingual", "protoclr", "surfperch", "animal2vec_xc", "avesecho_passt",
        "birdaves_especies", "vggish", "google_whale", "audiomae", "hbdet", "mix2"
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