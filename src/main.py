import hydra
import streamlit as st
from hydra.core.global_hydra import GlobalHydra

# Import components
from components.sidebar import setup_sidebar

# Import tab renderers
from tabs import compare_tab, data_tab, fine_tune_tab, original_tab, proto_tab


def reset_hydra_config():
    """Reset Hydra's global state to allow reinitialization"""
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()


def run_dashboard(cfg):
    st.title("Audio Embedding Visualization Dashboard")

    # SIDEBAR CONTROLS
    selected_model, test_size, hidden_dim, epochs = setup_sidebar()

    if "last_params" not in st.session_state:
        st.session_state.last_params = {
            "model": selected_model,
            "test_size": test_size,
            "hidden_dim": hidden_dim,
            "epochs": epochs,
        }

    # Check which parameters changed
    model_changed = selected_model != st.session_state.last_params["model"]
    test_size_changed = test_size != st.session_state.last_params["test_size"]
    hidden_dim_changed = hidden_dim != st.session_state.last_params["hidden_dim"]
    epochs_changed = epochs != st.session_state.last_params["epochs"]

    # Create tabs for different sections
    data_tab_ui, original_tab_ui, finetune_tab_ui, proto_tab_ui, compare_tab_ui = (
        st.tabs(
            [
                "Data Loading",
                "Original Embeddings",
                "Fine-tuned",
                "Prototypical",
                "Comparison",
            ]
        )
    )

    # Update state with current parameters
    st.session_state.last_params = {
        "model": selected_model,
        "test_size": test_size,
        "hidden_dim": hidden_dim,
        "epochs": epochs,
    }

    # Render each tab
    with data_tab_ui:
        data_tab.render(cfg, model_changed, test_size_changed)

    with original_tab_ui:
        original_tab.render(model_changed, test_size_changed)

    with finetune_tab_ui:
        fine_tune_tab.render(
            model_changed, test_size_changed, hidden_dim_changed, epochs_changed
        )

    with proto_tab_ui:
        proto_tab.render(
            model_changed, test_size_changed, hidden_dim_changed, epochs_changed
        )

    with compare_tab_ui:
        compare_tab.render()


@hydra.main(version_base=None, config_path="../", config_name="config")
def _main(cfg):
    # Store config in session state for persistence
    if "config" not in st.session_state:
        st.session_state["config"] = {
            "DATA_DIR": cfg.get("DATA_DIR", ""),
            "METADATA_PATH": cfg.get("METADATA_PATH", ""),
        }

    # Call the dashboard with the config
    run_dashboard(st.session_state["config"])


if __name__ == "__main__":
    reset_hydra_config()
    _main()
