import pandas as pd
import plotly.express as px
import streamlit as st

from utils.figures_utils import plot_confusion_matrix
from utils.metrics import calculate_classification_metrics


def render():
    if (
        "original_results" not in st.session_state
        or "finetune_results" not in st.session_state
        or "proto_results" not in st.session_state
    ):
        st.warning("Please complete all previous tabs before viewing comparison")
        return

    # Get data from session state
    original_embeddings = st.session_state.original_results["embeddings"]
    original_metrics = st.session_state.original_results["metrics"]

    transf_embeddings = st.session_state.finetune_results["embeddings"]
    transf_metrics = st.session_state.finetune_results["metrics"]

    proto_embeddings = st.session_state.proto_results["embeddings"]
    proto_metrics = st.session_state.proto_results["metrics"]

    # Get labels
    y_val = st.session_state.train_val_split["y_val"]

    summary_dashboard(
        transf_embeddings,
        transf_metrics,
        proto_embeddings,
        proto_metrics,
        original_embeddings,
        original_metrics,
        y_val,
    )


def summary_dashboard(
    transf_embeddings,
    transf_metrics,
    proto_embeddings,
    proto_metrics,
    original_embeddings,
    original_metrics,
    y_val,
):
    # Calculate classification metrics for each embedding type
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
            "Original": "#636EFA",
            "Fine-tuning": "#EF553B",
            "Prototypical": "#00CC96",
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
