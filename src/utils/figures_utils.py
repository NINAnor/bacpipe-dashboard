import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.manifold import TSNE


def get_2d_features(features, perplexity=8):
    return TSNE(n_components=2, perplexity=perplexity).fit_transform(features)


def get_figure(features_2d, labels, file_paths=None, fig_name=None):
    # Create a DataFrame for plotting
    data = {"x": features_2d[:, 0], "y": features_2d[:, 1], "label": labels}

    # Add file paths to the data if provided
    if file_paths is not None:
        data["file_path"] = file_paths

    df = pd.DataFrame(data)

    # Get unique labels and count them
    unique_labels = sorted(df["label"].unique())
    num_categories = len(unique_labels)

    # Determine hover data fields
    hover_data = ["label"]
    if "file_path" in df.columns:
        hover_data.append("file_path")

    if num_categories <= 10:
        # For few categories, use default color scheme
        fig = px.scatter(
            df,
            x="x",
            y="y",
            color="label",
            hover_data=hover_data,  # Add hover data
            height=600,
            title="ESC-50 Embeddings",
            labels={"x": "Dimension 1", "y": "Dimension 2", "label": "Sound Category"},
        )
    else:
        # For many categories, create a custom color map
        # Generate distinct colors using HSL color space
        hues = np.linspace(0, 1, num_categories, endpoint=False)
        saturations = np.ones(num_categories) * 0.7  # Fixed saturation
        lightnesses = np.ones(num_categories) * 0.5  # Fixed lightness

        # Convert HSL to RGB
        colors = []
        for h, s, light in zip(hues, saturations, lightnesses, strict=False):
            # HSL to RGB conversion
            c = (1 - abs(2 * light - 1)) * s
            x = c * (1 - abs((h * 6) % 2 - 1))
            m = light - c / 2

            if h < 1 / 6:
                r, g, b = c, x, 0
            elif h < 2 / 6:
                r, g, b = x, c, 0
            elif h < 3 / 6:
                r, g, b = 0, c, x
            elif h < 4 / 6:
                r, g, b = 0, x, c
            elif h < 5 / 6:
                r, g, b = x, 0, c
            else:
                r, g, b = c, 0, x

            colors.append(
                f"rgb({int((r + m) * 255)}, {int((g + m) * 255)}, {int((b + m) * 255)})"
            )

        # Create a color mapping
        color_map = {
            label: color for label, color in zip(unique_labels, colors, strict=False)
        }

        # Create figure using the custom color map
        fig = px.scatter(
            df,
            x="x",
            y="y",
            color="label",
            hover_data=hover_data,  # Add hover data
            color_discrete_map=color_map,
            height=600,
            title="ESC-50 Embeddings",
            labels={"x": "Dimension 1", "y": "Dimension 2", "label": "Sound Category"},
        )

        # Improve legend layout for many categories
        fig.update_layout(
            legend=dict(
                itemsizing="constant",
                font=dict(size=10),
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.05,
                title=dict(text="Sound Category"),
            )
        )

    return fig


def get_prototype_figure(features_2d, labels, prototypes_2d, prototype_labels):
    """Create a figure showing both data points and class prototypes"""

    # Create DataFrame for data points
    df_data = pd.DataFrame(
        {
            "x": features_2d[:, 0],
            "y": features_2d[:, 1],
            "label": labels,
            "type": "Data point",
        }
    )

    # Create DataFrame for prototypes
    df_proto = pd.DataFrame(
        {
            "x": prototypes_2d[:, 0],
            "y": prototypes_2d[:, 1],
            "label": prototype_labels,
            "type": "Prototype",
        }
    )

    # Combine the DataFrames
    df = pd.concat([df_data, df_proto])

    # Create the figure
    fig = px.scatter(
        df,
        x="x",
        y="y",
        color="label",
        symbol="type",
        height=600,
        title="Embeddings with Class Prototypes",
        labels={"x": "Dimension 1", "y": "Dimension 2", "label": "Sound Category"},
    )

    # Make prototypes larger
    fig.update_traces(
        selector=dict(mode="markers", name="Prototype"),
        marker=dict(size=15, line=dict(width=2, color="DarkSlateGrey")),
    )

    return fig


def plot_confusion_matrix(cm, class_names, title="Confusion Matrix"):
    # Normalize the confusion matrix
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)  # Replace NaNs with 0s

    # Create the heatmap
    fig = ff.create_annotated_heatmap(
        z=cm_norm,
        x=class_names,
        y=class_names,
        annotation_text=cm.astype(int),  # Show the raw counts as integers
        colorscale="Viridis",
    )

    # Add title and adjust layout
    fig.update_layout(
        title_text=title,
        xaxis=dict(title="Predicted Class"),
        yaxis=dict(
            title="True Class", autorange="reversed"
        ),  # Reverse to match sklearn's orientation
    )

    # If there are many classes, hide the axis labels
    if len(class_names) > 15:
        fig.update_layout(
            xaxis=dict(showticklabels=False, title="Predicted Class"),
            yaxis=dict(showticklabels=False, title="True Class", autorange="reversed"),
        )

    return fig
