import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.manifold import TSNE


def get_2d_features(features, perplexity=8):
    return TSNE(n_components=2, perplexity=perplexity).fit_transform(features)


def get_figure(features_2d, labels, fig_name=None):
    # Create a DataFrame for plotting
    df = pd.DataFrame({"x": features_2d[:, 0], "y": features_2d[:, 1], "label": labels})

    # Get unique labels and count them
    unique_labels = sorted(df["label"].unique())
    num_categories = len(unique_labels)

    if num_categories <= 10:
        # For few categories, use default color scheme
        fig = px.scatter(
            df,
            x="x",
            y="y",
            color="label",
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
