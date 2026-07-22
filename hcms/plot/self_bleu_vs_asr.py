"""Plot Self-BLEU vs ASR - Extended with four-plot functionality"""

from typing import Dict, List, Optional, Tuple

import pandas as pd
from hcms.plot.config import PlotConfig, _get_base_theme_elements, _get_text_element
from plotnine import (
    aes,
    facet_wrap,
    geom_point,
    ggplot,
    labs,
    scale_color_manual,
    scale_shape_manual,
    theme,
    theme_minimal,
    xlim,
    ylim,
)


def create_four_plots(
    datasets: List[Dict[str, Tuple[float, float]]],
    titles: Optional[List[str]] = None,
    fig_size: Tuple[int, int] = (16, 12),
) -> ggplot:
    """Create four publication-ready scatter plots in a 2x2 grid.

    Args:
        datasets: List of 4 dictionaries, each mapping model names
                 to (offensive_pct, self_bleu) tuples
        titles: List of 4 titles for each subplot.
               Defaults to "Plot 1", "Plot 2", etc.
        fig_size: Figure size as (width, height) tuple

    Returns:
        gg_plot object with faceted subplots
    """
    if len(datasets) != 4:
        raise ValueError("Exactly 4 datasets required for four-plot layout")

    if titles is None:
        titles = [f"Plot {i + 1}" for i in range(4)]
    if len(titles) != 4:
        raise ValueError("Exactly 4 titles required if provided")

    # Combine all datasets into single DataFrame with facet variable
    all_data = []
    for model_data, title in zip(datasets, titles):
        for model, (x, y) in model_data.items():
            all_data.append(
                {
                    "model": model,
                    "offensive_pct": x,
                    "self_bleu": y,
                    "model_family": get_model_family(model),
                    "facet": title,
                }
            )

    df = pd.DataFrame(all_data)

    # Create the plot with facets
    theme_elements = _get_base_theme_elements()
    theme_elements.update(
        {
            "figure_size": fig_size,
            "strip_text": _get_text_element(PlotConfig.FONT_SIZE_BOLD, bold=True),
        }
    )

    p = (
        ggplot(
            df,
            aes(x="offensive_pct", y="self_bleu", shape="model", color="model"),
        )
        + geom_point(size=6, alpha=0.8, stroke=0.6)
        + scale_shape_manual(values=get_shape_map(df["model"].unique()), name="Model")
        + scale_color_manual(
            values=get_model_color_map(df["model"].unique()),
            name="Model",
            na_value="gray",
        )
        + labs(
            x="Offensive Replies (%)",
            y="Self-BLEU Score",
        )
        # + scale_x_log10()
        # + scale_y_log10()
        + xlim(0, 25)
        + ylim(0.55, 0.9)
        + facet_wrap("~facet", nrow=2, ncol=2)
        + theme_minimal()
        + theme(**theme_elements)
    )

    return p


def create_publication_plot_with_shapes(
    model_data: Dict[str, Tuple[float, float]],
) -> ggplot:
    """Create publication-ready scatter plot with custom shapes.

    Args:
        model_data: Dictionary mapping model names to
                   (offensive_pct, self_bleu) tuples

    Returns:
        gg_plot object
    """
    # Convert data to DataFrame
    df = pd.DataFrame(
        [
            {
                "model": model,
                "offensive_pct": x,
                "self_bleu": y,
                "model_family": get_model_family(model),
            }
            for model, (x, y) in model_data.items()
        ]
    )

    # Create the plot with larger elements
    theme_elements = _get_base_theme_elements()
    theme_elements.update({"figure_size": (15, 8)})

    p = (
        ggplot(
            df,
            aes(x="offensive_pct", y="self_bleu", shape="model", color="model"),
        )
        + geom_point(size=8, alpha=0.9, stroke=1)
        + scale_shape_manual(values=get_shape_map(df["model"].unique()), name="Model")
        + scale_color_manual(
            values=get_model_color_map(df["model"].unique()),
            name="Model",
            na_value="gray",
        )
        + labs(
            # title="Self-BLEU (Diversity) vs ASR (Attack Success Rate)",
            # subtitle="Lower Self-BLEU indicates higher response diversity",
            x="Offensive Replies (%)",
            y="Self-BLEU Score",
        )
        + xlim(6, 20)
        + ylim(0.55, 0.83)
        + theme_minimal()
        + theme(**theme_elements)
    )

    return p


def get_model_family(model_name: str) -> str:
    """Categorize models by family.

    Args:
        model_name: Name of the model

    Returns:
        Model family name
    """
    if model_name.startswith("GPT"):
        return "GPT"
    if model_name.startswith("Gemini"):
        return "Gemini"
    if model_name.startswith("Llama"):
        return "Llama"
    return "Other"


def get_model_color_map(model_names: List[str]) -> Dict[str, str]:
    """Create a mapping from model names to family colors."""
    family_colors = {
        "GPT": "#b3589a",
        "Gemini": "#4285F4",
        "Llama": "#FF8C00",
        "Other": "gray",
    }
    return {
        name: family_colors.get(get_model_family(name), "gray") for name in model_names
    }


def get_shape_map(model_names: List[str]) -> Dict[str, str]:
    """Create a mapping from model names to a list of shapes."""
    # A list of shapes that plotnine can use
    shapes = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "X", "+", "x"]
    # Sort names to ensure consistent shape assignment
    return {name: shapes[i % len(shapes)] for i, name in enumerate(sorted(model_names))}
