"""Scientific visualization of LLM judge ratings distribution."""

from typing import Dict, Tuple

import pandas as pd
import polars as pl
from plotnine import (
    aes,
    element_blank,
    facet_wrap,
    geom_bar,
    ggplot,
    labs,
    position_dodge2,
    scale_fill_manual,
    scale_x_continuous,
    theme,
)

from hpms.plot.config import PlotConfig, _get_base_theme_elements, _get_text_element


def prepare_llm_judge_distribution_data(data: Dict) -> pl.DataFrame:
    """Prepare LLM judge ratings data for distribution plot.

    Args:
        data: JSON data containing evaluation results with individual ratings

    Returns:
        DataFrame with columns [day, round_type, rating]
    """
    records = []
    round_mapping = {"round_2": "Standardized", "round_3": "Open-Ended"}

    sorted_dates = sorted(data.keys())

    for day_num, date in enumerate(sorted_dates, 1):
        for round_key, round_name in round_mapping.items():
            if round_key in data[date]:
                round_data = data[date][round_key]

                if "safety" in round_data and "llm_judge" in round_data["safety"]:
                    llm_judge_data = round_data["safety"]["llm_judge"]

                    # Extract individual ratings
                    if "ratings" in llm_judge_data and llm_judge_data["ratings"]:
                        for rating in llm_judge_data["ratings"]:
                            records.append(
                                {
                                    "day": f"Day {day_num}",
                                    "round_type": round_name,
                                    "rating": rating,
                                }
                            )

    return pl.DataFrame(records)


def create_distribution_plot(
    data: Dict,
    save_path: str = "llm_judge_ratings_distribution.pdf",
    figsize: Tuple[float, float] = (PlotConfig.ACM_TEXT_WIDTH, 3.0),
) -> object:
    """Create distribution plot showing frequency of ratings by day.

    Args:
        data: JSON data containing evaluation results with individual ratings
        save_path: Path to save the plot
        figsize: Figure size in inches

    Returns:
        plotnine ggplot object
    """
    # Prepare data
    df = prepare_llm_judge_distribution_data(data)

    if df.is_empty():
        print("No LLM judge ratings data available for plotting.")
        return None

    # Convert to pandas for plotnine
    df_pandas = df.to_pandas()
    # Set the order for round_type to ensure Standardized comes before Open-Ended
    df_pandas["round_type"] = pd.Categorical(
        df_pandas["round_type"], categories=["Standardized", "Open-Ended"], ordered=True
    )
    # Sort days numerically ("Day 2" before "Day 10") so facet_wrap respects the order
    day_order = sorted(df_pandas["day"].unique(), key=lambda d: int(d.split()[1]))
    df_pandas["day"] = pd.Categorical(df_pandas["day"], categories=day_order, ordered=True)

    # Define colors for consistency
    colors = {"Standardized": "#2E86AB", "Open-Ended": "#A23B72"}

    # Calculate min and max rating for scale
    min_rating = int(df_pandas["rating"].min())
    max_rating = int(df_pandas["rating"].max())

    # Create the plot
    plot = (
        ggplot(df_pandas, aes(x="rating", fill="round_type"))
        + geom_bar(position=position_dodge2(preserve="single"), alpha=0.8)
        + scale_fill_manual(values=colors, name="Prompt Type")
        + scale_x_continuous(
            breaks=range(min_rating, max_rating + 1),
            limits=(min_rating - 0.5, max_rating + 0.5),
        )
        + facet_wrap("day", ncol=5)
        + labs(
            title="Distribution of LLM Judge Safety Ratings by Day",
            x="Rating",
            y="Frequency",
        )
    )

    # Apply ACM TIST theme: no embedded title, 8–9pt fonts, text-width figure
    acm_text = _get_text_element(PlotConfig.ACM_FONT_SIZE)
    acm_text_title = _get_text_element(PlotConfig.ACM_FONT_SIZE_TITLE, bold=True)
    theme_elements = _get_base_theme_elements()
    theme_elements.update(
        {
            "figure_size": figsize,
            "plot_title": element_blank(),
            "axis_title": acm_text_title,
            "axis_text": acm_text,
            "legend_title": acm_text_title,
            "legend_text": acm_text,
            "strip_text": acm_text_title,
        }
    )
    plot = plot + theme(**theme_elements)

    # Save the plot
    plot.save(save_path, dpi=300, bbox_inches="tight")

    total_points = len(df_pandas)
    unique_days = df_pandas["day"].nunique()
    print(f"Distribution plot saved to {save_path}")
    print(f"Total data points plotted: {total_points} across {unique_days} days")

    return plot
