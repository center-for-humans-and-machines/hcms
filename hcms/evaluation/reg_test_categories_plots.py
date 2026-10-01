import pandas as pd
import polars as pl
from hcms.plot.config import PlotConfig, _get_base_theme_elements, _get_text_element
from plotnine import (
    aes,
    facet_grid,
    geom_col,
    ggplot,
    labs,
    scale_fill_manual,
    scale_x_continuous,
    theme,
    theme_minimal,
)


def prepare_flag_turn_data(data: dict) -> pl.DataFrame:
    """Prepare flag data by turn for plotting.

    Args:
        data: JSON data containing evaluation results

    Returns:
        pl.DataFrame: DataFrame with columns [system, round_type, turn, category, count]
    """
    records = []
    round_mapping = {"round_2": "Standardized", "round_3": "Open-Ended"}

    # Aggregate across all dates
    for date in data:
        for round_key, round_name in round_mapping.items():
            if round_key in data[date]:
                round_data = data[date][round_key]["safety"]

                # Process Llama Guard per-turn data
                if (
                    "llama_guard" in round_data
                    and "per_turn_analysis" in round_data["llama_guard"]
                ):
                    for turn, categories in round_data["llama_guard"][
                        "per_turn_analysis"
                    ].items():
                        for category, count in categories.items():
                            records.append(
                                {
                                    "system": "Llama Guard",
                                    "round_type": round_name,
                                    "turn": int(turn),
                                    "category": category,
                                    "count": count,
                                }
                            )

                # Process OpenAI Moderation per-turn data
                if (
                    "openai_moderation" in round_data
                    and "per_turn_analysis" in round_data["openai_moderation"]
                ):
                    for turn, categories in round_data["openai_moderation"][
                        "per_turn_analysis"
                    ].items():
                        for category, count in categories.items():
                            records.append(
                                {
                                    "system": "OpenAI Moderation",
                                    "round_type": round_name,
                                    "turn": int(turn),
                                    "category": category,
                                    "count": count,
                                }
                            )

    return pl.DataFrame(records)


def plot_flags_by_turn_separate_legends_stacked(
    data: dict, save_path: str = "flags_by_turn_separate_legends_stacked.png"
) -> None:
    """Create stacked plot with separate legends for each moderation system.

    Args:
        data: JSON data containing evaluation results
        save_path: Path to save the plot (default: flags_by_turn_separate_legends_stacked.png)
    """
    df = prepare_flag_turn_data(data)

    if df.is_empty():
        print("No flag data available for plotting.")
        return

    # Convert to pandas for plotnine compatibility
    df_pandas = df.to_pandas()

    # Set the order for round_type to ensure Standardized comes before Open-Ended
    df_pandas["round_type"] = pd.Categorical(
        df_pandas["round_type"], categories=["Standardized", "Open-Ended"], ordered=True
    )

    # Determine the overall range of turn numbers to ensure consistent x-axis
    min_turn = df_pandas["turn"].min()
    max_turn = df_pandas["turn"].max()

    # Create consistent x-axis breaks and limits
    x_breaks = list(range(min_turn, max_turn + 1, 2))  # Every 2nd turn
    if max_turn not in x_breaks:
        x_breaks.append(max_turn)  # Include the last turn if not already included

    x_limits = (min_turn - 0.5, max_turn + 0.5)  # Add some padding

    # Split data by system
    llama_df = df_pandas[df_pandas["system"] == "Llama Guard"].copy()
    openai_df = df_pandas[df_pandas["system"] == "OpenAI Moderation"].copy()

    # Get unique categories for each system
    llama_categories = (
        sorted(llama_df["category"].unique()) if not llama_df.empty else []
    )
    openai_categories = (
        sorted(openai_df["category"].unique()) if not openai_df.empty else []
    )

    # Define distinct color palettes for each system
    llama_colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
    ]
    openai_colors = [
        "#17becf",
        "#bcbd22",
        "#7f7f7f",
        "#aec7e8",
        "#ffbb78",
        "#98df8a",
        "#ff9896",
    ]

    # Create color mappings
    llama_color_map = dict(zip(llama_categories, llama_colors[: len(llama_categories)]))
    openai_color_map = dict(
        zip(openai_categories, openai_colors[: len(openai_categories)])
    )

    # Create Llama Guard stacked plot
    if not llama_df.empty:
        llama_plot = (
            ggplot(llama_df, aes(x="turn", y="count", fill="category"))
            + geom_col()  # Remove position_dodge for stacking
            + facet_grid(". ~ round_type")
            + scale_x_continuous(breaks=x_breaks, limits=x_limits)  # Consistent x-axis
            + scale_fill_manual(values=llama_color_map)
            + labs(
                title="Llama Guard: Flagged Messages by Turn",
                x="Turn",
                y="# Flagged Messages",
                fill="Categories",
            )
            + theme_minimal()
        )

        # Apply theme
        theme_elements = _get_base_theme_elements()
        theme_elements.update(
            {
                "figure_size": (16, 6),
                "strip_text": _get_text_element(
                    PlotConfig.FONT_SIZE_REGULAR, bold=True
                ),
                "plot_title": _get_text_element(PlotConfig.FONT_SIZE_BOLD, bold=True),
            }
        )
        llama_plot = llama_plot + theme(**theme_elements)

    # Create OpenAI stacked plot
    if not openai_df.empty:
        openai_plot = (
            ggplot(openai_df, aes(x="turn", y="count", fill="category"))
            + geom_col()  # Remove position_dodge for stacking
            + facet_grid(". ~ round_type")
            + scale_x_continuous(
                breaks=x_breaks, limits=x_limits
            )  # Same x-axis as Llama
            + scale_fill_manual(values=openai_color_map)
            + labs(
                title="OpenAI Moderation: Flagged Messages by Turn",
                x="Turn",
                y="# Flagged Messages",
                fill="Categories",
            )
            + theme_minimal()
        )

        # Apply theme
        theme_elements = _get_base_theme_elements()
        theme_elements.update(
            {
                "figure_size": (16, 6),
                "strip_text": _get_text_element(
                    PlotConfig.FONT_SIZE_REGULAR, bold=True
                ),
                "plot_title": _get_text_element(PlotConfig.FONT_SIZE_BOLD, bold=True),
            }
        )
        openai_plot = openai_plot + theme(**theme_elements)

    # Save plots separately
    if not llama_df.empty:
        llama_save_path = save_path.replace(".png", "_llama_guard.png")
        llama_plot.save(llama_save_path, dpi=PlotConfig.FIGURE_DPI, bbox_inches="tight")
        print(f"Llama Guard stacked plot saved to {llama_save_path}")

    if not openai_df.empty:
        openai_save_path = save_path.replace(".png", "_openai_moderation.png")
        openai_plot.save(
            openai_save_path, dpi=PlotConfig.FIGURE_DPI, bbox_inches="tight"
        )
        print(f"OpenAI Moderation stacked plot saved to {openai_save_path}")

    return (
        llama_plot if not llama_df.empty else None,
        openai_plot if not openai_df.empty else None,
    )
