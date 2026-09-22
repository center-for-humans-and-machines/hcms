"""Configuration settings for plots."""

from typing import Dict

from plotnine import (
    element_blank,
    element_line,
    element_text,
)


# pylint: disable-next=too-few-public-methods
class PlotConfig:
    """Configuration settings for plots."""

    # Font settings
    FONT_FAMILY = "Linux Libertine"
    FONT_SIZE_BOLD = 24
    FONT_SIZE_REGULAR = 20

    # Color settings
    PRIMARY_COLOR = "#1f77b4"
    SECONDARY_COLOR = "#ff7f0e"

    # Figure settings
    FIGURE_DPI = 300


def _get_text_element(size: int, bold: bool = False) -> element_text:
    """Create standardized text element with consistent font settings."""
    weight = "bold" if bold else "normal"
    return element_text(size=size, weight=weight, fontfamily=PlotConfig.FONT_FAMILY)


def _get_base_theme_elements() -> Dict:
    """Get common theme elements used across all plots."""
    return {
        "axis_title": _get_text_element(PlotConfig.FONT_SIZE_BOLD, bold=True),
        "axis_text": _get_text_element(PlotConfig.FONT_SIZE_REGULAR),
        "legend_title": _get_text_element(PlotConfig.FONT_SIZE_BOLD, bold=True),
        "legend_text": _get_text_element(PlotConfig.FONT_SIZE_REGULAR),
        "legend_position": "bottom",
        "panel_grid_major": element_line(alpha=0.3),
        "panel_grid_minor": element_line(alpha=0.1),
        "panel_background": element_blank(),
        "plot_background": element_blank(),
        "axis_line_x": element_line(color="gray", size=0.5),
        "axis_line_y": element_line(color="gray", size=0.5),
    }
