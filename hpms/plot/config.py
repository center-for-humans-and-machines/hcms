"""Configuration settings for plots."""

from pathlib import Path
from typing import Dict

import matplotlib as _mpl
import matplotlib.font_manager as _fm
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

    # ACM TIST figure dimensions (inches) and font sizes
    # Single column: 3.33", double column (text width): 6.97"
    ACM_COLUMN_WIDTH = 3.33
    ACM_TEXT_WIDTH = 6.97
    ACM_FONT_SIZE = 8         # axis labels / legend
    ACM_FONT_SIZE_TITLE = 9   # axis titles / strip labels
    ACM_FONT_SIZE_HEADING = 11 # plot-level title


def _register_linux_libertine() -> None:
    """Explicitly register Linux Libertine TTF files with matplotlib.

    macOS doesn't expose ~/Library/Fonts to matplotlib's default scan, so we
    register every matching file by path.
    """
    search_dirs = [
        Path.home() / "Library" / "Fonts",
        Path("/Library/Fonts"),
        Path("/usr/share/fonts"),
    ]
    for d in search_dirs:
        for ttf in d.glob("LinLibertine*.ttf"):
            _fm.fontManager.addfont(str(ttf))


def _resolve_font() -> str:
    """Return FONT_FAMILY if available after registration, else a safe fallback."""
    _register_linux_libertine()
    available = {f.name for f in _fm.fontManager.ttflist}
    if PlotConfig.FONT_FAMILY in available:
        return PlotConfig.FONT_FAMILY
    return "DejaVu Serif"


# Set matplotlib rcParams globally so plotnine elements that don't go through
# _get_text_element() also use the correct font.
_RESOLVED = _resolve_font()
_mpl.rcParams["font.family"] = "serif"
_mpl.rcParams["font.serif"] = [_RESOLVED, "DejaVu Serif"]


def _get_text_element(size: int, bold: bool = False) -> element_text:
    """Create standardized text element with consistent font settings."""
    weight = "bold" if bold else "normal"
    return element_text(size=size, weight=weight, fontfamily=_RESOLVED)


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
