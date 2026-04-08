"""Stacked bar plots of safety-flagged messages by conversation turn."""

from dataclasses import dataclass
from typing import Optional, Tuple

import math

import pandas as pd
import polars as pl
from plotnine import (
    aes,
    facet_grid,
    geom_col,
    ggplot,
    labs,
    scale_fill_manual,
    scale_x_continuous,
    scale_y_continuous,
    theme,
    theme_minimal,
)

from hpms.plot.config import PlotConfig, _get_base_theme_elements, _get_text_element


# Allowed Llama Guard category names (guards against malformed model output).
LLAMA_GUARD_CATEGORIES = {
    "Child Sexual Exploitation",
    "Defamation",
    "Elections",
    "Hate",
    "Indiscriminate Weapons",
    "Intellectual Property",
    "Non-Violent Crimes",
    "Privacy",
    "Sex-Related Crimes",
    "Sexual Content",
    "Specialized Advice",
    "Suicide & Self-Harm",
    "Violent Crimes",
}

# Fixed category→color mappings using ColorBrewer Paired-12.
#
# Colors are grouped by semantic theme so that similar categories share the same
# color family across both moderation systems:
#   red    → violence / weapons
#   blue   → self-harm / suicide
#   purple → sexual content
#   orange → hate / harassment
#   green  → crime / illicit
#   brown/yellow/gray → informational (advice, IP, privacy, elections, defamation)

LLAMA_GUARD_COLOR_MAP: dict = {
    # --- violence (red family) ---
    "Violent Crimes":            "#E31A1C",  # dark red
    "Indiscriminate Weapons":    "#FB9A99",  # light red
    # --- self-harm (blue family) ---
    "Suicide & Self-Harm":       "#1F78B4",  # dark blue
    # --- sexual (purple family) ---
    "Sexual Content":            "#6A3D9A",  # dark purple
    "Sex-Related Crimes":        "#9E9AC8",  # medium purple
    "Child Sexual Exploitation": "#CAB2D6",  # light purple
    # --- hate (orange family) ---
    "Hate":                      "#FF7F00",  # dark orange
    # --- crime / illicit (green family) ---
    "Non-Violent Crimes":        "#33A02C",  # dark green
    # --- informational (brown / yellow / gray) ---
    "Specialized Advice":        "#B15928",  # brown
    "Intellectual Property":     "#FFFF99",  # light yellow
    "Privacy":                   "#A6CEE3",  # light blue (distinct from self-harm)
    "Defamation":                "#999999",  # gray
    "Elections":                 "#B2DF8A",  # light green
}

# OpenAI categories are title-cased via .title() in process_openai_moderation_scores.
# Each category shares the same color family as its Llama Guard semantic equivalent.
OPENAI_MODERATION_COLOR_MAP: dict = {
    # --- violence (red family) ---
    "Violence":               "#E31A1C",  # dark red  ↔ Violent Crimes
    "Violence/Graphic":       "#FB9A99",  # light red
    # --- self-harm (blue family) ---
    "Self-Harm":              "#1F78B4",  # dark blue  ↔ Suicide & Self-Harm
    "Self-Harm/Intent":       "#A6CEE3",  # light blue
    "Self-Harm/Instructions": "#4292C6",  # medium blue
    # --- sexual (purple family) ---
    "Sexual":                 "#6A3D9A",  # dark purple  ↔ Sexual Content
    "Sexual/Minors":          "#CAB2D6",  # light purple
    # --- hate / harassment (orange family) ---
    "Hate":                   "#FF7F00",  # dark orange  ↔ Hate
    "Hate/Threatening":       "#FDBF6F",  # light orange
    "Harassment":             "#D94701",  # dark orange-red
    "Harassment/Threatening": "#FD8D3C",  # medium orange
    # --- crime / illicit (green family) ---
    "Illicit":                "#33A02C",  # dark green  ↔ Non-Violent Crimes
    "Illicit/Violent":        "#B2DF8A",  # light green
}

_FALLBACK_COLOR = "#AAAAAA"  # light gray for any unknown / future category


def _color_map_for(categories: list, fixed_map: dict) -> dict:
    """Return a color map restricted to *categories*, using fixed_map where known."""
    return {c: fixed_map.get(c, _FALLBACK_COLOR) for c in categories}


@dataclass
class _SystemSpec:
    """Bundles per-system metadata to keep helper function signatures short."""
    system_label: str
    title: str
    fixed_colors: dict
    suffix: str


def _collect_system_records(
    round_data: dict,
    round_name: str,
    system_key: str,
    system_label: str,
    category_allowlist: Optional[set] = None,
) -> list:
    """Extract per-turn flag records for one moderation system from one round."""
    records = []
    per_turn = round_data.get(system_key, {}).get("per_turn_analysis", {})
    for turn, categories in per_turn.items():
        for category, count in categories.items():
            if category_allowlist and category not in category_allowlist:
                continue
            records.append(
                {
                    "system": system_label,
                    "round_type": round_name,
                    "turn": int(turn),
                    "category": category,
                    "count": count,
                }
            )
    return records


def prepare_flag_turn_data(data: dict) -> pl.DataFrame:
    """Prepare flag data by turn for plotting.

    Args:
        data: JSON data containing evaluation results.

    Returns:
        pl.DataFrame with columns [system, round_type, turn, category, count].
    """
    records = []
    round_mapping = {"round_2": "Standardized", "round_3": "Open-Ended"}

    for date in data:
        for round_key, round_name in round_mapping.items():
            if round_key not in data[date]:
                continue
            safety = data[date][round_key]["safety"]
            records.extend(
                _collect_system_records(
                    safety, round_name, "llama_guard", "Llama Guard", LLAMA_GUARD_CATEGORIES
                )
            )
            records.extend(
                _collect_system_records(
                    safety, round_name, "openai_moderation", "OpenAI Moderation"
                )
            )

    return pl.DataFrame(records)


def _integer_breaks(limits: tuple) -> list:
    """Return integer-only axis breaks, keeping tick count reasonable."""
    low = max(0, math.ceil(limits[0]))
    high = math.floor(limits[1])
    span = max(high - low, 1)
    step = max(1, math.ceil(span / 8))
    return list(range(low, high + step, step))


def _build_flag_plot(
    df_system: pd.DataFrame,
    labels: dict,
    color_map: dict,
    x_scale: dict,
    flag_theme: dict,
) -> ggplot:
    """Build one stacked bar plot for a single moderation system."""
    return (
        ggplot(df_system, aes(x="turn", y="count", fill="category"))
        + geom_col()
        + facet_grid(". ~ round_type")
        + scale_x_continuous(**x_scale)
        + scale_y_continuous(breaks=_integer_breaks)
        + scale_fill_manual(values=color_map)
        + labs(**labels)
        + theme_minimal()
        + theme(**flag_theme)
    )


def _build_flag_theme() -> dict:
    """Return the ACM-style theme dict for flag plots."""
    acm_text = _get_text_element(PlotConfig.ACM_FONT_SIZE)
    acm_text_title = _get_text_element(PlotConfig.ACM_FONT_SIZE_TITLE, bold=True)
    return {
        **_get_base_theme_elements(),
        "figure_size": (PlotConfig.ACM_TEXT_WIDTH, 3.2),
        "axis_title": acm_text_title,
        "axis_text": acm_text,
        "legend_title": acm_text_title,
        "legend_text": acm_text,
        "strip_text": acm_text_title,
        "plot_title": _get_text_element(PlotConfig.ACM_FONT_SIZE_HEADING, bold=True),
        "plot_subtitle": _get_text_element(PlotConfig.ACM_FONT_SIZE_TITLE),
    }


def _save_system_plot(
    df_pandas: pd.DataFrame,
    spec: _SystemSpec,
    subtitle: str,
    x_scale: dict,
    flag_theme: dict,
    save_path: str,
) -> Optional[ggplot]:
    """Build, save, and return a flag plot for one moderation system."""
    df_sys = df_pandas[df_pandas["system"] == spec.system_label]
    if df_sys.empty:
        return None
    cats = sorted(df_sys["category"].unique())
    color_map = _color_map_for(cats, spec.fixed_colors)
    labels = {"title": spec.title, "subtitle": subtitle,
              "x": "Turn", "y": "# Flagged Messages", "fill": "Category"}
    plot = _build_flag_plot(df_sys, labels, color_map, x_scale, flag_theme)
    ext = save_path.rsplit(".", 1)
    out = f"{ext[0]}_{spec.suffix}.{ext[1]}"
    plot.save(out, dpi=PlotConfig.FIGURE_DPI, bbox_inches="tight")
    print(f"{spec.title} plot saved to {out}")
    return plot


def plot_flags_by_turn_separate_legends_stacked(
    data: dict,
    save_path: str = "flags_by_turn_separate_legends_stacked.png",
) -> Tuple[Optional[ggplot], Optional[ggplot]]:
    """Create stacked bar plots with separate legends for each moderation system.

    Args:
        data: JSON data containing evaluation results.
        save_path: Base path for saved plots; ``_llama_guard`` / ``_openai_moderation``
            suffixes are appended automatically.

    Returns:
        Tuple of (llama_plot, openai_plot); either may be None if no data.
    """
    df = prepare_flag_turn_data(data)

    if df.is_empty():
        print("No flag data available for plotting.")
        return None, None

    df_pandas = df.to_pandas()
    df_pandas["round_type"] = pd.Categorical(
        df_pandas["round_type"], categories=["Standardized", "Open-Ended"], ordered=True
    )

    sorted_dates = sorted(data.keys())
    period = (
        sorted_dates[0]
        if len(sorted_dates) == 1
        else f"{sorted_dates[0]} \u2013 {sorted_dates[-1]}"
    )

    min_turn, max_turn = df_pandas["turn"].min(), df_pandas["turn"].max()
    x_breaks = list(range(min_turn, max_turn + 1, 2))
    if max_turn not in x_breaks:
        x_breaks.append(max_turn)
    x_scale = {"breaks": x_breaks, "limits": (min_turn - 0.5, max_turn + 0.5)}
    flag_theme = _build_flag_theme()

    llama_spec = _SystemSpec(
        "Llama Guard", "Llama Guard: Flagged Messages by Turn",
        LLAMA_GUARD_COLOR_MAP, "llama_guard",
    )
    openai_spec = _SystemSpec(
        "OpenAI Moderation", "OpenAI Moderation: Flagged Messages by Turn",
        OPENAI_MODERATION_COLOR_MAP, "openai_moderation",
    )
    llama_plot = _save_system_plot(df_pandas, llama_spec, period, x_scale, flag_theme, save_path)
    openai_plot = _save_system_plot(df_pandas, openai_spec, period, x_scale, flag_theme, save_path)
    return llama_plot, openai_plot
