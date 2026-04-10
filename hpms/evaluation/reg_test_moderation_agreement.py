"""Cross-system moderation agreement analysis: Llama Guard vs OpenAI Moderation."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import polars as pl
from plotnine import (
    aes,
    element_text,
    facet_grid,
    geom_col,
    geom_line,
    geom_point,
    geom_text,
    geom_tile,
    ggplot,
    labs,
    scale_color_manual,
    scale_fill_gradient,
    scale_fill_manual,
    scale_x_datetime,
    scale_y_continuous,
    theme,
    theme_minimal,
)

from hpms.evaluation.reg_test_categories_plots import LLAMA_GUARD_CATEGORIES
from hpms.plot.config import PlotConfig, _get_base_theme_elements, _get_text_element


# Display labels and colors for the four overlap categories
_OVERLAP_LABELS = {
    "Both":     "Both (LG ∩ OAI)",
    "LG only":  "LG only",
    "OAI only": "OAI only",
    "Neither":  "Neither",
}
_OVERLAP_COLORS = {
    "Both":     "#E31A1C",  # red   – flagged by both, highest concern
    "LG only":  "#1F78B4",  # blue  – LG-exclusive
    "OAI only": "#FF7F00",  # orange – OAI-exclusive
    "Neither":  "#CCCCCC",  # gray  – clean
}
_OVERLAP_ORDER = ["Both", "LG only", "OAI only", "Neither"]


def compute_moderation_agreement(lg_file: Path, oai_file: Path) -> Dict:
    """Compute per-message overlap between Llama Guard and OpenAI Moderation.

    Args:
        lg_file: Path to the rated Llama Guard JSON file.
        oai_file: Path to the rated OpenAI Moderation JSON file.

    Returns:
        Dict with keys:
          total_messages, both_flagged, lg_only, oai_only, neither_flagged,
          both_rate, lg_only_rate, oai_only_rate, neither_rate,
          jaccard, cohen_kappa.
    """
    lg_df = pl.read_json(lg_file).with_row_index()
    oai_df = pl.read_json(oai_file).with_row_index()

    # Llama Guard: flagged when score is a known valid category
    lg_flagged = lg_df.with_columns(
        pl.col("llama_guard_score")
        .is_in(list(LLAMA_GUARD_CATEGORIES))
        .alias("lg_flagged")
    ).select(["index", "lg_flagged"])

    # OpenAI Moderation: flagged when any element in the array is not "0"
    oai_flagged = oai_df.with_columns(
        pl.col("OpenAI_Moderation")
        .list.eval(pl.element() != "0")
        .list.any()
        .alias("oai_flagged")
    ).select(["index", "oai_flagged"])

    combined = lg_flagged.join(oai_flagged, on="index")
    n = len(combined)

    both    = combined.filter( pl.col("lg_flagged") &  pl.col("oai_flagged")).shape[0]
    lg_only = combined.filter( pl.col("lg_flagged") & ~pl.col("oai_flagged")).shape[0]
    oai_only= combined.filter(~pl.col("lg_flagged") &  pl.col("oai_flagged")).shape[0]
    neither = combined.filter(~pl.col("lg_flagged") & ~pl.col("oai_flagged")).shape[0]

    union   = both + lg_only + oai_only
    jaccard = both / union if union > 0 else 0.0

    # Cohen's κ
    p_o   = (both + neither) / n
    p_lg  = (both + lg_only) / n
    p_oai = (both + oai_only) / n
    p_e   = p_lg * p_oai + (1 - p_lg) * (1 - p_oai)
    kappa = (p_o - p_e) / (1 - p_e) if p_e < 1 else 1.0

    return {
        "moderation_agreement": {
            "total_messages":  n,
            "both_flagged":    both,
            "lg_only":         lg_only,
            "oai_only":        oai_only,
            "neither_flagged": neither,
            "both_rate":       both    / n,
            "lg_only_rate":    lg_only / n,
            "oai_only_rate":   oai_only / n,
            "neither_rate":    neither / n,
            "jaccard":         jaccard,
            "cohen_kappa":     kappa,
        }
    }


def _build_agreement_records(data: dict) -> list:
    """Extract per-date/round overlap counts into a flat list of records."""
    records = []
    round_mapping = {"round_2": "Standardized", "round_3": "Open-Ended"}
    for date in sorted(data.keys()):
        for round_key, round_name in round_mapping.items():
            if round_key not in data[date]:
                continue
            agr = data[date][round_key].get("safety", {}).get("moderation_agreement")
            if agr is None:
                continue
            n = agr["total_messages"]
            for label in _OVERLAP_ORDER:
                key_map = {
                    "Both":    "both_flagged",
                    "LG only": "lg_only",
                    "OAI only":"oai_only",
                    "Neither": "neither_flagged",
                }
                records.append({
                    "date":       date,
                    "round_type": round_name,
                    "overlap":    label,
                    "ratio":      agr[key_map[label]] / n if n > 0 else 0.0,
                    "jaccard":    round(agr["jaccard"], 3),
                    "kappa":      round(agr["cohen_kappa"], 3),
                })
    return records


def _agreement_theme() -> dict:
    """Build the ACM theme dict for agreement plots."""
    acm_text       = _get_text_element(PlotConfig.ACM_FONT_SIZE)
    acm_text_title = _get_text_element(PlotConfig.ACM_FONT_SIZE_TITLE, bold=True)
    return {
        **_get_base_theme_elements(),
        "figure_size":    (PlotConfig.ACM_TEXT_WIDTH, 3.5),
        "axis_title":     acm_text_title,
        "axis_text":      acm_text,
        "legend_title":   acm_text_title,
        "legend_text":    acm_text,
        "strip_text":     acm_text_title,
        "plot_title":     _get_text_element(PlotConfig.ACM_FONT_SIZE_HEADING, bold=True),
        "plot_subtitle":  acm_text_title,
    }


def _prep_df(data: dict) -> Optional[pd.DataFrame]:
    """Return a prepared DataFrame from agreement records, or None if empty."""
    records = _build_agreement_records(data)
    if not records:
        return None
    df = pd.DataFrame(records)
    sorted_dates = sorted(df["date"].unique())
    df["date"] = pd.Categorical(df["date"], categories=sorted_dates, ordered=True)
    df["round_type"] = pd.Categorical(
        df["round_type"], categories=["Standardized", "Open-Ended"], ordered=True
    )
    df["overlap"] = pd.Categorical(
        df["overlap"], categories=_OVERLAP_ORDER, ordered=True
    )
    return df


def _period_str(df: pd.DataFrame) -> str:
    dates = sorted(df["date"].cat.categories)
    return dates[0] if len(dates) == 1 else f"{dates[0]} \u2013 {dates[-1]}"


def plot_overlap_breakdown(
    data: dict,
    save_path: str = "agreement_overlap.pdf",
) -> Optional[ggplot]:
    """100%-stacked ratio bar: Co-flag / LG-only / OAI-only / Neither per date."""
    df = _prep_df(data)
    if df is None:
        print("No moderation agreement data for overlap breakdown plot.")
        return None

    thm = _agreement_theme()
    plot = (
        ggplot(df, aes(x="date", y="ratio", fill="overlap"))
        + geom_col()
        + facet_grid(". ~ round_type")
        + scale_fill_manual(values=_OVERLAP_COLORS, labels=_OVERLAP_LABELS)
        + scale_y_continuous(limits=(0, 1), breaks=[0, 0.25, 0.5, 0.75, 1.0],
                             labels=["0", "0.25", "0.5", "0.75", "1"])
        + labs(title="Moderation Overlap Breakdown",
               subtitle=_period_str(df),
               x="Date", y="Ratio of Messages", fill="Overlap")
        + theme_minimal()
        + theme(**thm)
    )
    plot.save(save_path, dpi=PlotConfig.FIGURE_DPI, bbox_inches="tight")
    print(f"Overlap breakdown plot saved to {save_path}")
    return plot


def _build_metric_series(data: dict) -> pd.DataFrame:
    """Return a long DataFrame with jaccard and cohen_kappa per date/round."""
    rows = []
    round_mapping = {"round_2": "Standardized", "round_3": "Open-Ended"}
    for date in sorted(data.keys()):
        for round_key, round_name in round_mapping.items():
            if round_key not in data[date]:
                continue
            agr = data[date][round_key].get("safety", {}).get("moderation_agreement")
            if agr is None:
                continue
            rows.append({
                "date":       pd.to_datetime(date),
                "round_type": round_name,
                "Jaccard":    agr["jaccard"],
                "Cohen's κ":  agr["cohen_kappa"],
            })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["round_type"] = pd.Categorical(
        df["round_type"], categories=["Standardized", "Open-Ended"], ordered=True
    )
    return df


def plot_jaccard_over_time(
    data: dict,
    save_path: str = "agreement_jaccard.pdf",
) -> Optional[ggplot]:
    """Line plot of Jaccard similarity over time, one line per round type."""
    df = _build_metric_series(data)
    if df.empty:
        print("No agreement data for Jaccard plot.")
        return None

    colors = {"Standardized": "#2E86AB", "Open-Ended": "#A23B72"}
    thm = _agreement_theme()
    thm["axis_text_x"] = element_text(
        size=PlotConfig.ACM_FONT_SIZE, angle=45, ha="right"
    )
    period = f"{df['date'].dt.strftime('%Y-%m-%d').min()} \u2013 {df['date'].dt.strftime('%Y-%m-%d').max()}"

    plot = (
        ggplot(df, aes(x="date", y="Jaccard", color="round_type"))
        + geom_line(size=1.2)
        + geom_point(size=3)
        + scale_color_manual(values=colors, name="Prompt Type")
        + scale_x_datetime(date_labels="%b %d", date_breaks="1 day")
        + scale_y_continuous(limits=(0, 1), breaks=[0, 0.25, 0.5, 0.75, 1.0])
        + labs(title="Jaccard Similarity Over Time",
               subtitle=period, x="Date", y="Jaccard Similarity")
        + theme_minimal()
        + theme(**thm)
    )
    plot.save(save_path, dpi=PlotConfig.FIGURE_DPI, bbox_inches="tight")
    print(f"Jaccard plot saved to {save_path}")
    return plot


def plot_kappa_over_time(
    data: dict,
    save_path: str = "agreement_kappa.pdf",
) -> Optional[ggplot]:
    """Line plot of Cohen's κ over time, one line per round type."""
    df = _build_metric_series(data)
    if df.empty:
        print("No agreement data for κ plot.")
        return None

    colors = {"Standardized": "#2E86AB", "Open-Ended": "#A23B72"}
    thm = _agreement_theme()
    thm["axis_text_x"] = element_text(
        size=PlotConfig.ACM_FONT_SIZE, angle=45, ha="right"
    )
    period = f"{df['date'].dt.strftime('%Y-%m-%d').min()} \u2013 {df['date'].dt.strftime('%Y-%m-%d').max()}"

    plot = (
        ggplot(df, aes(x="date", y="Cohen's κ", color="round_type"))
        + geom_line(size=1.2)
        + geom_point(size=3)
        + scale_color_manual(values=colors, name="Prompt Type")
        + scale_x_datetime(date_labels="%b %d", date_breaks="1 day")
        + scale_y_continuous(limits=(-0.05, 1.0))
        + labs(title="Cohen's κ Over Time",
               subtitle=period, x="Date", y="Cohen's κ")
        + theme_minimal()
        + theme(**thm)
    )
    plot.save(save_path, dpi=PlotConfig.FIGURE_DPI, bbox_inches="tight")
    print(f"Cohen's κ plot saved to {save_path}")
    return plot


def plot_confusion_heatmap(
    data: dict,
    save_path: str = "agreement_confusion.pdf",
) -> Optional[ggplot]:
    """Confusion matrix heatmap (LG flagged × OAI flagged) aggregated over all dates."""
    rows = []
    round_mapping = {"round_2": "Standardized", "round_3": "Open-Ended"}
    for date in sorted(data.keys()):
        for round_key, round_name in round_mapping.items():
            if round_key not in data[date]:
                continue
            agr = data[date][round_key].get("safety", {}).get("moderation_agreement")
            if agr is None:
                continue
            n = agr["total_messages"]
            rows += [
                {"round_type": round_name, "LG": "Flagged",     "OAI": "Flagged",
                 "n": agr["both_flagged"],    "pct": agr["both_flagged"]    / n},
                {"round_type": round_name, "LG": "Flagged",     "OAI": "Not flagged",
                 "n": agr["lg_only"],         "pct": agr["lg_only"]         / n},
                {"round_type": round_name, "LG": "Not flagged", "OAI": "Flagged",
                 "n": agr["oai_only"],        "pct": agr["oai_only"]        / n},
                {"round_type": round_name, "LG": "Not flagged", "OAI": "Not flagged",
                 "n": agr["neither_flagged"], "pct": agr["neither_flagged"] / n},
            ]

    if not rows:
        print("No agreement data for confusion heatmap.")
        return None

    df = pd.DataFrame(rows)
    # Aggregate across dates
    df = df.groupby(["round_type", "LG", "OAI"], as_index=False)[["n", "pct"]].sum()
    # Re-normalise pct per round after summing
    totals = df.groupby("round_type")["n"].transform("sum")
    df["pct"] = df["n"] / totals
    df["label"] = df.apply(lambda r: f"{r['n']}\n({r['pct']:.1%})", axis=1)

    df["round_type"] = pd.Categorical(
        df["round_type"], categories=["Standardized", "Open-Ended"], ordered=True
    )
    df["LG"]  = pd.Categorical(df["LG"],  categories=["Flagged", "Not flagged"], ordered=True)
    df["OAI"] = pd.Categorical(df["OAI"], categories=["Flagged", "Not flagged"], ordered=True)

    thm = _agreement_theme()
    thm["figure_size"] = (PlotConfig.ACM_TEXT_WIDTH, 3.0)

    period_dates = sorted(data.keys())
    period = period_dates[0] if len(period_dates) == 1 else f"{period_dates[0]} \u2013 {period_dates[-1]}"

    plot = (
        ggplot(df, aes(x="OAI", y="LG", fill="pct"))
        + geom_tile(color="white", size=0.5)
        + geom_text(aes(label="label"), size=PlotConfig.ACM_FONT_SIZE, color="black")
        + facet_grid(". ~ round_type")
        + scale_fill_gradient(low="#FFFFFF", high="#E31A1C", limits=(0, 1), name="Ratio")
        + labs(title="Confusion Matrix: LG vs OAI Moderation",
               subtitle=period, x="OpenAI Moderation", y="Llama Guard")
        + theme_minimal()
        + theme(**thm)
    )
    plot.save(save_path, dpi=PlotConfig.FIGURE_DPI, bbox_inches="tight")
    print(f"Confusion heatmap saved to {save_path}")
    return plot


def plot_moderation_agreement(
    data: dict,
    save_path: str = "moderation_agreement.pdf",
) -> List[Optional[ggplot]]:
    """Run all four agreement plots and return them as a list.

    Args:
        data: JSON evaluation results.
        save_path: Base path; plot-specific suffixes are appended before the extension.

    Returns:
        [overlap_plot, jaccard_plot, kappa_plot, confusion_plot]
    """
    ext = save_path.rsplit(".", 1)

    def _p(suffix: str) -> str:
        return f"{ext[0]}_{suffix}.{ext[1]}"

    return [
        plot_overlap_breakdown(data,      save_path=_p("overlap")),
        plot_jaccard_over_time(data,      save_path=_p("jaccard")),
        plot_kappa_over_time(data,        save_path=_p("kappa")),
        plot_confusion_heatmap(data,      save_path=_p("confusion")),
    ]


def print_agreement_summary(data: dict) -> None:
    """Print a text summary table of agreement metrics across all dates and rounds."""
    records = _build_agreement_records(data)
    if not records:
        print("No moderation agreement data found.")
        return

    df = pd.DataFrame(records).drop_duplicates(["date", "round_type"])
    print("\n── Moderation Agreement Summary ──────────────────────────────")
    print(f"{'Date':<12} {'Round':<14} {'Both':>5} {'LG-only':>8} {'OAI-only':>9} {'Neither':>8} {'Jaccard':>8} {'κ':>7}")
    print("-" * 72)
    for _, row in df.sort_values(["date", "round_type"]).iterrows():
        agr = data[row["date"]][
            "round_2" if row["round_type"] == "Standardized" else "round_3"
        ]["safety"]["moderation_agreement"]
        n = agr["total_messages"]
        print(
            f"{row['date']:<12} {row['round_type']:<14}"
            f" {agr['both_flagged']:>5} ({agr['both_rate']*100:4.1f}%)"
            f" {agr['lg_only']:>5} ({agr['lg_only_rate']*100:4.1f}%)"
            f" {agr['oai_only']:>5} ({agr['oai_only_rate']*100:4.1f}%)"
            f" {agr['neither_flagged']:>5} ({agr['neither_rate']*100:4.1f}%)"
            f"  {agr['jaccard']:.3f}  {agr['cohen_kappa']:+.3f}"
        )
    print()
