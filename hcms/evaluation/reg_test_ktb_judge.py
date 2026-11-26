"""Daily regression statistics from Kill The Bot (KTB) judge ratings."""

from pathlib import Path
from typing import Any, Dict

import polars as pl


def process_ktb_judge_scores(file_path: Path) -> Dict[str, Any]:
    """Process Kill The Bot (KTB) judge verdicts from a rated-ktb-judge file.

    Replies without a verdict (``ktb_unsafe`` null, ``ktb_error`` set) count
    as failed and are left out of ``unsafe_share``.

    Args:
        file_path (Path): Path to a ``rated-ktb-judge-*.json`` file.

    Returns:
        Dict[str, Any]: Under ``ktb_judge``: total_replies, rated_replies,
            failed_replies, unsafe_replies, unsafe_share (of rated replies),
            categories_count and per_turn_analysis ({turn: unsafe count}).
    """
    df = pl.read_json(file_path.expanduser())
    rated = df.filter(pl.col("ktb_unsafe").is_not_null())
    unsafe = rated.filter(pl.col("ktb_unsafe"))

    categories_df = (
        unsafe.select(pl.col("ktb_categories").explode())
        .drop_nulls()
        .group_by("ktb_categories")
        .agg(pl.len().alias("count"))
        .sort("ktb_categories")
    )
    per_turn_df = unsafe.group_by("turn").agg(pl.len().alias("count")).sort("turn")

    return {
        "ktb_judge": {
            "total_replies": df.height,
            "rated_replies": rated.height,
            "failed_replies": df.height - rated.height,
            "unsafe_replies": unsafe.height,
            "unsafe_share": unsafe.height / rated.height if rated.height else None,
            "categories_count": dict(
                zip(categories_df["ktb_categories"], categories_df["count"])
            ),
            "per_turn_analysis": dict(zip(per_turn_df["turn"], per_turn_df["count"])),
        }
    }
