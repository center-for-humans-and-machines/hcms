"""Aggregate and visualize diversity and safety statistics from regression test pipeline
over multiple days."""

import ast
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import polars as pl
from hcms.evaluation.reg_test_categories_plots import (
    plot_flags_by_turn_separate_legends_stacked,
)
from hcms.evaluation.reg_test_categories_table import (
    generate_flag_categories_latex_table,
)
from hcms.evaluation.reg_test_table import generate_latex_table, load_json_data
from hcms.evaluation.self_bleu import calculate_dataset_self_bleu, calculate_vocab_size
from hcms.evaluation.unique_n_grams import percent_unique_ngrams
from hcms.evaluation.zipf import calculate_zipf_coefficient
from hcms.plot.config import PlotConfig, _get_base_theme_elements, _get_text_element
from IPython.display import display
from plotnine import (
    aes,
    facet_wrap,
    geom_line,
    geom_point,
    ggplot,
    labs,
    scale_color_manual,
    scale_x_datetime,
    theme,
    theme_minimal,
)


def filter_assistant_messages(df: pl.DataFrame) -> pl.DataFrame:
    """Filter conversation column to only include assistant messages.

    Args:
        df: DataFrame with conversation column containing list of messages

    Returns:
        DataFrame with filtered conversations containing only assistant messages
    """
    return df.with_columns(
        pl.col("conversation").map_elements(
            lambda conv: [msg for msg in conv if msg["role"] == "assistant"],
            return_dtype=pl.List(pl.Struct({"role": pl.String, "content": pl.String})),
        )
    )


def count_unique_messages(conversations: List[List[Dict]]) -> int:
    """Count unique messages in conversation content field.

    Args:
        conversations: List of conversations, each containing message dicts

    Returns:
        Number of unique message contents
    """
    unique_contents = set()
    for conv in conversations:
        for msg in conv:
            unique_contents.add(msg["content"])
    return len(unique_contents)


def process_llama_guard_scores(file_path: Path) -> Dict[str, Any]:
    """Process llama guard scores from a JSON file.

    Args:
        file_path (Path): Path to the JSON file containing llama guard data.

    Returns:
        Dict[str, Any]: Dictionary containing:
            - total_messages: Total number of messages
            - flagged_messages: Number of flagged messages (score != "0")
            - categories_count: Dict of category names and their flag counts
            - per_turn_analysis: Dict with turn numbers as keys and flag counts as values
    """
    # Read the JSON file
    df = pl.read_json(file_path.expanduser())

    # Total messages
    total_messages = df.shape[0]

    # Flagged messages (score != "0")
    flagged_messages = df.filter(pl.col("llama_guard_score") != "0").shape[0]

    # Categories and their counts
    categories_df = (
        df.filter(pl.col("llama_guard_score") != "0")
        .group_by("llama_guard_score")
        .agg(pl.len().alias("count"))
        .sort("llama_guard_score")
    )
    categories_count = dict(
        zip(categories_df["llama_guard_score"], categories_df["count"])
    )

    # Per turn analysis
    per_turn_df = (
        df.filter(pl.col("llama_guard_score") != "0")
        .group_by(["turn", "llama_guard_score"])
        .agg(pl.len().alias("count"))
        .sort(["turn", "llama_guard_score"])
    )

    # Convert to nested dict: {turn: {category: count}}
    per_turn_analysis = {}
    for row in per_turn_df.iter_rows(named=True):
        turn = row["turn"]
        category = row["llama_guard_score"]
        count = row["count"]

        if turn not in per_turn_analysis:
            per_turn_analysis[turn] = {}
        per_turn_analysis[turn][category] = count

    return {
        "llama_guard": {
            "total_messages": total_messages,
            "flagged_messages": flagged_messages,
            "categories_count": categories_count,
            "per_turn_analysis": per_turn_analysis,
        }
    }


def process_llm_judge_scores(file_path: Path) -> Dict[str, Any]:
    """Process LLM judge scores from a JSON file.

    Args:
        file_path (Path): Path to the JSON file containing LLM judge data.

    Returns:
        Dict[str, Any]: Dictionary containing:
            - total_ratings: Total number of ratings
            - mean_rating: Mean rating value
            - std_rating: Standard deviation of ratings
            - ratings: List of all individual rating values
            - per_turn_analysis: Dict with turn numbers as keys and rating lists as values
    """
    # Read the file as text and parse with ast.literal_eval
    with open(file_path.expanduser(), "r") as f:
        data = ast.literal_eval(f.read())

    # Convert to polars DataFrame
    df = pl.DataFrame(data)

    # Extract ratings from the nested JSON structure
    ratings = []
    ratings_with_turn = []  # Store (turn, rating) pairs

    # Turn mapping for 5-turn conversations: (1, 3, 5, 7, 9)
    turn_mapping = [1, 3, 5, 7, 9]

    print(file_path)
    for idx, row in enumerate(df.iter_rows(named=True)):
        try:
            # Navigate to the content field
            content = row["response"]["body"]["choices"][0]["message"]["content"]
            # Parse the JSON content to extract rating
            content_json = json.loads(content)
            rating = content_json.get("rating")

            if rating is not None:
                ratings.append(rating)

                # Calculate turn: each conversation has 5 elements (turns 1,3,5,7,9)
                # idx % 5 gives position within conversation (0,1,2,3,4)
                # Map to turns (1,3,5,7,9)
                turn = turn_mapping[idx % 5]
                ratings_with_turn.append((turn, rating))

        except (KeyError, TypeError, json.JSONDecodeError):
            # Skip rows with missing or malformed data
            continue

    # Convert to polars DataFrame for statistics
    ratings_df = pl.DataFrame({"rating": ratings})

    # Calculate statistics
    total_ratings = len(ratings)

    if total_ratings > 0:
        stats = ratings_df.select(
            [
                pl.col("rating").mean().alias("mean_rating"),
                pl.col("rating").std().alias("std_rating"),
            ]
        )
        mean_rating = stats["mean_rating"][0]
        std_rating = (
            stats["std_rating"][0] if stats["std_rating"][0] is not None else 0.0
        )
    else:
        mean_rating = 0.0
        std_rating = 0.0

    # Create per-turn analysis
    per_turn_analysis = {}
    for turn, rating in ratings_with_turn:
        if turn not in per_turn_analysis:
            per_turn_analysis[turn] = []
        per_turn_analysis[turn].append(rating)

    # Debug information to verify expected structure
    print(f"Total ratings processed: {total_ratings}")
    print(
        f"Per-turn breakdown: {[(turn, len(ratings)) for turn, ratings in sorted(per_turn_analysis.items())]}"
    )

    # Verify we have the expected structure: 500 total ratings, 100 per turn
    expected_turns = [1, 3, 5, 7, 9]
    if set(per_turn_analysis.keys()) == set(expected_turns):
        print("✅ Found expected turns: 1, 3, 5, 7, 9")
        if total_ratings == 500:
            print("✅ Found expected total: 500 ratings")
        else:
            print(f"⚠️  Found {total_ratings} ratings, expected 500")

        # Check if each turn has 100 ratings
        all_turns_correct = all(
            len(per_turn_analysis[turn]) == 100 for turn in expected_turns
        )
        if all_turns_correct:
            print("✅ Each turn has 100 ratings as expected")
        else:
            print("⚠️  Turn distribution not as expected (should be 100 per turn)")
    else:
        print(
            f"⚠️  Found turns: {sorted(per_turn_analysis.keys())}, expected: {expected_turns}"
        )

    return {
        "llm_judge": {
            "total_ratings": total_ratings,
            "mean_rating": mean_rating,
            "std_rating": std_rating,
            "ratings": ratings,  # Add the individual ratings list
            "per_turn_analysis": per_turn_analysis,  # Add per-turn ratings
        }
    }


def process_openai_moderation_scores(file_path: Path) -> Dict[str, Any]:
    """Process OpenAI moderation scores from a JSON file.

    Args:
        file_path (Path): Path to the JSON file containing OpenAI moderation data.

    Returns:
        Dict[str, Any]: Dictionary containing:
            - total_messages: Total number of messages
            - flagged_messages: Number of messages with any flags
            - categories_count: Dict of normalized category names and their flag counts
            - per_turn_analysis: Dict with turn numbers as keys and flag counts as values
    """
    # Category mapping to merge underscore/slash variants
    category_mapping = {
        "harassment_threatening": "harassment/threatening",
        "hate_threatening": "hate/threatening",
        "illicit_violent": "illicit/violent",
        "self_harm_instructions": "self-harm/instructions",
        "self_harm_intent": "self-harm/intent",
        "self_harm": "self-harm",
        "sexual_minors": "sexual/minors",
        "violence_graphic": "violence/graphic",
    }

    def normalize_category(category: str) -> str:
        """Normalize category names by merging underscore/slash variants."""
        normalized = category_mapping.get(category, category)
        return normalized.title()

    # Read the JSON file
    df = pl.read_json(file_path.expanduser())

    # Total messages
    total_messages = df.shape[0]

    # Explode the OpenAI_Moderation array, filter out "0" values, normalize, and deduplicate per message
    moderation_df = (
        df.with_row_index()
        .select(["index", "turn", "OpenAI_Moderation"])
        .explode("OpenAI_Moderation")
        .filter(pl.col("OpenAI_Moderation") != "0")
        .with_columns(
            pl.col("OpenAI_Moderation")
            .map_elements(normalize_category, return_dtype=pl.Utf8)
            .alias("category")
        )
        .unique(["index", "category"])  # Remove duplicate categories per message
    )

    # Count unique messages with flags
    flagged_messages = moderation_df.select("index").unique().shape[0]

    # Categories and their counts (normalized)
    categories_df = (
        moderation_df.group_by("category").agg(pl.len().alias("count")).sort("category")
    )
    categories_count = dict(zip(categories_df["category"], categories_df["count"]))

    # Per turn analysis
    per_turn_df = (
        moderation_df.group_by(["turn", "category"])
        .agg(pl.len().alias("count"))
        .sort(["turn", "category"])
    )

    # Convert to nested dict: {turn: {category: count}}
    per_turn_analysis = {}
    for row in per_turn_df.iter_rows(named=True):
        turn = row["turn"]
        category = row["category"]
        count = row["count"]

        if turn not in per_turn_analysis:
            per_turn_analysis[turn] = {}
        per_turn_analysis[turn][category] = count

    return {
        "openai_moderation": {
            "total_messages": total_messages,
            "flagged_messages": flagged_messages,
            "categories_count": categories_count,
            "per_turn_analysis": per_turn_analysis,
        }
    }


def calculate_comprehensive_statistics(
    data_dir: Path,
    ratings_dir: Path,
    output_file: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """Calculate diversity and safety statistics for each day (with caching)."""
    results: Dict[str, Dict[str, Any]] = defaultdict(dict)

    # find all date folders
    date_folders = [
        d for d in ratings_dir.iterdir() if d.is_dir() and d.name.startswith("2025-08-")
    ]
    date_folders.sort()

    for date_folder in date_folders:
        date = date_folder.name
        print(f"Processing {date}...")

        for round_num in [2, 3]:
            round_key = f"round_{round_num}"
            print(f"  Processing round {round_num}...")

            # declare inputs
            dataset_file = (
                data_dir
                / date
                / f"dataset-round-{round_num}-gpt-4o-mini-2024-07-18.json"
            )
            llama_file = (
                date_folder
                / f"rated-llama-guard-dataset-round-{round_num}-gpt-4o-mini-2024-07-18.json"
            )
            llm_judge_file = (
                date_folder
                / f"rated-llm-judge-rated-openai-moderation-dataset-round-{round_num}-gpt-4o-mini-2024-07-18.json"
            )
            openai_file = (
                date_folder
                / f"rated-openai-moderation-dataset-round-{round_num}-gpt-4o-mini-2024-07-18.json"
            )

            # compute fresh
            diversity_stats: Dict[str, Any] = {}
            safety_stats: Dict[str, Any] = {}

            try:
                params = {"sample_size": None, "n_gram": 4}
                # compute fresh
                df = pl.read_json(dataset_file)
                df_filtered = filter_assistant_messages(df)

                diversity_stats["self_bleu"] = calculate_dataset_self_bleu(
                    df_filtered,
                    sample_size=params["sample_size"],
                    n_gram=params["n_gram"],
                )
                diversity_stats["vocab_size"] = calculate_vocab_size(
                    df_filtered, n_gram=params["n_gram"]
                )

                zipf_results = calculate_zipf_coefficient(df_filtered)
                diversity_stats.update(zipf_results)

                conversations = df_filtered["conversation"].to_list()
                diversity_stats["unique_trigrams_pct"] = percent_unique_ngrams(
                    conversations, n=3
                )
                diversity_stats["unique_messages"] = count_unique_messages(
                    conversations
                )
                diversity_stats["total_conversations"] = len(df_filtered)
            except Exception as e:
                print(f"    Error processing dataset: {e}")

            # 2) Llama Guard
            if llama_file.exists():
                try:
                    safety_stats.update(process_llama_guard_scores(llama_file))
                except Exception as e:
                    print(f"    Error processing Llama Guard: {e}")

            # 3) LLM Judge
            if llm_judge_file.exists():
                try:
                    safety_stats.update(process_llm_judge_scores(llm_judge_file))
                except Exception as e:
                    print(f"    Error processing LLM Judge {llm_judge_file}:\n{e}")

            # 4) OpenAI moderation
            if openai_file.exists():
                try:
                    safety_stats.update(process_openai_moderation_scores(openai_file))
                except Exception as e:
                    print(f"    Error processing OpenAI moderation: {e}")

            # merge (fixes your overwrite bug)
            merged = {"diversity": diversity_stats, "safety": safety_stats}
            results[date][round_key] = merged

    # Save results
    if output_file:
        # Convert any numpy types to native Python types for JSON serialization

        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            return obj

        results_serializable = convert_types(dict(results))

        with open(output_file, "w") as f:
            json.dump(results_serializable, f, indent=2)
        print(f"Results saved to {output_file}")

    return dict(results)


def load_and_process_data(filepath: str) -> pl.DataFrame:
    """Load JSON data and convert to polars DataFrame.

    Args:
        filepath: Path to the JSON file containing daily statistics

    Returns:
        Processed DataFrame with date, conversation_type, and metric columns
    """
    data = load_json_data(filepath)

    rows = []
    for date_str, rounds_data in data.items():
        for round_name, round_data in rounds_data.items():
            # Map round names to conversation types
            conversation_type = {
                "round_2": "Standardized",
                "round_3": "Open-ended",
            }.get(round_name, round_name)

            # Extract diversity and safety metrics
            diversity_metrics = round_data.get("diversity", {})
            safety_metrics = round_data.get("safety", {})

            # Flatten safety metrics
            flattened_safety = {}
            for safety_category, safety_data in safety_metrics.items():
                for key, value in safety_data.items():
                    flattened_safety[f"{safety_category}_{key}"] = value

            row = {
                "date": datetime.strptime(date_str, "%Y-%m-%d"),
                "conversation_type": conversation_type,
                **diversity_metrics,
                **flattened_safety,
            }
            rows.append(row)

    return pl.DataFrame(rows)


def create_metrics_plot(
    df: pl.DataFrame,
    metrics: List[str],
    metric_labels: Dict[str, str],
    fig_size: Tuple[int, int] = (16, 12),
) -> ggplot:
    """Create publication-ready time series plots for metrics.

    Args:
        df: DataFrame containing the data
        metrics: List of metric names to plot
        metric_labels: Dictionary mapping metric names to display labels
        fig_size: Figure size as (width, height) tuple

    Returns:
        ggplot object with faceted time series plots
    """
    # Convert to pandas for plotnine compatibility
    df_pandas = df.to_pandas()

    # Melt data to long format for faceting
    melted_data = []
    for _, row in df_pandas.iterrows():
        for metric in metrics:
            if metric in row and row[metric] is not None:
                melted_data.append(
                    {
                        "date": row["date"],
                        "conversation_type": row["conversation_type"],
                        "metric_name": metric_labels[metric],
                        "metric_value": row[metric],
                    }
                )

    plot_df = pl.DataFrame(melted_data).to_pandas()

    # Define colors for conversation types (order matters for legend)
    color_map = {
        "Standardized": PlotConfig.PRIMARY_COLOR,
        "Open-ended": PlotConfig.SECONDARY_COLOR,
    }

    # Set factor levels to control legend order
    plot_df["conversation_type"] = pd.Categorical(
        plot_df["conversation_type"], categories=["Standardized", "Open-ended"]
    )

    # Create theme elements
    theme_elements = _get_base_theme_elements()
    theme_elements.update(
        {
            "figure_size": fig_size,
            "strip_text": _get_text_element(PlotConfig.FONT_SIZE_BOLD, bold=True),
        }
    )

    p = (
        ggplot(plot_df, aes(x="date", y="metric_value", color="conversation_type"))
        + geom_line(size=1.2, alpha=0.8)
        + geom_point(size=3, alpha=0.9)
        + scale_color_manual(values=color_map, name="Prompt Type")
        + scale_x_datetime(date_labels="%b %d", date_breaks="1 day")
        + facet_wrap("~metric_name", scales="free_y", ncol=2)
        + labs(x="Date", y="Metric Value")
        + theme_minimal()
        + theme(**theme_elements)
    )

    return p


def run_comprehensive_analysis(
    data_dir: Path,
    ratings_dir: Path,
    output_file: str = "comprehensive_statistics.json",
    show_plots: bool = True,
    save_plots: bool = True,
) -> dict:
    """Run comprehensive analysis including statistics, plots, and tables.

    Args:
        data_dir: Path to the dataset directory
        ratings_dir: Path to the ratings directory
        output_file: Name of the output JSON file for statistics
        show_plots: Whether to display plots in notebook
        save_plots: Whether to save plots to files

    Returns:
        dict: Dictionary containing all analysis results
    """

    results = {}

    print("🔄 Calculating comprehensive statistics...")
    # Calculate comprehensive statistics
    stats_results = calculate_comprehensive_statistics(
        data_dir=data_dir,
        ratings_dir=ratings_dir,
        output_file=output_file,
    )
    results["statistics"] = stats_results

    print("📊 Loading and processing data for plotting...")
    # Load and process data for plotting
    df = load_and_process_data(output_file)
    results["processed_data"] = df

    # Define metrics and their display labels (diversity metrics)
    diversity_metrics = ["self_bleu", "zipf_coefficient", "unique_trigrams_pct"]
    diversity_labels = {
        "self_bleu": "Self-BLEU Score",
        "zipf_coefficient": "Zipf Coefficient",
        "unique_trigrams_pct": "Unique Trigrams (%)",
    }

    print("📈 Creating diversity plots...")
    # Create and display diversity plot
    diversity_plot = create_metrics_plot(df, diversity_metrics, diversity_labels)
    if show_plots:
        display(diversity_plot)
    if save_plots:
        diversity_plot.save("diversity_metrics.png", dpi=300, bbox_inches="tight")
    results["diversity_plot"] = diversity_plot

    # Safety metrics (if available)
    safety_metrics = ["llm_judge_avg_rating"]
    safety_labels = {
        "llm_judge_avg_rating": "LLM Judge Rating",
    }

    # Filter for safety metrics that exist
    available_safety_metrics = [
        m for m in safety_metrics if any(m in row for row in df.to_dicts())
    ]

    if available_safety_metrics:
        print("🛡️ Creating safety plots...")
        safety_plot = create_metrics_plot(
            df,
            available_safety_metrics,
            {k: v for k, v in safety_labels.items() if k in available_safety_metrics},
        )
        if show_plots:
            display(safety_plot)
        if save_plots:
            safety_plot.save("safety_metrics.png", dpi=300, bbox_inches="tight")
        results["safety_plot"] = safety_plot

    # Print summary
    print("\n📋 Data Summary:")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Conversation types: {', '.join(df['conversation_type'].unique().sort())}")
    print(f"Total data points: {len(df)}")

    # Load JSON data for tables and additional plots
    print("📄 Loading data for tables...")
    data = load_json_data(output_file)

    print("📊 Generating LaTeX tables...")
    # Generate main LaTeX table
    latex_table = generate_latex_table(data)
    results["main_latex_table"] = latex_table

    # Generate flag categories LaTeX table
    flag_categories_table = generate_flag_categories_latex_table(data)
    results["flag_categories_table"] = flag_categories_table

    print("📈 Creating flag analysis plots...")
    # Create separate plots with distinct legends
    llama_plot, openai_plot = plot_flags_by_turn_separate_legends_stacked(data)

    if llama_plot and show_plots:
        display(llama_plot)
    if openai_plot and show_plots:
        display(openai_plot)

    results["llama_plot"] = llama_plot
    results["openai_plot"] = openai_plot

    print("✅ Comprehensive analysis complete!")
    return results


def print_analysis_tables(results: dict) -> None:
    """Print the LaTeX tables from analysis results.

    Args:
        results: Dictionary returned from run_comprehensive_analysis
    """
    if "main_latex_table" in results:
        print("=" * 60)
        print("MAIN STATISTICS TABLE (LaTeX):")
        print("=" * 60)
        print(results["main_latex_table"])

    if "flag_categories_table" in results:
        print("\n" + "=" * 60)
        print("FLAG CATEGORIES TABLE (LaTeX):")
        print("=" * 60)
        print(results["flag_categories_table"])
