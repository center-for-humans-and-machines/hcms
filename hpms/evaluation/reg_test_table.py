"""Generate LaTeX table from JSON evaluation data for regression testing."""

import json

import numpy as np

from hpms.utils import UTF_8


def load_json_data(filepath) -> dict:
    """Load JSON data from file."""
    with open(filepath, "r", encoding=UTF_8) as f:
        return json.load(f)


def find_best_values(data, metric_path, higher_is_better=True):
    """Find the best value for a metric across all dates and rounds."""
    values = []
    for date in data:
        for round_type in ["round_2", "round_3"]:
            if round_type in data[date]:
                # Navigate through nested structure
                current = data[date][round_type]
                for key in metric_path:
                    if key in current:
                        current = current[key]
                    else:
                        current = None
                        break
                if current is not None:
                    values.append(current)

    if not values:
        return None

    return max(values) if higher_is_better else min(values)


def format_value(value, is_best, decimal_places=2):
    """Format value with bold if it's the best."""
    if isinstance(value, float):
        formatted = f"{value:.{decimal_places}f}"
    else:
        formatted = str(value)

    return f"\\textbf{{{formatted}}}" if is_best else formatted


def format_short_number(n):
    """Format large numbers in a shortened readable form (e.g., 23.2k)."""
    if n is None:
        return "N/A"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def generate_latex_table(data, skip_llm_judge=True) -> str:
    """Generate LaTeX table from JSON data in ACM TIST style."""

    # Sort dates
    dates = sorted(data.keys())
    # Create day number headers starting from 1
    date_headers = [str(i + 1) for i, _ in enumerate(dates)]

    # Define metrics and their properties
    diversity_metrics = [
        # (category, metric_key, display_name, higher_is_better, decimal_places, vocab_key)
        ("diversity", "self_bleu", "Self-BLEU $\\downarrow$", False, 2, "vocab_size"),
        ("diversity", "zipf_coefficient", "Zipf $\\downarrow$", False, 2, None),
        (
            "diversity",
            "unique_trigrams_pct",
            "\\% Unique trigrams $\\uparrow$",
            True,
            1,
            None,
        ),
    ]

    safety_metrics = [
        ("safety", "llm_judge", "mean_rating", "LLM-as-a-Judge $\\uparrow$", True, 2),
        (
            "safety",
            "llama_guard",
            "flagged_messages",
            "\\#~\llama~Guard flagged $\\downarrow$",
            False,
            0,
        ),
        (
            "safety",
            "openai_moderation",
            "flagged_messages",
            "\\#~\openai~Moderation flagged $\\downarrow$",
            False,
            0,
        ),
    ]

    # Start building the table
    # ACM TIST style: booktabs, small font, tight column spacing, no resizebox
    table = []
    table.append("\\begin{table*}[t!]")
    table.append("\\centering")
    table.append("\\small")
    table.append("\\setlength{\\tabcolsep}{4pt}")
    table.append(f"\\begin{{tabular}}{{lll|{'c' * len(dates)}|c}}")
    table.append("\\toprule")

    # Header
    from datetime import datetime as _dt
    date_cols = " & ".join(date_headers)
    # Short date labels (e.g. "Apr~8") shown under each day number
    short_date_cols = " & ".join(
        _dt.strptime(d, "%Y-%m-%d").strftime("%b~%d") for d in dates
    )
    first_date_fmt = _dt.strptime(dates[0], "%Y-%m-%d").strftime("%b~%d,~%Y")
    last_date_fmt = _dt.strptime(dates[-1], "%Y-%m-%d").strftime("%b~%d,~%Y")
    period_label = (
        first_date_fmt if len(dates) == 1 else f"{first_date_fmt}--{last_date_fmt}"
    )
    table.append(
        f"{{}} & {{}} & {{}} & \\multicolumn{{{len(dates)}}}{{c|}}{{\\textbf{{Days}} (\\textit{{{period_label}}})}} & \\textbf{{Mean $\\pm$ SD}} \\\\"
    )
    table.append(
        f"\\textbf{{Prompt Type}} & \\textbf{{Metric Type}} & \\textbf{{Metric}} & {date_cols} & \\\\"
    )
    table.append(
        f"& & & {short_date_cols} & \\\\"
    )
    table.append("\\midrule")

    # Process both prompt types
    for round_key, round_name in [
        ("round_2", "Standardized"),
        ("round_3", "Open-Ended"),
    ]:
        first_diversity = True
        for (
            category,
            metric_key,
            display_name,
            higher_is_better,
            decimal_places,
            vocab_key,
        ) in diversity_metrics:
            metric_path = [category, metric_key]
            vocab_path = [category, vocab_key] if vocab_key else None

            # Find best value across all dates for this round only
            values_for_round = []
            for date in dates:
                if round_key in data[date]:
                    current = data[date][round_key]
                    for key in metric_path:
                        if key in current:
                            current = current[key]
                        else:
                            current = None
                            break
                    if current is not None:
                        # Round self_bleu values to two decimals for comparison
                        if metric_key == "self_bleu" and isinstance(current, float):
                            current = round(current, decimal_places)
                        values_for_round.append(current)

            best_value = (
                max(values_for_round)
                if higher_is_better and values_for_round
                else min(values_for_round)
                if values_for_round
                else None
            )

            # Generate row values and collect numeric values for statistics
            values = []
            numeric_values = []
            for date in dates:
                if round_key in data[date]:
                    current = data[date][round_key]
                    for key in metric_path:
                        if key in current:
                            current = current[key]
                        else:
                            current = "N/A"
                            break

                    # Get vocab size if relevant
                    vocab_val = None
                    if vocab_path:
                        vocab_current = data[date][round_key]
                        for key in vocab_path:
                            if key in vocab_current:
                                vocab_current = vocab_current[key]
                            else:
                                vocab_current = None
                                break
                        vocab_val = vocab_current

                    # Round self_bleu for comparison and display
                    display_current = current
                    if metric_key == "self_bleu" and isinstance(current, float):
                        display_current = round(current, decimal_places)
                        current = display_current  # For is_best and numeric_values

                    if isinstance(current, (int, float)):
                        numeric_values.append(current)

                    is_best = (
                        (current == best_value)
                        if isinstance(current, (int, float))
                        else False
                    )

                    if (
                        isinstance(display_current, float)
                        and metric_key == "unique_trigrams_pct"
                    ):
                        display_current *= 100  # Convert to percentage

                    # --- Add vocab size next to Self-BLEU ---
                    if metric_key == "self_bleu" and vocab_val is not None:
                        formatted = format_value(
                            display_current, is_best, decimal_places
                        )
                        formatted_vocab = format_short_number(vocab_val)
                        formatted = f"${formatted}_{{V={formatted_vocab}}}$"
                        values.append(formatted)
                    else:
                        values.append(
                            format_value(display_current, is_best, decimal_places)
                        )
                else:
                    values.append("N/A")

            # Calculate mean and std
            if numeric_values:
                if metric_key == "unique_trigrams_pct":
                    numeric_values = [val * 100 for val in numeric_values]
                numeric_values_rounded = [
                    np.round(val, decimal_places) for val in numeric_values
                ]
                mean_val = np.mean(numeric_values_rounded)
                std_val = (
                    np.round(np.std(numeric_values_rounded, ddof=1), decimal_places)
                    if len(numeric_values_rounded) > 1
                    else 0
                )
                mean_std_str = (
                    f"{mean_val:.{decimal_places}f} $\\pm$ {std_val:.{decimal_places}f}"
                )
            else:
                mean_std_str = "N/A"

            values_str = " & ".join(values)

            if first_diversity:
                table.append(
                    f"\\textbf{{{round_name}}} & Diversity & {display_name} & {values_str} & {mean_std_str} \\\\"
                )
                first_diversity = False
            else:
                table.append(f"& & {display_name} & {values_str} & {mean_std_str} \\\\")

        # Process safety metrics for this round
        first_safety = True
        for (
            category,
            subcategory,
            metric_key,
            display_name,
            higher_is_better,
            decimal_places,
        ) in safety_metrics:
            metric_path = [category, subcategory, metric_key]

            if subcategory == "llm_judge" and skip_llm_judge:
                continue

            # Find best value across all dates for this round only
            values_for_round = []
            for date in dates:
                if round_key in data[date]:
                    current = data[date][round_key]
                    for key in metric_path:
                        if key in current:
                            current = current[key]
                        else:
                            current = None
                            break
                    if current is not None:
                        values_for_round.append(current)

            best_value = (
                max(values_for_round)
                if higher_is_better and values_for_round
                else min(values_for_round)
                if values_for_round
                else None
            )

            # Generate row values and collect numeric values for statistics
            values = []
            numeric_values = []
            for date in dates:
                if round_key in data[date]:
                    current = data[date][round_key]
                    for key in metric_path:
                        if key in current:
                            current = current[key]
                        else:
                            current = "N/A"
                            break

                    # Special handling for LLM-as-a-Judge to include std_rating
                    if (
                        metric_key == "mean_rating"
                        and subcategory == "llm_judge"
                        and isinstance(current, (int, float))
                    ):
                        # Get the std_rating
                        std_path = [category, subcategory, "std_rating"]
                        std_current = data[date][round_key]
                        for key in std_path:
                            if key in std_current:
                                std_current = std_current[key]
                            else:
                                std_current = None
                                break

                        if std_current is not None:
                            formatted_value = f"{current:.{decimal_places}f} $\\pm$ {std_current:.{decimal_places}f}"
                        else:
                            formatted_value = f"{current:.{decimal_places}f}"
                    # For flagged messages, add percentage
                    elif metric_key == "flagged_messages" and isinstance(
                        current, (int, float)
                    ):
                        total_messages_path = [category, subcategory, "total_messages"]
                        total_current = data[date][round_key]
                        for key in total_messages_path:
                            if key in total_current:
                                total_current = total_current[key]
                            else:
                                total_current = None
                                break

                        if total_current and total_current > 0:
                            percentage = (current / total_current) * 100
                            formatted_value = f"{current} ({percentage:.1f}\\%)"
                        else:
                            formatted_value = str(current)
                    else:
                        formatted_value = current

                    if isinstance(current, (int, float)):
                        numeric_values.append(current)

                    is_best = (
                        (current == best_value)
                        if isinstance(current, (int, float))
                        else False
                    )

                    if metric_key == "flagged_messages":
                        values.append(
                            f"\\textbf{{{formatted_value}}}"
                            if is_best
                            else formatted_value
                        )
                    elif metric_key == "mean_rating" and subcategory == "llm_judge":
                        # For LLM-as-a-Judge, formatted_value already includes std
                        values.append(
                            f"\\textbf{{{formatted_value}}}"
                            if is_best
                            else formatted_value
                        )
                    else:
                        values.append(format_value(current, is_best, decimal_places))
                else:
                    values.append("N/A")

            # Calculate mean and std
            if numeric_values:
                numeric_values_rounded = [
                    np.round(val, decimal_places) for val in numeric_values
                ]
                mean_val = np.mean(numeric_values_rounded)
                std_val = (
                    np.round(np.std(numeric_values_rounded, ddof=1), decimal_places)
                    if len(numeric_values_rounded) > 1
                    else 0
                )
                mean_std_str = (
                    f"{mean_val:.{decimal_places}f} $\\pm$ {std_val:.{decimal_places}f}"
                )
            else:
                mean_std_str = "N/A"

            values_str = " & ".join(values)

            # Add metric type for first safety row only
            if first_safety:
                table.append(
                    f"& Safety & {display_name} & {values_str} & {mean_std_str} \\\\"
                )
                first_safety = False
            else:
                table.append(f"& & {display_name} & {values_str} & {mean_std_str} \\\\")

        if round_name == "Standardized":
            table.append("\\midrule")

    table.append("\\bottomrule")
    table.append("\\end{tabular}")

    # Dynamic date range for caption (reuse variables computed for the header)
    date_range = (
        first_date_fmt
        if len(dates) == 1
        else f"{first_date_fmt}--{last_date_fmt}"
    )

    table.append(
        f"\\caption{{\\textbf{{Regression testing results over {len(dates)} consecutive days with diversity and safety metrics.}}"
        f" (\\openai=GPT and \\llama=Llama.)"
        f" $\\uparrow$/$\\downarrow$ indicates whether higher/lower scores are better."
        f" \\#~\\llama~Guard flagged and \\#~\\openai~Moderation flagged report the number of flagged messages"
        f" for at least one category; percentages are over the total messages per day ($n=500$)."
        f" The last column shows mean~$\\pm$~SD."
        f" Self-BLEU subscripts report 4-gram vocabulary size ($V$)."
        f" Results were collected from {date_range}."
        f" Best scores per metric and prompt type are in \\textbf{{bold}}.}}"
    )
    table.append("\\label{tab:comprehensive_metrics}")
    table.append("\\end{table*}")

    return "\n".join(table)
