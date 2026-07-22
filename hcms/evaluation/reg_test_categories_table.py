"""Generate summary of flag categories from JSON evaluation data."""

import json
from collections import defaultdict
from typing import Dict, Tuple

from hcms.utils import UTF_8


def load_json_data(filepath: str = "comprehensive_statistics.json") -> dict:
    """Load JSON data from file.

    Args:
        filepath: Path to JSON file (default: comprehensive_statistics.json)

    Returns:
        dict: Loaded JSON data
    """
    with open(filepath, "r", encoding=UTF_8) as f:
        return json.load(f)


def analyze_flag_categories(
    data: dict,
) -> Tuple[Dict[str, Dict[str, int]], Dict[str, Dict[str, int]]]:
    """Analyze flag categories across all days for both safety systems.

    Args:
        data: JSON data containing evaluation results

    Returns:
        tuple: (llama_guard_totals, openai_moderation_totals) where each is a dict
               with round names as keys and category counts as values
    """
    llama_guard_totals = defaultdict(lambda: defaultdict(int))
    openai_moderation_totals = defaultdict(lambda: defaultdict(int))

    # Mapping for round names
    round_mapping = {"round_2": "Standardized", "round_3": "Open-Ended"}

    # Aggregate across all dates
    for date in data:
        for round_key, round_name in round_mapping.items():
            if round_key in data[date]:
                round_data = data[date][round_key]["safety"]

                # Process Llama Guard categories
                if (
                    "llama_guard" in round_data
                    and "categories_count" in round_data["llama_guard"]
                ):
                    for category, count in round_data["llama_guard"][
                        "categories_count"
                    ].items():
                        llama_guard_totals[round_name][category] += count

                # Process OpenAI Moderation categories
                if (
                    "openai_moderation" in round_data
                    and "categories_count" in round_data["openai_moderation"]
                ):
                    for category, count in round_data["openai_moderation"][
                        "categories_count"
                    ].items():
                        openai_moderation_totals[round_name][category] += count

    return dict(llama_guard_totals), dict(openai_moderation_totals)


def display_flag_categories(data: dict) -> None:
    """Display flag categories summary for both safety systems.

    Args:
        data: JSON data containing evaluation results
    """
    llama_guard_totals, openai_moderation_totals = analyze_flag_categories(data)

    print("=== FLAG CATEGORIES SUMMARY ===\n")

    # Display Llama Guard results
    print("LLAMA GUARD FLAGS:")
    print("-" * 50)
    for round_name in ["Standardized", "Open-Ended"]:
        if round_name in llama_guard_totals:
            print(f"\n{round_name} Prompts:")
            if llama_guard_totals[round_name]:
                for category, count in sorted(llama_guard_totals[round_name].items()):
                    print(f"  {category}: {count}")
            else:
                print("  No flags")
        else:
            print(f"\n{round_name} Prompts: No data")

    # Display OpenAI Moderation results
    print("\n\nOPENAI MODERATION FLAGS:")
    print("-" * 50)
    for round_name in ["Standardized", "Open-Ended"]:
        if round_name in openai_moderation_totals:
            print(f"\n{round_name} Prompts:")
            if openai_moderation_totals[round_name]:
                for category, count in sorted(
                    openai_moderation_totals[round_name].items()
                ):
                    print(f"  {category}: {count}")
            else:
                print("  No flags")
        else:
            print(f"\n{round_name} Prompts: No data")


def escape_latex_chars(text: str) -> str:
    """Escape special LaTeX characters in text.

    Args:
        text: Input text that may contain special LaTeX characters

    Returns:
        str: Text with escaped LaTeX characters
    """
    return (
        text.replace("&", "\\&")
        .replace("%", "\\%")
        .replace("$", "\\$")
        .replace("#", "\\#")
        .replace("_", "\\_")
    )


def generate_flag_categories_latex_table(data: dict) -> str:
    """Generate LaTeX table showing flag categories by round type.

    Args:
        data: JSON data containing evaluation results

    Returns:
        str: LaTeX table code
    """
    llama_guard_totals, openai_moderation_totals = analyze_flag_categories(data)

    # Collect all unique categories
    all_llama_categories = set()
    all_openai_categories = set()

    for round_name in ["Standardized", "Open-Ended"]:
        if round_name in llama_guard_totals:
            all_llama_categories.update(llama_guard_totals[round_name].keys())
        if round_name in openai_moderation_totals:
            all_openai_categories.update(openai_moderation_totals[round_name].keys())

    # Sort categories alphabetically
    sorted_llama_categories = sorted(all_llama_categories)
    sorted_openai_categories = sorted(all_openai_categories)

    table = []
    table.append("\\begin{table}[t!]")
    table.append("\\centering")
    table.append("\\begin{tabular}{ll|cc}")
    table.append("\\toprule")
    table.append("& & \\multicolumn{2}{c}{\\textbf{Prompt Type}} \\\\")
    # table.append("\\cmidrule(lr){3-4}")
    table.append(
        "\\textbf{Evaluation} & \\textbf{Flag Category} & Standardized & Open-Ended \\\\"
    )
    table.append("\\midrule")

    # Calculate totals for percentage calculations
    llama_std_total = sum(llama_guard_totals.get("Standardized", {}).values())
    llama_open_total = sum(llama_guard_totals.get("Open-Ended", {}).values())
    openai_std_total = sum(openai_moderation_totals.get("Standardized", {}).values())
    openai_open_total = sum(openai_moderation_totals.get("Open-Ended", {}).values())

    # Llama Guard section
    first_llama = True
    for category in sorted_llama_categories:
        escaped_category = escape_latex_chars(category)
        std_count = llama_guard_totals.get("Standardized", {}).get(category, 0)
        open_count = llama_guard_totals.get("Open-Ended", {}).get(category, 0)

        # Calculate percentages
        std_pct = (std_count / llama_std_total * 100) if llama_std_total > 0 else 0
        open_pct = (open_count / llama_open_total * 100) if llama_open_total > 0 else 0

        if first_llama:
            table.append(
                f"\\textbf{{\\llama~Guard}} & {escaped_category} & {std_count} ({std_pct:.1f}\\%) & {open_count} ({open_pct:.1f}\\%) \\\\"
            )
            first_llama = False
        else:
            table.append(
                f"& {escaped_category} & {std_count} ({std_pct:.1f}\\%) & {open_count} ({open_pct:.1f}\\%) \\\\"
            )

    # Add Llama Guard sum row
    if sorted_llama_categories:
        table.append("\\cmidrule(lr){2-4}")
        table.append(
            f"& \\textbf{{Sum}} & \\textbf{{{llama_std_total}}} (100.0\\%) & \\textbf{{{llama_open_total}}} (100.0\\%) \\\\"
        )

    if sorted_llama_categories and sorted_openai_categories:
        table.append("\\midrule")

    # OpenAI Moderation section
    if sorted_openai_categories:
        first_openai = True
        for category in sorted_openai_categories:
            escaped_category = escape_latex_chars(category)
            std_count = openai_moderation_totals.get("Standardized", {}).get(
                category, 0
            )
            open_count = openai_moderation_totals.get("Open-Ended", {}).get(category, 0)

            # Calculate percentages
            std_pct = (
                (std_count / openai_std_total * 100) if openai_std_total > 0 else 0
            )
            open_pct = (
                (open_count / openai_open_total * 100) if openai_open_total > 0 else 0
            )

            if first_openai:
                table.append(
                    f"\\textbf{{\\openai~Moderation}} & {escaped_category} & {std_count} ({std_pct:.1f}\\%) & {open_count} ({open_pct:.1f}\\%) \\\\"
                )
                first_openai = False
            else:
                table.append(
                    f"& {escaped_category} & {std_count} ({std_pct:.1f}\\%) & {open_count} ({open_pct:.1f}\\%) \\\\"
                )

        # Add OpenAI Moderation sum row
        table.append("\\cmidrule(lr){2-4}")
        table.append(
            f"& \\textbf{{Sum}} & \\textbf{{{openai_std_total}}} (100.0\\%) & \\textbf{{{openai_open_total}}} (100.0\\%) \\\\"
        )
    else:
        table.append(
            "\\textbf{\\openai~Moderation} & No flags & 0 (0.0\\%) & 0 (0.0\\%) \\\\"
        )

    table.append("\\bottomrule")
    table.append("\\end{tabular}")
    table.append(
        "\\caption{\\textbf{Flagged messages by prompt type and category.} Number and percentage of flagged messages per category for standardized and open-ended prompts, as classified by Llama Guard and OpenAI Moderation models.}"
    )
    table.append("\\label{tab:flag_categories}")
    table.append("\\end{table}")

    return "\n".join(table)
