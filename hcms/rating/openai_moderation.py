"""Check if a message is safe using OpenAI's moderation API."""

from pathlib import Path
from typing import List

import polars as pl
from hcms.loading.constants import DATASET_DIR
from hcms.utils import get_env_variable
from hcms.utils.cache import get_cached_result, save_cached_result
from hcms.utils.moderation import RatingConfig, _rate_and_save_conversations
from openai import OpenAI

client = OpenAI(api_key=get_env_variable("OPENAI_MODERATION_API_KEY"))


def _rate_text_with_openai_moderation(content: str) -> List[int | str]:
    """Evaluates text content for safety using OpenAI's moderation API.

    Args:
        content (str): The text content to be evaluated.
    Returns:
        List[int | str]: A list containing category names (str) for flagged categories
                        and 0 (int) for non-flagged categories."""
    if not content:
        return []

    # Check cache first using the new caching function
    cached_result = get_cached_result(content, "openai_moderation")
    if cached_result is not None:
        # ensure values in cached_result are integers (0 or 1)
        if isinstance(cached_result, list) and all(
            isinstance(x, (int, str)) for x in cached_result
        ):
            return cached_result
        # modify string into int
        return [1 if x == "1" else 0 for x in cached_result]

    all_categories = [
        "harassment",
        "harassment_threatening",
        "hate",
        "hate_threatening",
        "illicit",
        "illicit_violent",
        "self_harm",
        "self_harm_instructions",
        "self_harm_intent",
        "sexual",
        "sexual_minors",
        "violence",
        "violence_graphic",
        "harassment/threatening",
        "hate/threatening",
        "illicit/violent",
        "self-harm/intent",
        "self-harm/instructions",
        "self-harm",
        "sexual/minors",
        "violence/graphic",
    ]

    response = client.moderations.create(
        model="omni-moderation-2024-09-26", input=content.strip()
    )

    category_flags = dict(response.results[0].categories)

    # Explicitly type the list as List[int | str]
    category_bin: List[int | str] = []
    for cat in all_categories:
        if category_flags[cat]:
            category_bin.append(cat)  # String (category name)
        else:
            category_bin.append(0)  # Integer 0, not string "0"

    # Save to cache using the new caching function
    save_cached_result(content, category_bin, "openai_moderation")

    return category_bin


def rate_conversations_with_openai(
    file_pattern: str = "dataset-round-2-*.json",
    input_dir: Path = DATASET_DIR,
    output_dir: Path = DATASET_DIR,
    output_prefix: str = "rated-openai-moderation-",
) -> pl.DataFrame:
    """
    Processes dataset files matching a pattern, rates messages for safety,
    and saves the results to new JSON files.

    Args:
        file_pattern (str): The glob pattern for files to process.
        input_dir: The directory containing the input JSON files.
        output_dir: The directory where rated JSON files will be saved.
        output_prefix: The prefix to add to the output filenames.

    Returns:
        pl.DataFrame: A Polars DataFrame containing the rated conversations
          from the last processed file.
    """
    config = RatingConfig(
        file_pattern=file_pattern,
        rating_function=_rate_text_with_openai_moderation,
        output_column_name="OpenAI_Moderation",
        output_column_dtype=pl.List(pl.String),
        input_dir=input_dir,
        output_dir=output_dir,
        output_prefix=output_prefix,
    )

    return _rate_and_save_conversations(config)
