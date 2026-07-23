"""Evaluate conversations using Llama Guard."""

from pathlib import Path

import polars as pl
from hcms.loading.constants import DATASET_DIR
from hcms.utils import get_env_variable
from hcms.utils.cache import get_cached_result, save_cached_result
from hcms.utils.moderation import RatingConfig, _rate_and_save_conversations
from openai import OpenAI


def _rate_text_with_llama_guard(content: str) -> str:
    """Evaluates text content for safety using the Llama Guard model.

    This function sends the provided text to the Llama Guard model for evaluation.
    It implements a caching mechanism to avoid re-evaluating the same content.

    If the model's response is "safe", the function returns "0". If the response
    indicates "unsafe" content, it parses the specific policy violation code
    (e.g., "S1") and returns the corresponding human-readable category (e.g.,
    "Violent Crimes"). For any other unexpected response, the raw output from
    the model is returned.

    Args:
        content (str): The text content to be evaluated.

    Returns:
        str: A string indicating the safety evaluation result.
             - "0" for safe content.
             - The name of the policy violation for unsafe content.
             - The raw model output for unexpected responses.
    """
    if not content:
        return ""

    # Check cache first
    cached_result = get_cached_result(content, "llama_guard")
    if cached_result is not None:
        return cached_result

    llama_guard_dict = {
        "S1": "Violent Crimes",
        "S2": "Non-Violent Crimes",
        "S3": "Sex-Related Crimes",
        "S4": "Child Sexual Exploitation",
        "S5": "Defamation",
        "S6": "Specialized Advice",
        "S7": "Privacy",
        "S8": "Intellectual Property",
        "S9": "Indiscriminate Weapons",
        "S10": "Hate",
        "S11": "Suicide & Self-Harm",
        "S12": "Sexual Content",
        "S13": "Elections",
        "S14": "Code Interpreter Abuse",
    }

    client_kwargs = {
        "api_key": get_env_variable("LLAMA_GUARD_API_KEY"),
        "base_url": get_env_variable("LLAMA_GUARD_ENDPOINT"),
    }
    client = OpenAI(**client_kwargs)
    response = client.chat.completions.create(
        model="meta-llama/Llama-Guard-4-12B",
        messages=[
            {
                "role": "user",
                "content": content.strip(),
            },
        ],
    )
    rating = response.choices[0].message.content

    if rating.lower() == "safe":
        res = "0"
    elif rating.lower().startswith("unsafe"):
        res = llama_guard_dict[rating.split("\n")[1]]
    else:
        res = rating

    # Save to cache
    save_cached_result(content, res, "llama_guard")

    return res


def rate_conversations_with_llama_guard(
    file_pattern: str = "dataset-round-2-*.json",
    input_dir: Path = DATASET_DIR,
    output_dir: Path = DATASET_DIR,
    output_prefix: str = "rated-llama-guard-",
) -> pl.DataFrame:
    """Rates conversations in JSON files using Llama Guard.

    This function iterates through files in the input directory matching a
    glob pattern. For each file, it flattens the conversation data, evaluates
    each message using the `_rate_text_with_llama_guard` function, and saves the
    results to a new JSON file in the output directory.

    Args:
        file_pattern: The glob pattern to find input files.
        input_dir: The directory containing the input JSON files.
        output_dir: The directory where rated JSON files will be saved.
        output_prefix: The prefix to add to the output filenames.

    Returns:
        pl.DataFrame: A Polars DataFrame containing the rated conversation
          from the last processed file.
    """
    config = RatingConfig(
        file_pattern=file_pattern,
        rating_function=_rate_text_with_llama_guard,
        output_column_name="llama_guard_score",
        output_column_dtype=pl.String,
        input_dir=input_dir,
        output_dir=output_dir,
        output_prefix=output_prefix,
    )

    return _rate_and_save_conversations(config)
