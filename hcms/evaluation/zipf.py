"""Calculate the Zipf coefficient for a given list of frequencies.

Inspiration:
https://github.com/ari-holtzman/degen/blob/0acfd2d0ba8484e24e9c5241f75f34be15ef2609/metrics/zipf.py#L26C5-L40C30
"""

import re
import unicodedata
from collections import Counter
from typing import Dict, Generator, List, Optional, Tuple, Union

import numpy as np
import polars as pl
from scipy import stats

# words, allowing internal ' and -   (e.g., don't, state-of-the-art)
TOK_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*")


def tokenize_text(text: str) -> list[str]:
    """Tokenize text into words using a simple regex.

    Args:
        text: Input text to tokenize.

    Returns:
        List of tokens extracted from the text.
    """
    s = unicodedata.normalize("NFKC", text).lower()
    return TOK_RE.findall(s)


def extract_content_from_conversation(
    conversations: List[Union[str, List[Dict[str, str]]]],
    role_filter: Optional[str] = None,
) -> Generator[str, None, None]:
    """Extract content from conversations with optional role filtering.

    Args:
        conversations: List containing either strings or lists of message dictionaries
            with 'role' and 'content' keys.
        role_filter: Optional role to filter by (e.g., 'user', 'assistant').
            If None, extracts content from all roles.

    Yields:
        Text content from conversations, filtered by role if specified.
    """
    for conv in conversations:
        if isinstance(conv, str):
            # If conversation is already a string, yield it directly
            if role_filter is None:
                yield conv
        else:
            # Process list of message dictionaries
            for msg in conv:
                if isinstance(msg, dict):
                    content: str = msg.get("content", "")
                    msg_role: str = msg.get("role", "")

                    # Apply role filter if specified
                    if role_filter is None or msg_role == role_filter:
                        yield content


def count_frequencies_in_df(
    df: pl.DataFrame,
    conversation_column: str = "conversation",
    top_n: int = 5000,
    role_filter: Optional[str] = None,
) -> Tuple[List[int], int, int]:
    """Count token frequencies from a Polars DataFrame containing conversations.

    Args:
        df: Polars DataFrame with conversation data.
        conversation_column: Name of column containing conversation objects.
        top_n: Number of most frequent tokens to consider for analysis.
        role_filter: Optional role to filter messages by (e.g., 'assistant', 'user').
            If None, analyzes all messages regardless of role.

    Returns:
        Tuple containing:
            - List of frequencies for the most common tokens
            - Total number of unique tokens
            - Total number of token occurrences

    Raises:
        ValueError: If no tokens are found in the dataset.
    """
    # Initialize token counter
    token_counter: Counter[str] = Counter()

    # Convert to Python list for easier processing
    conversations: List[Union[str, List[Dict[str, str]]]] = df[
        conversation_column
    ].to_list()

    # Process each conversation in the DataFrame
    for text_content in extract_content_from_conversation(conversations, role_filter):
        # Tokenize the content
        tokens: List[str] = tokenize_text(text_content)

        # Update counter with tokens
        token_counter.update(tokens)

    # Check if we have enough data
    if len(token_counter) == 0:
        raise ValueError("No tokens found in the dataset")

    # Calculate totals before filtering to top_n
    total_unique_tokens: int = len(token_counter)
    total_tokens: int = sum(token_counter.values())

    # Get the most frequent tokens up to top_n
    num_tokens: int = min(len(token_counter), top_n)
    most_frequent: List[Tuple[str, int]] = token_counter.most_common(num_tokens)

    # Extract frequencies for analysis
    frequencies: List[int] = [count for _, count in most_frequent]

    return frequencies, total_unique_tokens, total_tokens


# pylint: disable-next=too-many-locals
def calculate_zipf_coefficient(
    df: pl.DataFrame,
    conversation_column: str = "conversation",
    top_n: int = 5000,
    role_filter: Optional[str] = None,
) -> Dict[str, float]:
    """Calculate Zipf coefficient from a Polars DataFrame containing conversations.

    The Zipf coefficient measures how well the token frequency distribution
    follows Zipf's law (frequency ∝ rank^(-α)). Higher values indicate
    more uneven distributions.

    Args:
        df: Polars DataFrame with conversation data.
        conversation_column: Name of column containing conversation objects.
        top_n: Number of most frequent tokens to consider for Zipf analysis.
        role_filter: Optional role to filter messages by (e.g., 'assistant', 'user').
            If None, analyzes all messages regardless of role.

    Returns:
        Dictionary containing:
            - zipf_coefficient: The Zipf exponent (α in frequency ∝ rank^(-α))
            - correlation_coefficient: Correlation of log-log regression
            - p_value: Statistical significance of the regression
            - total_unique_tokens: Number of unique tokens found
            - total_tokens: Total number of token occurrences
            - tokens_analyzed: Number of tokens used in Zipf analysis

    Raises:
        ValueError: If no tokens are found in the dataset.
    """
    # Get token frequencies and totals
    frequencies: List[int]
    total_unique_tokens: int
    total_tokens: int
    frequencies, total_unique_tokens, total_tokens = count_frequencies_in_df(
        df, conversation_column, top_n, role_filter
    )

    # Create rank array (1, 2, 3, ...)
    ranks: np.ndarray = np.arange(1, len(frequencies) + 1)

    # Perform log-log linear regression
    # Zipf's law: frequency ∝ rank^(-α), so log(frequency) = -α * log(rank) + constant
    log_ranks: np.ndarray = np.log(ranks)  # pylint: disable=no-member
    log_frequencies: np.ndarray = np.log(frequencies)  # pylint: disable=no-member

    # Linear regression on log-log data
    slope: float
    # pylint: disable-next=unused-variable
    intercept: float
    r_value: float
    p_value: float
    # pylint: disable-next=unused-variable
    std_err: float
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        log_ranks, log_frequencies
    )

    # Zipf coefficient is the negative of the slope
    zipf_coefficient: float = -slope

    return {
        "zipf_coefficient": zipf_coefficient,
        "correlation_coefficient": -r_value,  # Negative because we want positive correlation
        "p_value": p_value,
        "total_unique_tokens": total_unique_tokens,
        "total_tokens": total_tokens,
        "tokens_analyzed": len(frequencies),
    }


def print_zipf_results(
    results: Dict[str, float], dataset_name: str = "dataset"
) -> None:
    """Print Zipf coefficient results in a formatted way.

    Args:
        results: Dictionary containing Zipf analysis results from calculate_zipf_coefficient.
        dataset_name: Name of the dataset for display purposes.
    """
    print(f"Dataset: {dataset_name}")
    print(f"Zipf coefficient: {results['zipf_coefficient']:.4f}")
    print(f"Correlation coefficient: {results['correlation_coefficient']:.4f}")
    print(f"P-value: {results['p_value']:.6f}")
    print(f"Total unique tokens: {results['total_unique_tokens']:,}")
    print(f"Total tokens: {results['total_tokens']:,}")
    print(f"Tokens analyzed: {results['tokens_analyzed']:,}")
