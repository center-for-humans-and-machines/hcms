"""Calculate the percentage of unique n-grams in a dataset."""

from collections import Counter
from typing import Any, Dict, List, Optional, Union

from hcms.evaluation.zipf import extract_content_from_conversation, tokenize_text


def percent_unique_ngrams(
    conversations: List[Union[str, List[Dict[str, str]]]],
    n: int = 3,
    role_filter: Optional[str] = None,
) -> float:
    """% of all generated n-grams that occur exactly once (singletons).

    Source:
    https://github.com/ari-holtzman/degen/blob/0acfd2d0ba8484e24e9c5241f75f34be15ef2609/metrics/distinct_n.py#L17-L30

    Args:
        conversations: List of conversation objects or strings.
        n: Size of the n-grams to consider.
        role_filter: Optional role to filter messages by (e.g., 'assistant', 'user').

    Returns:
        Percentage of unique n-grams (singletons) over total n-grams.
    """
    counter: Counter[Any] = Counter()
    n_total: int = 0
    n_unique: int = 0

    for text in extract_content_from_conversation(conversations, role_filter):
        tokens = tokenize_text(text)
        for ng in zip(*(tokens[i:] for i in range(n))):
            if counter[ng] == 0:
                n_unique += 1  # first time seen -> becomes unique
            elif counter[ng] == 1:
                n_unique -= 1  # second time -> no longer unique
            counter[ng] += 1
            n_total += 1

    if not n_total:
        return 0.0

    return n_unique / n_total
    return n_unique / n_total
    return n_unique / n_total
