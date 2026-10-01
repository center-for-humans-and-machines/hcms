"""Calculate Self-BLEU score.

Requirements: Run the following Python commands to download the necessary NLTK resources
import nltk
nltk.download("punkt_tab")

Source:
General
https://www.digitalocean.com/community/tutorials/automated-metrics-for-evaluating-generated-text#self-bleu

n-grams from
https://github.com/ari-holtzman/degen/blob/master/metrics/self_bleu.py

Weights from
https://www.nltk.org/_modules/nltk/translate/bleu_score.html

Original implementation:
https://github.com/geek-ai/Texygen/blob/3104e22ac75f3cc2070da2bf5e2da6d2bef149ad/utils/metrics/SelfBleu.py

Explanation of Self-BLEU:
https://github.com/geek-ai/Texygen/blob/master/docs/evaluation.md
"""

import random
from typing import List, Optional

import numpy as np
import polars as pl
from hcms.utils.cache import get_cached_result, save_cached_result
from nltk import word_tokenize
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu


def calculate_self_bleu(
    sentences: List[str], n_sample: Optional[int] = None, n_gram: int = 4
) -> float:
    """
    Calculate Self-BLEU score following the paper's methodology.

    Args:
        sentences: List of sentences (strings)
        n_sample: Number of sentences to sample for calculation (None = no sampling)
        n_gram: N-gram level for BLEU (1-5). The default BLEU calculates a score for up to 4-grams
                using uniform weights (this is called BLEU-4).

    Returns:
        Self-BLEU score
    """
    # Tokenize all sentences
    tokenized_sentences = [word_tokenize(sent.lower()) for sent in sentences]

    # Sample sentences if n_sample is specified and dataset is large
    if n_sample is not None and len(tokenized_sentences) > n_sample:
        sampled_indices = random.sample(range(len(tokenized_sentences)), n_sample)
        sampled_sentences = [tokenized_sentences[i] for i in sampled_indices]
    else:
        sampled_sentences = tokenized_sentences

    # Set weights based on n-gram
    weights = [
        (1.0,),  # BLEU-1
        (1.0 / 2.0, 1.0 / 2.0),  # BLEU-2
        (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),  # BLEU-3
        (1.0 / 4.0, 1.0 / 4.0, 1.0 / 4.0, 1.0 / 4.0),  # BLEU-4
        (1.0 / 5.0, 1.0 / 5.0, 1.0 / 5.0, 1.0 / 5.0, 1.0 / 5.0),  # BLEU-5
    ]

    if n_gram < 1 or n_gram > 5:
        raise ValueError("n_gram must be between 1 and 5")

    selected_weights = weights[n_gram - 1]  # Convert to 0-based index

    smoothing_function = SmoothingFunction().method1
    bleu_scores = []

    # Calculate BLEU for each sentence against all others
    for i, hypothesis in enumerate(sampled_sentences):
        # Get references (all sentences except current one)
        references = sampled_sentences[:i] + sampled_sentences[i + 1 :]

        # Calculate BLEU score
        score = sentence_bleu(
            references=references,
            hypothesis=hypothesis,
            weights=selected_weights,
            smoothing_function=smoothing_function,
        )
        bleu_scores.append(score)

    return np.mean(bleu_scores)


def calculate_self_bleu_all_ngrams(
    sentences: List[str], n_sample: Optional[int] = None
) -> dict:
    """Calculate Self-BLEU for all n-gram levels (1-5).

    Args:
        sentences: List of sentences (strings)
        n_sample: Number of sentences to sample for calculation (None = no sampling)

    Returns:
        Dictionary with Self-BLEU scores for n-grams 1 to 5
    """
    results = {}

    for n_gram in range(1, 6):
        score = calculate_self_bleu(sentences, n_sample, n_gram)
        results[f"bleu-{n_gram}"] = score
        print(f"Self-BLEU-{n_gram}: {score:.4f}")

    return results


def extract_conversation_texts(df) -> List[str]:
    """Extract text content from conversations for Self-BLEU calculation.

    Args:
        df: DataFrame containing the dataset with a "conversation" column

    Returns:
        List of all text contents from the conversations
    """
    all_texts = []

    for conv in df["conversation"].to_list():
        # Extract content from each message in the conversation
        conv_texts = []
        for message in conv:
            if message["content"].strip():  # Skip empty messages
                conv_texts.append(message["content"])

        all_texts.extend(conv_texts)

    return all_texts


def calculate_dataset_self_bleu(df, sample_size=None, n_gram=4) -> np.float64:
    """Calculate Self-BLEU score for the dataset.

    Args:
        df: DataFrame containing the dataset
        sample_size: Number of samples to use (None = no sampling)
        n_gram: N-gram level for BLEU (1-5)

    Returns:
        Self-BLEU score for the dataset
    """
    # Extract all text content
    texts: List[str] = extract_conversation_texts(df)
    texts_str: str = "\n".join(texts)
    cache_content = f"{texts_str}|sample_size:{sample_size}|n_gram:{n_gram}"

    # Check cache first
    cached_result = get_cached_result(cache_content, "self_bleu")
    if cached_result is not None:
        print(f"Using cached Self-BLEU-{n_gram} result for {len(texts)} texts...")
        return np.float64(cached_result)

    sample_info = f"(sampling {sample_size})" if sample_size else "(no sampling)"
    print(f"Calculating Self-BLEU-{n_gram} for {len(texts)} texts {sample_info}...")

    # Calculate Self-BLEU score
    self_bleu_score = calculate_self_bleu(texts, sample_size, n_gram)

    # Save to cache
    save_cached_result(cache_content, np.float64(self_bleu_score), "self_bleu")

    return self_bleu_score


def calculate_self_bleu_by_group(
    df, group_col="model", sample_size=None, n_gram=4
) -> dict:
    """Calculate Self-BLEU scores grouped by a specific column using corrected implementation.

    Args:
        df: DataFrame containing the dataset
        group_col: Column name to group by (default: "model")
        sample_size: Number of samples to use for each group (None = no sampling)
        n_gram: N-gram level for BLEU (1-5)

    Returns:
        Dictionary with group names as keys and Self-BLEU scores as values
    """
    results = {}

    # For Polars DataFrame
    for group_name in df[group_col].unique():
        group_df = df.filter(pl.col(group_col) == group_name)
        texts = extract_conversation_texts(group_df)

        if len(texts) > 1:  # Need at least 2 texts for comparison
            score = calculate_self_bleu(texts, sample_size, n_gram)
            results[group_name] = score
            print(f"{group_name}: Self-BLEU-{n_gram} = {score:.4f}")

    return results


def calculate_bleu(candidate, reference) -> np.float64:
    """
    Calculate BLEU score for a single candidate sentence against a reference sentence.

    Args:
        candidate: generated sentence
        reference: reference sentence

    Returns:
        BLEU score for the generated sentence against the reference sentence
    """
    reference = word_tokenize(reference)
    candidate = word_tokenize(candidate)
    score = sentence_bleu(
        reference, candidate, smoothing_function=SmoothingFunction().method1
    )
    return score


def get_bleu_score(sentence, remaining_sentences) -> List[np.float64]:
    """
    Calculate BLEU scores for a generated sentence against a list of remaining sentences.

    Args:
        sentence: generated sentence
        remaining_sentences: list of sentences generated by NLG system

    Returns:
        List of BLEU scores for the generated sentence against the remaining sentences
    """
    lst = []
    for i in remaining_sentences:
        bleu = sentence_bleu(
            sentence, i, smoothing_function=SmoothingFunction().method1
        )
        lst.append(bleu)
    return lst


def interpret_self_bleu_score(score: float) -> str:
    """
    Interpret the Self-BLEU score.

    Args:
        score: Self-BLEU score

    Returns:
        Interpretation of the score
    """
    if score < 0.3:
        return "High diversity"

    if score < 0.6:
        return "Medium diversity"

    return "Low diversity"


def calculate_vocab_size(df, n_gram=4) -> int:
    """
    Calculate the vocabulary size (number of unique n-grams) in the dataset.

    Args:
        df: DataFrame containing the dataset with a "conversation" column
        n_gram: n-gram level (default: 1 for unigram)

    Returns:
        Number of unique n-grams in the dataset
    """
    texts = extract_conversation_texts(df)
    ngrams_set = set()

    for text in texts:
        tokens = word_tokenize(text.lower())
        ngrams = zip(*[tokens[i:] for i in range(n_gram)])
        ngrams_set.update(ngrams)

    return len(ngrams_set)
