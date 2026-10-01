""" "Generate and display word clouds from conversation data.
Source: Adapted from https://tinyurl.com/5x9hm2ne
"""

import matplotlib.pyplot as plt
from wordcloud import STOPWORDS, WordCloud


def extract_conversation_text(df, role_filter=None):
    """
    Extract text from conversations with optional role filtering.

    Args:
        df: Polars DataFrame with conversation data
        role_filter: "assistant", "user", or None (no filter)

    Returns:
        str: Concatenated text from conversations
    """
    messages = [
        message["content"]
        for conversation in df["conversation"]
        for message in conversation
        if role_filter is None or message["role"] == role_filter
    ]
    return " ".join(messages)


def create_wordcloud(
    text,
    max_words=2000,
    width=3000,
    height=1500,
):
    """
    Create a WordCloud from text.

    Args:
        text: Input text to generate wordcloud from
        max_words: Maximum number of words to include in the wordcloud
        width: Width of the wordcloud image
        height: Height of the wordcloud image

    Returns:
        WordCloud: Generated wordcloud object
    """
    stopwords = set(STOPWORDS)

    wc = WordCloud(
        background_color="white",
        max_words=max_words,
        width=width,
        height=height,
        stopwords=stopwords,
        random_state=42,
        contour_width=3,
        contour_color="black",
    )
    wc.generate(text)
    return wc


def display_wordcloud(wordcloud) -> None:
    """Display wordcloud using matplotlib.

    Args:
        wordcloud: WordCloud object to display
    """
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()


def save_wordcloud(wordcloud, filepath) -> None:
    """Save wordcloud to file.

    Args:
        wordcloud: WordCloud object to save
        filepath: Path to save the wordcloud image
    """
    wordcloud.to_file(filepath)
