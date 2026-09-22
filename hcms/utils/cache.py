import ast
import hashlib
from pathlib import Path
from typing import Any, Optional

from hcms.loading.constants import DATASET_DIR
from hcms.utils import UTF_8, get_env_variable


def get_cached_result(
    content: str,
    rater_name: str,
    cache_dir: Optional[Path] = None,
    file_extension: str = ".txt",
) -> Optional[Any]:
    """
    Retrieves cached moderation result for given content and rater.

    Args:
        content (str): The text content to check for cached results
        rater_name (str): Name of the rating system (e.g., 'openai', 'llama', 'llm_judge')
        cache_dir (Optional[Path]): Directory for cache files. Defaults to DATASET_DIR/cache
        file_extension (str): File extension for cache files. Defaults to '.txt'

    Returns:
        Optional[Any]: Cached result if found, None otherwise
    """
    # Check if caching is disabled via environment variable
    skip_cache = get_env_variable("SKIP_CACHE", default="false").lower() == "true"
    if skip_cache:
        return None

    if cache_dir is None:
        cache_dir = Path(DATASET_DIR) / "cache"

    # Create content hash and filename with rater prefix
    content_hash = hashlib.sha256(content.encode("utf8")).hexdigest()
    filename = f"{rater_name}_{content_hash}{file_extension}"
    filepath = cache_dir / filename

    if not filepath.exists():
        return None

    try:
        with open(filepath, "r", encoding=UTF_8) as f:
            result = ast.literal_eval(f.read().strip())
            return result
    except (ValueError, SyntaxError):
        # If cached file is corrupted, return None to regenerate
        return None


def save_cached_result(
    content: str,
    result: Any,
    rater_name: str,
    cache_dir: Optional[Path] = None,
    file_extension: str = ".txt",
) -> None:
    """
    Saves moderation result to cache for given content and rater.

    Args:
        content (str): The text content that was rated
        result (Any): The rating result to cache
        rater_name (str): Name of the rating system (e.g., 'openai', 'llama', 'llm_judge')
        cache_dir (Optional[Path]): Directory for cache files. Defaults to DATASET_DIR/cache
        file_extension (str): File extension for cache files. Defaults to '.txt'
    """
    # Check if caching is disabled via environment variable
    skip_cache = get_env_variable("SKIP_CACHE", default="false").lower() == "true"
    if skip_cache:
        return

    if cache_dir is None:
        cache_dir = Path(DATASET_DIR) / "cache"

    # Ensure cache directory exists
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Create content hash and filename with rater prefix
    content_hash = hashlib.sha256(content.encode("utf8")).hexdigest()
    filename = f"{rater_name}_{content_hash}{file_extension}"
    filepath = cache_dir / filename

    with open(filepath, "w", encoding=UTF_8) as f:
        f.write(str(result))
