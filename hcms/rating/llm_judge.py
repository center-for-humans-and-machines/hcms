"""Rate conversations in batch using LLM-as-a-judge."""

from pathlib import Path
from typing import List

from hcms.loading.constants import DATASET_DIR
from hcms.monitoring import RateMessagesProcessor
from hcms.utils import UTF_8
from hcms.utils.cache import get_cached_result, save_cached_result


def _process_single_file(filename: str) -> str:
    """Process a single file using LLM-as-a-judge with caching support.

    Args:
        filename (str): Name of the dataset file to process.

    Returns:
        str: The rating results as a string.
    """
    print(f"Processing {filename}")

    # Check if results are already cached using filename as content key
    cached_result = get_cached_result(filename, "llm-judge-batch")
    if cached_result is not None:
        print(f"Found cached results for {filename}")
        return str(cached_result)

    # Process the file using LLM judge
    input_path = DATASET_DIR / Path(filename)
    processor = RateMessagesProcessor(input_file=str(input_path))
    results = processor.process_batch()

    # Save results to cache using filename as content key
    save_cached_result(filename, results, "llm-judge-batch")

    # Save results to output file
    output_path = DATASET_DIR / f"rated-llm-judge-{filename}"
    with open(output_path, "w", encoding=UTF_8) as f:
        f.write(str(results))

    return str(results)


def rate_conversations_with_llm_as_judge(
    file_pattern: str = "dataset-round-2-*.json",
    input_dir: Path = DATASET_DIR,
) -> List[str]:
    """Rate conversations in batch using LLM-as-a-judge for multiple files.

    This function processes multiple dataset files matching a pattern through
    an LLM judge, with caching support to avoid reprocessing the same files.

    Args:
        file_pattern (str): The glob pattern for files to process.
        input_dir (Path): The directory containing the input JSON files.

    Returns:
        List[str]: A list of rating results as strings, one per processed file.

    Note:
        - Cache files are stored in DATASET_DIR/cache/
        - Output files are prefixed with 'rated-llm-judge'
        - Cache is based on filename hash to avoid reprocessing same files
    """
    input_files = list(input_dir.glob(file_pattern))

    if not input_files:
        print(f"No files found matching pattern: {file_pattern}")
        return []

    print(f"Found {len(input_files)} files to process")

    results = []
    for file_path in input_files:
        filename = file_path.name
        result = _process_single_file(filename)
        results.append(result)

    print(f"Completed processing {len(results)} files")
    return results
