"""Generate a dataset of conversations for specified round."""

import argparse
import asyncio
import pickle
from pathlib import Path
from typing import List, Optional

import backoff
import polars as pl
from langfuse import get_client
from tqdm import tqdm

from hpms.loading.constants import (
    DATASET_2_CONVERSATION_STARTERS,
    DATASET_3_CONVERSATION_STARTERS,
    DATASET_DIR,
)
from hpms.loading.models import ConversationType
from hpms.monitoring.processors import InSilicoConversationProcessor
from hpms.utils import UTF_8, get_env_variable


def read_lines(file_path: Path) -> List[str]:
    """Read file lines into a list of strings.

    Args:
        file_path: Path to the file

    Returns:
        List of strings, one per line
    """
    with open(file_path, "r", encoding=UTF_8) as f:
        return [line.strip() for line in f.readlines()]


def get_conversation_type(round_number: int) -> ConversationType:
    """Map round number to conversation type."""
    if round_number == 2:
        return ConversationType.STANDARDIZED_SAFETY

    if round_number == 3:
        return ConversationType.OPEN_ENDED

    raise ValueError(f"Unsupported round number: {round_number}")


def get_conversation_starters(round_number: int) -> List[str]:
    """Get conversation starters based on round number."""
    if round_number == 2:
        return pl.read_csv(DATASET_2_CONVERSATION_STARTERS)["Question"].to_list()

    if round_number == 3:
        return read_lines(DATASET_3_CONVERSATION_STARTERS)

    raise ValueError(f"Unsupported round number: {round_number}")


@backoff.on_exception(
    backoff.expo,
    (Exception, AttributeError),
    max_time=60,
    max_tries=6,
    on_backoff=lambda details: print(
        f"Backing off {details['wait']:.1f}s after {details['tries']} tries for trace retrieval"
    ),
)
async def get_cost_per_conversation(trace_id: Optional[str]) -> float:
    """Retrieve the cost of a conversation based on its trace ID from Langfuse."""

    if not trace_id:
        print("Warning: No trace ID provided, returning cost as 0.0")
        return 0.0

    try:
        # Initialize Langfuse client
        langfuse = get_client()

        # Retrieve the trace using the trace_id
        trace = await langfuse.async_api.trace.get(trace_id)

        # Get the total cost from the trace
        # Langfuse stores cost information in the trace's total_cost field
        if trace and hasattr(trace, "total_cost") and trace.total_cost is not None:
            return float(trace.total_cost)

        # If no cost information is available, return 0.0
        return 0.0
    except Exception as e:
        print(f"Error retrieving cost for trace {trace_id}: {e}")
        return 0.0


# pylint: disable-next=too-many-locals
async def generate_dataset(
    round_number: int, max_turns: int, tags: Optional[List[str]] = None
):
    """Generate dataset for specified round and model."""
    model_id: str = get_env_variable("CHAT_COMPLETIONS_MODEL_NAME")

    # Replace forward slashes with underscores in model_id
    model_id_sanitized: str = model_id.replace("/", "_")

    # Setup file paths
    dataset_filename: Path = (
        DATASET_DIR / f"dataset-round-{round_number}-{model_id_sanitized}.json"
    )
    dataset_pickle_name: Path = DATASET_DIR / Path(
        f"dataset_round_{round_number}_{model_id_sanitized}_all_dataframes_tmp.pkl"
    )

    # Initialize list to store all dataframes
    all_dataframes: List[pl.DataFrame] = []

    # Get conversation type and starters based on round
    conversation_type = get_conversation_type(round_number)
    conversation_starters = get_conversation_starters(round_number)

    # Run all starters
    for i, starter in tqdm(
        enumerate(conversation_starters),
        total=len(conversation_starters),
        desc=f"Processing conversations (Round {round_number}, {model_id})",
    ):
        # Create processor instance
        processor = InSilicoConversationProcessor(max_turns=max_turns, tags=tags)
        await processor.simulate_conversation(starter)
        conversation = [processor.get_conversation_history()]

        df = pl.DataFrame(
            {
                "conversation_id": f"c{i}",
                "round": round_number,
                "conversation": conversation,
                "conversation_type": conversation_type,
                "model": model_id,
                "model_provider": processor.model_config.provider,
                "api_version": processor.model_config.api_version,
                "temperature": processor.model_config.temperature,
                "companion_system_prompt": processor.prompts.companion,
                "trace_id": processor.session.trace_id,  # Store trace_id temporarily
                "clinician_system_prompt": processor.prompts.clinician,
                "created_at": processor.session.created_at,
                "updated_at": processor.session.updated_at,
                "conversation_duration_s": processor.get_conversation_duration_in_seconds(),
            }
        )

        all_dataframes.append(df)
        # Ensure the dataset directory exists
        dataset_pickle_name.parent.mkdir(parents=True, exist_ok=True)
        # Save list of conversations to a pickle file
        with open(dataset_pickle_name, "wb") as f:
            pickle.dump(all_dataframes, f)

    # Concatenate all dataframes first
    final_dataset = pl.concat(all_dataframes, how="vertical")

    # Collect all costs at once
    print("Collecting cost information for all conversations...")
    trace_ids = final_dataset["trace_id"].to_list()
    costs = []

    for trace_id in tqdm(trace_ids, desc="Fetching costs"):
        cost = await get_cost_per_conversation(trace_id)
        costs.append(cost)

    # Add cost column and remove temporary trace_id column
    final_dataset = final_dataset.with_columns(pl.Series("cost_in_usd", costs)).drop(
        "trace_id"
    )

    # Save the final dataset
    final_dataset.write_json(dataset_filename)

    # Display info about the dataset
    print(f"\nTotal conversations: {len(final_dataset)}")
    print(f"Round: {round_number}")
    print(f"Model: {model_id}")


async def main():
    """Main function to parse arguments and generate dataset."""
    parser = argparse.ArgumentParser(description="Generate conversation dataset")
    parser.add_argument(
        "--round-number", type=int, required=True, help="Round number (2, 3)"
    )
    parser.add_argument(
        "--tag",
        type=str,
        required=False,
        help="Tag for Langfuse trace (e.g., acm-tist)",
    )

    args = parser.parse_args()

    if args.round_number not in [2, 3]:
        raise ValueError("Round number must be either 2 or 3.")

    tag: str | None = None
    if args.tag:
        tag = [args.tag]

    await generate_dataset(args.round_number, max_turns=5, tags=tag)


if __name__ == "__main__":
    asyncio.run(main())
