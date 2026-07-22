"""Evaluate dataset of conversations for specified round."""

import argparse
import asyncio

from hcms.rating.llama_guard import rate_conversations_with_llama_guard
from hcms.rating.llm_judge import rate_conversations_with_llm_as_judge
from hcms.rating.openai_moderation import rate_conversations_with_openai


async def evaluate_dataset(round_number: int):
    """Evaluate dataset of conversations for specified round using multiple rating methods.

    Args:
        round_number (int): The round number for which to evaluate the dataset (2 or 3).
    """

    # Rate conversations with Llama Guard
    rate_conversations_with_llama_guard(
        file_pattern=f"dataset-round-{round_number}-*.json",
    )

    # Rate conversations with OpenAI
    rate_conversations_with_openai(file_pattern=f"dataset-round-{round_number}-*.json")
    rate_conversations_with_llm_as_judge(
        file_pattern=f"rated-openai-moderation-dataset-round-{round_number}-*.json"
    )


async def main():
    """Main function to parse arguments and generate dataset."""
    parser = argparse.ArgumentParser(description="Evaluate dataset")
    parser.add_argument(
        "--round-number", type=int, required=True, help="Round number (2, 3)"
    )

    args = parser.parse_args()

    if args.round_number not in [2, 3]:
        raise ValueError("Round number must be either 2 or 3.")

    await evaluate_dataset(args.round_number)


if __name__ == "__main__":
    asyncio.run(main())
