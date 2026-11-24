"""Evaluate dataset of conversations for specified round."""

import argparse
import asyncio

from hcms.rating.ktb_judge_dataset import rate_conversations_with_ktb_judge
from hcms.rating.llama_guard import rate_conversations_with_llama_guard
from hcms.rating.llm_judge import rate_conversations_with_llm_as_judge
from hcms.rating.openai_moderation import rate_conversations_with_openai
from hcms.utils import get_env_variable


def _flag_enabled(name: str, default: str) -> bool:
    """Whether the environment variable ``name`` is 'true' (case-insensitive)."""
    return get_env_variable(name, default=default).lower() == "true"


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

    # Rate every assistant reply with the Kill The Bot (KTB) judge, unless
    # disabled. Fails the run when too many replies get no verdict.
    if _flag_enabled("LLM_AS_JUDGE_ENABLED", default="true"):
        rate_conversations_with_ktb_judge(
            file_pattern=f"dataset-round-{round_number}-*.json"
        )
    else:
        print("Skipping KTB judge rating: LLM_AS_JUDGE_ENABLED is not 'true'")

    # The previous judge (RateMessagesProcessor with the BotRatingResponse
    # rubric through the OpenAI batch API) is kept behind its own flag, off by
    # default, until the KTB judge has replaced it in practice.
    if _flag_enabled("LEGACY_LLM_JUDGE_ENABLED", default="false"):
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
