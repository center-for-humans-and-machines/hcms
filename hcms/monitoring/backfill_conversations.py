"""Rate historical messages once, without a change stream.

The realtime watcher only backfills messages newer than `MONGODB_BACKFILL_SINCE`,
so live detection is never stuck behind history. This job rates the rest, one
provider at a time, and can be run again: rated messages are skipped.

Usage:
    poetry run python -m hcms.monitoring.backfill_conversations \\
        --provider llm_judge --since 2026-01-01 --until 2026-11-03
"""

import argparse
import logging
import sys

from hcms.monitoring.realtime_watcher import RealtimeConversationWatcher
from hcms.monitoring.watch_mongo_conversations import (
    MODERATION_PROVIDERS,
    configure_logging,
    create_repository,
    parse_timestamp,
)

LOGGER = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line.

    Args:
        argv: Arguments without the program name (default: `sys.argv[1:]`).

    Returns:
        argparse.Namespace: The parsed options.
    """
    parser = argparse.ArgumentParser(
        description="Rate historical messages with one moderation provider."
    )
    parser.add_argument(
        "--provider",
        required=True,
        choices=sorted(MODERATION_PROVIDERS),
        help="Provider to run (its endpoint and credentials come from the environment).",
    )
    parser.add_argument(
        "--since",
        type=lambda value: parse_timestamp(value, "--since"),
        default=None,
        help="Only messages with a timestamp at or after this ISO 8601 time.",
    )
    parser.add_argument(
        "--until",
        type=lambda value: parse_timestamp(value, "--until"),
        default=None,
        help="Only messages with a timestamp before this ISO 8601 time.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="Messages fetched per batch (default: 20).",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Attempts per message before it is left unrated (default: 3).",
    )
    parser.add_argument(
        "--retry-sleep-seconds",
        type=float,
        default=2.0,
        help="Pause between retries of failed messages (default: 2).",
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    """Run one backfill pass and return the process exit code."""
    repository = create_repository()
    try:
        watcher = RealtimeConversationWatcher(
            repository=repository,
            reviewer_ids={MODERATION_PROVIDERS[args.provider]},
            backfill_batch_size=args.batch_size,
            backfill_max_retries=args.max_retries,
            backfill_retry_sleep_seconds=args.retry_sleep_seconds,
            backfill_since=args.since,
            backfill_until=args.until,
        )
        LOGGER.info(
            "Backfilling %s for messages in [%s, %s)",
            args.provider,
            args.since,
            args.until,
        )
        watcher.run_startup_backfill()
        unresolved = watcher._unresolved_backfill_targets  # pylint: disable=protected-access
    finally:
        repository.close()

    if unresolved:
        LOGGER.error("Backfill left %d message rating(s) unresolved", unresolved)
        return 1
    LOGGER.info("Backfill finished")
    return 0


def main() -> None:
    """Entrypoint."""
    configure_logging()
    sys.exit(run(parse_args()))


if __name__ == "__main__":
    main()
