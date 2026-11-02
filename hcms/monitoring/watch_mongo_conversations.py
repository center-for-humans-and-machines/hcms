"""Run the hcms realtime MongoDB conversation watcher."""

# pylint: disable=duplicate-code

import logging
from datetime import datetime, timezone
from typing import Optional

from hcms.database.repository import (
    SYSTEM_LLAMA_REVIEWER_ID,
    SYSTEM_LLM_JUDGE_REVIEWER_ID,
    SYSTEM_OPENAI_REVIEWER_ID,
    MongoConversationRepository,
)
from hcms.monitoring.realtime_watcher import RealtimeConversationWatcher
from hcms.utils import get_env_variable

# Values accepted in HCMS_MODERATION_PROVIDERS, mapped to their reviewer IDs.
MODERATION_PROVIDERS = {
    "openai_moderation": SYSTEM_OPENAI_REVIEWER_ID,
    "llama_guard": SYSTEM_LLAMA_REVIEWER_ID,
    "llm_judge": SYSTEM_LLM_JUDGE_REVIEWER_ID,
}
# Providers run when HCMS_MODERATION_PROVIDERS is unset: all of them, the LLM
# judge like Llama Guard. A deployment without a provider's endpoint names the
# ones it runs.
DEFAULT_MODERATION_PROVIDERS = ("openai_moderation", "llama_guard", "llm_judge")


def _get_int_env(var_name: str, default_value: int) -> int:
    """Read integer env var with a fallback default."""
    raw_value = get_env_variable(var_name, default=str(default_value))
    return int(raw_value)


def _get_float_env(var_name: str, default_value: float) -> float:
    """Read float env var with a fallback default."""
    raw_value = get_env_variable(var_name, default=str(default_value))
    return float(raw_value)


def parse_timestamp(value: str, var_name: str = "timestamp") -> datetime:
    """Parse an ISO 8601 timestamp; a value without timezone is taken as UTC.

    Args:
        value: The timestamp text, e.g. `2026-11-03` or `2026-11-03T08:00:00Z`.
        var_name: Name used in the error message.

    Returns:
        datetime: A timezone-aware datetime.

    Raises:
        ValueError: If the text is not an ISO 8601 timestamp.
    """
    try:
        parsed = datetime.fromisoformat(value.strip())
    except ValueError as error:
        raise ValueError(
            f"{var_name}={value!r}: expected an ISO 8601 timestamp such as "
            "2026-11-03T08:00:00Z"
        ) from error
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _get_datetime_env(var_name: str) -> Optional[datetime]:
    """Read an optional ISO 8601 env var; empty or unset means None."""
    raw_value = get_env_variable(var_name, default="").strip()
    if not raw_value:
        return None
    return parse_timestamp(raw_value, var_name)


def _get_reviewer_ids() -> set[str]:
    """Read which providers this watcher runs from HCMS_MODERATION_PROVIDERS.

    Comma-separated provider names (default: `DEFAULT_MODERATION_PROVIDERS`). Run
    one watcher per provider when they need different hosts, e.g. Llama Guard
    and the LLM judge next to their models.
    """
    raw_value = get_env_variable(
        "HCMS_MODERATION_PROVIDERS", default=",".join(DEFAULT_MODERATION_PROVIDERS)
    )
    providers = {name.strip() for name in raw_value.split(",") if name.strip()}
    unknown = providers - MODERATION_PROVIDERS.keys()
    if not providers or unknown:
        raise ValueError(
            f"HCMS_MODERATION_PROVIDERS={raw_value!r}: expected a comma-separated "
            f"list of {sorted(MODERATION_PROVIDERS)}"
        )
    return {MODERATION_PROVIDERS[name] for name in providers}


def configure_logging() -> None:
    """Configure basic logging for long-running watcher runtime."""
    configured_level = get_env_variable("HCMS_LOG_LEVEL", default="INFO").upper()
    log_level = getattr(logging, configured_level, logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if configured_level not in logging.getLevelNamesMapping():
        logging.getLogger(__name__).warning(
            "Unknown HCMS_LOG_LEVEL=%s; using INFO", configured_level
        )


_configure_logging = configure_logging


def create_repository() -> MongoConversationRepository:
    """Connect to the conversations collection named by the environment."""
    return MongoConversationRepository(
        mongo_uri=get_env_variable("MONGODB_URI"),
        database_name=get_env_variable("MONGODB_DATABASE"),
        collection_name=get_env_variable("MONGODB_COLLECTION", default="Conversations"),
    )


def main() -> None:
    """Create repository/watcher and start processing change events."""
    configure_logging()

    repository = create_repository()

    reviewer_ids = _get_reviewer_ids()
    worker_threads = _get_int_env("HCMS_WORKER_THREADS", 4)
    logging.getLogger(__name__).info(
        "Moderation providers: %s; worker threads: %d",
        ", ".join(sorted(reviewer_ids)),
        worker_threads,
    )

    watcher = RealtimeConversationWatcher(
        repository=repository,
        reviewer_ids=reviewer_ids,
        backfill_batch_size=_get_int_env("MONGODB_BACKFILL_BATCH_SIZE", 200),
        change_stream_max_await_ms=_get_int_env(
            "MONGODB_CHANGE_STREAM_MAX_AWAIT_MS", 1000
        ),
        backfill_max_retries=_get_int_env("MONGODB_BACKFILL_MAX_RETRIES", 10),
        backfill_retry_sleep_seconds=_get_float_env(
            "MONGODB_BACKFILL_RETRY_SLEEP_SECONDS", 2.0
        ),
        reconnect_backoff_base_seconds=_get_float_env(
            "MONGODB_RECONNECT_BACKOFF_BASE_SECONDS", 1.0
        ),
        reconnect_backoff_max_seconds=_get_float_env(
            "MONGODB_RECONNECT_BACKOFF_MAX_SECONDS", 30.0
        ),
        reconnect_backoff_jitter_seconds=_get_float_env(
            "MONGODB_RECONNECT_BACKOFF_JITTER_SECONDS", 0.25
        ),
        backfill_since=_get_datetime_env("MONGODB_BACKFILL_SINCE"),
        live_max_events_per_cycle=_get_int_env("MONGODB_LIVE_MAX_EVENTS_PER_CYCLE", 3),
        live_max_seconds_per_cycle=_get_float_env(
            "MONGODB_LIVE_MAX_SECONDS_PER_CYCLE", 60.0
        ),
        retry_max_per_cycle=_get_int_env("MONGODB_RETRY_MAX_PER_CYCLE", 1),
        worker_threads=worker_threads,
        rating_lag_warn_seconds=_get_float_env("HCMS_RATING_LAG_WARN_SECONDS", 120.0),
    )

    try:
        watcher.run()
    finally:
        watcher.shutdown()
        repository.close()


if __name__ == "__main__":
    main()
