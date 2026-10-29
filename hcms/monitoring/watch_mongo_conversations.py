"""Run the hcms realtime MongoDB conversation watcher."""

# pylint: disable=duplicate-code

import logging

from hcms.database.repository import (
    SYSTEM_LLAMA_REVIEWER_ID,
    SYSTEM_OPENAI_REVIEWER_ID,
    MongoConversationRepository,
)
from hcms.monitoring.realtime_watcher import RealtimeConversationWatcher
from hcms.utils import get_env_variable

# Values accepted in HCMS_MODERATION_PROVIDERS, mapped to their reviewer IDs.
MODERATION_PROVIDERS = {
    "openai_moderation": SYSTEM_OPENAI_REVIEWER_ID,
    "llama_guard": SYSTEM_LLAMA_REVIEWER_ID,
}


def _get_int_env(var_name: str, default_value: int) -> int:
    """Read integer env var with a fallback default."""
    raw_value = get_env_variable(var_name, default=str(default_value))
    return int(raw_value)


def _get_float_env(var_name: str, default_value: float) -> float:
    """Read float env var with a fallback default."""
    raw_value = get_env_variable(var_name, default=str(default_value))
    return float(raw_value)


def _get_reviewer_ids() -> set[str]:
    """Read which providers this watcher runs from HCMS_MODERATION_PROVIDERS.

    Comma-separated provider names (default: all). Run one watcher per provider
    when they need different hosts, e.g. Llama Guard next to its GPU.
    """
    raw_value = get_env_variable(
        "HCMS_MODERATION_PROVIDERS", default=",".join(MODERATION_PROVIDERS)
    )
    providers = {name.strip() for name in raw_value.split(",") if name.strip()}
    unknown = providers - MODERATION_PROVIDERS.keys()
    if not providers or unknown:
        raise ValueError(
            f"HCMS_MODERATION_PROVIDERS={raw_value!r}: expected a comma-separated "
            f"list of {sorted(MODERATION_PROVIDERS)}"
        )
    return {MODERATION_PROVIDERS[name] for name in providers}


def _configure_logging() -> None:
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


def main() -> None:
    """Create repository/watcher and start processing change events."""
    _configure_logging()

    repository = MongoConversationRepository(
        mongo_uri=get_env_variable("MONGODB_URI"),
        database_name=get_env_variable("MONGODB_DATABASE"),
        collection_name=get_env_variable("MONGODB_COLLECTION", default="Conversations"),
    )

    reviewer_ids = _get_reviewer_ids()
    logging.getLogger(__name__).info(
        "Moderation providers: %s", ", ".join(sorted(reviewer_ids))
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
    )

    try:
        watcher.run()
    finally:
        repository.close()


if __name__ == "__main__":
    main()
