"""Run the HPMS realtime MongoDB conversation watcher."""

# pylint: disable=duplicate-code

import logging

from hpms.database.repository import MongoConversationRepository
from hpms.monitoring.realtime_watcher import RealtimeConversationWatcher
from hpms.utils import get_env_variable


def _get_int_env(var_name: str, default_value: int) -> int:
    """Read integer env var with a fallback default."""
    raw_value = get_env_variable(var_name, default=str(default_value))
    return int(raw_value)


def _get_float_env(var_name: str, default_value: float) -> float:
    """Read float env var with a fallback default."""
    raw_value = get_env_variable(var_name, default=str(default_value))
    return float(raw_value)


def _configure_logging() -> None:
    """Configure basic logging for long-running watcher runtime."""
    configured_level = get_env_variable("HPMS_LOG_LEVEL", default="INFO").upper()
    log_level = getattr(logging, configured_level, logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if configured_level not in logging.getLevelNamesMapping():
        logging.getLogger(__name__).warning(
            "Unknown HPMS_LOG_LEVEL=%s; using INFO", configured_level
        )


def main() -> None:
    """Create repository/watcher and start processing change events."""
    _configure_logging()

    repository = MongoConversationRepository(
        mongo_uri=get_env_variable("MONGODB_URI"),
        database_name=get_env_variable("MONGODB_DATABASE"),
        collection_name=get_env_variable("MONGODB_COLLECTION", default="Conversations"),
    )

    watcher = RealtimeConversationWatcher(
        repository=repository,
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
