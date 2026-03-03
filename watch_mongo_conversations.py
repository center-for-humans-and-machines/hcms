"""Run the HPMS realtime MongoDB conversation watcher."""

# pylint: disable=duplicate-code

from hpms.database.repository import MongoConversationRepository
from hpms.monitoring.realtime_watcher import RealtimeConversationWatcher
from hpms.utils import get_env_variable


def _get_int_env(var_name: str, default_value: int) -> int:
    """Read integer env var with a fallback default."""
    raw_value = get_env_variable(var_name, default=str(default_value))
    return int(raw_value)


def main() -> None:
    """Create repository/watcher and start processing change events."""
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
    )

    try:
        watcher.run()
    finally:
        repository.close()


if __name__ == "__main__":
    main()
