"""Tests for the realtime watcher's lag reporting and worker threads."""

# pylint: disable=missing-function-docstring,protected-access,duplicate-code

from __future__ import annotations

import logging
import threading
from datetime import datetime, timedelta, timezone

from bson import ObjectId

from hcms.database.repository import SYSTEM_OPENAI_REVIEWER_ID, MessageBackfillTarget
from hcms.monitoring.realtime_watcher import RealtimeConversationWatcher
from tests.test_realtime_watcher import (
    StubRepository,
    _conversation_document,
    _message,
)

WATCHER_LOGGER = "hcms.monitoring.realtime_watcher"


def _stale_message(content: str, age_seconds: float, *, role: str = "assistant"):
    message = _message(content, [], role=role)
    message["timestamp"] = datetime.now(timezone.utc) - timedelta(seconds=age_seconds)
    return message


def _insert_event(conversation_id: str, messages: list[dict]) -> dict:
    return {
        "operationType": "insert",
        "fullDocument": _conversation_document(ObjectId(), conversation_id, messages),
    }


def test_rating_lag_is_recorded_and_warned_for_late_realtime_ratings(caplog):
    watcher = RealtimeConversationWatcher(
        repository=StubRepository(),
        openai_rater=lambda _text: [0],
        reviewer_ids={SYSTEM_OPENAI_REVIEWER_ID},
        rating_lag_warn_seconds=60.0,
    )
    event = _insert_event(
        "conversation-lag",
        [_stale_message("late", 300.0), _stale_message("fresh", 1.0)],
    )

    with caplog.at_level(logging.WARNING, logger=WATCHER_LOGGER):
        watcher.handle_change_event(event)

    lag_warnings = [
        record for record in caplog.records if record.message.startswith("Rating lag")
    ]
    assert len(lag_warnings) == 1
    assert "message_index=0" in lag_warnings[0].message
    assert watcher._lag_count == 2
    assert watcher._lag_max_seconds >= 300.0


def test_startup_backfill_counts_lag_without_warning(caplog):
    target = MessageBackfillTarget(
        document_id=ObjectId(),
        message_index=0,
        content="old",
        missing_reviewer_ids={SYSTEM_OPENAI_REVIEWER_ID},
        message_timestamp=datetime.now(timezone.utc) - timedelta(hours=1),
    )
    watcher = RealtimeConversationWatcher(
        repository=StubRepository(backfill_batches=[[target], []]),
        openai_rater=lambda _text: [0],
        reviewer_ids={SYSTEM_OPENAI_REVIEWER_ID},
        rating_lag_warn_seconds=60.0,
    )

    with caplog.at_level(logging.WARNING, logger=WATCHER_LOGGER):
        watcher.run_startup_backfill()

    assert not [r for r in caplog.records if r.message.startswith("Rating lag")]
    assert watcher._lag_count == 1
    assert watcher._lag_max_seconds >= 3600.0
    assert not watcher._suppress_lag_warnings


def test_summary_reports_lag_per_window(caplog):
    watcher = RealtimeConversationWatcher(
        repository=StubRepository(),
        openai_rater=lambda _text: [0],
        reviewer_ids={SYSTEM_OPENAI_REVIEWER_ID},
        rating_lag_warn_seconds=0.0,
        summary_log_interval_events=1,
    )
    event = _insert_event("conversation-summary", [_stale_message("one", 10.0)])

    with caplog.at_level(logging.INFO, logger=WATCHER_LOGGER):
        watcher.handle_change_event(event)

    summaries = [r.message for r in caplog.records if "Watcher summary" in r.message]
    assert len(summaries) == 1
    assert "rated=1" in summaries[0]
    assert "max_lag=1" in summaries[0]  # about 10 s
    assert watcher._lag_count == 0  # window reset after the summary


def test_worker_threads_rate_messages_of_one_event_concurrently():
    repository = StubRepository()
    barrier = threading.Barrier(3, timeout=5)

    def openai_rater(_text: str):
        barrier.wait()  # only passes if three ratings run at the same time
        return [0]

    watcher = RealtimeConversationWatcher(
        repository=repository,
        openai_rater=openai_rater,
        reviewer_ids={SYSTEM_OPENAI_REVIEWER_ID},
        worker_threads=3,
    )
    event = _insert_event(
        "conversation-threads",
        [_message("a", [], role="user"), _message("b", []), _message("c", [])],
    )

    watcher.handle_change_event(event)
    watcher.wait_for_pending(timeout=5)
    watcher.shutdown()

    assert sorted(call.message_index for call in repository.calls) == [0, 1, 2]
    assert not watcher._in_flight


def test_worker_threads_skip_message_already_in_flight():
    repository = StubRepository()
    started = threading.Event()
    release = threading.Event()
    rater_calls: list[str] = []

    def openai_rater(text: str):
        rater_calls.append(text)
        started.set()
        assert release.wait(timeout=5)
        return [0]

    watcher = RealtimeConversationWatcher(
        repository=repository,
        openai_rater=openai_rater,
        reviewer_ids={SYSTEM_OPENAI_REVIEWER_ID},
        worker_threads=2,
    )
    event = _insert_event("conversation-in-flight", [_message("once", [])])

    watcher.handle_change_event(event)
    assert started.wait(timeout=5)
    # The same message arrives again (e.g. another provider's write) while its
    # first rating is still running.
    watcher.handle_change_event(event)
    release.set()
    watcher.wait_for_pending(timeout=5)
    watcher.shutdown()

    assert rater_calls == ["once"]
    assert len(repository.calls) == 1


def test_startup_backfill_rates_batch_on_worker_threads():
    document_id = ObjectId()
    barrier = threading.Barrier(2, timeout=5)

    def openai_rater(_text: str):
        barrier.wait()
        return [0]

    repository = StubRepository(
        backfill_batches=[
            [
                MessageBackfillTarget(
                    document_id=document_id,
                    message_index=index,
                    content=f"message {index}",
                    missing_reviewer_ids={SYSTEM_OPENAI_REVIEWER_ID},
                )
                for index in range(2)
            ],
            [],
        ]
    )
    watcher = RealtimeConversationWatcher(
        repository=repository,
        openai_rater=openai_rater,
        reviewer_ids={SYSTEM_OPENAI_REVIEWER_ID},
        worker_threads=2,
    )

    watcher.run_startup_backfill()
    watcher.shutdown()

    assert sorted(call.message_index for call in repository.calls) == [0, 1]
    assert watcher._unresolved_backfill_targets == 0


def test_shutdown_without_worker_threads_is_a_no_op():
    watcher = RealtimeConversationWatcher(
        repository=StubRepository(),
        openai_rater=lambda _text: [0],
        reviewer_ids={SYSTEM_OPENAI_REVIEWER_ID},
    )

    watcher.wait_for_pending(timeout=0)
    watcher.shutdown()
