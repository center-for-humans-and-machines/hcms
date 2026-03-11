"""Tests for realtime watcher normalization and event handling."""

# pylint: disable=missing-function-docstring,missing-class-docstring
# pylint: disable=too-many-arguments,too-many-positional-arguments
# pylint: disable=use-implicit-booleaness-not-comparison,duplicate-code
# pylint: disable=protected-access

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
import os
import subprocess
import sys
from typing import Any, Iterable

from bson import ObjectId
import pytest
from pydantic import ValidationError

from hpms.database.models import ConversationDocument
from hpms.database.repository import (
    SYSTEM_LLAMA_REVIEWER_ID,
    SYSTEM_OPENAI_REVIEWER_ID,
    MessageBackfillTarget,
    MongoConversationRepository,
)
from hpms.monitoring.realtime_watcher import (
    RealtimeConversationWatcher,
    normalize_llama_guard_categories,
    normalize_openai_categories,
)


@dataclass
class _Call:
    document_id: Any
    message_index: int
    reviewer_id: str
    categories: list[str]
    category_other: str


@dataclass
class StubRepository(MongoConversationRepository):
    calls: list[_Call] = field(default_factory=list)
    backfill_batches: list[list[MessageBackfillTarget]] = field(default_factory=list)

    @staticmethod
    def is_message_eligible_for_system_reviewers(message: dict[str, Any]):
        return MongoConversationRepository.is_message_eligible_for_system_reviewers(
            message
        )

    @staticmethod
    def missing_system_reviewers(message: dict[str, Any]):
        return MongoConversationRepository.missing_system_reviewers(message)

    @staticmethod
    def validate_conversation_document(conversation: dict[str, Any]):
        try:
            return ConversationDocument.model_validate(conversation).model_dump(
                by_alias=True
            )
        except ValidationError:
            return None

    def get_backfill_targets(
        self,
        batch_size: int = 200,
        excluded_provider_keys: set[tuple[Any, int, str]] | None = None,
    ) -> list[MessageBackfillTarget]:
        _ = batch_size
        _ = excluded_provider_keys
        if not self.backfill_batches:
            return []
        return self.backfill_batches.pop(0)

    def upsert_system_reviewer_flag(
        self,
        document_id: Any,
        message_index: int,
        reviewer_id: str,
        categories: Iterable[str],
        category_other: str,
        created_at: datetime | None = None,
    ) -> None:
        _ = created_at
        self.calls.append(
            _Call(
                document_id=document_id,
                message_index=message_index,
                reviewer_id=reviewer_id,
                categories=list(categories),
                category_other=category_other,
            )
        )


@dataclass
class FilteringStubRepository(StubRepository):
    """In-memory repository that filters exhausted provider keys per fetch."""

    targets: list[MessageBackfillTarget] = field(default_factory=list)

    def get_backfill_targets(
        self,
        batch_size: int = 200,
        excluded_provider_keys: set[tuple[Any, int, str]] | None = None,
    ) -> list[MessageBackfillTarget]:
        excluded_provider_keys = excluded_provider_keys or set()
        eligible_targets: list[MessageBackfillTarget] = []

        for target in self.targets:
            remaining_reviewer_ids = {
                reviewer_id
                for reviewer_id in target.missing_reviewer_ids
                if (
                    target.document_id,
                    target.message_index,
                    reviewer_id,
                )
                not in excluded_provider_keys
            }
            if not remaining_reviewer_ids:
                continue

            eligible_targets.append(
                MessageBackfillTarget(
                    document_id=target.document_id,
                    message_index=target.message_index,
                    content=target.content,
                    missing_reviewer_ids=remaining_reviewer_ids,
                )
            )
            if len(eligible_targets) >= batch_size:
                break

        return eligible_targets

    def upsert_system_reviewer_flag(
        self,
        document_id: Any,
        message_index: int,
        reviewer_id: str,
        categories: Iterable[str],
        category_other: str,
        created_at: datetime | None = None,
    ) -> None:
        super().upsert_system_reviewer_flag(
            document_id=document_id,
            message_index=message_index,
            reviewer_id=reviewer_id,
            categories=categories,
            category_other=category_other,
            created_at=created_at,
        )

        updated_targets: list[MessageBackfillTarget] = []
        for target in self.targets:
            if (
                target.document_id != document_id
                or target.message_index != message_index
            ):
                updated_targets.append(target)
                continue

            remaining_reviewer_ids = set(target.missing_reviewer_ids)
            remaining_reviewer_ids.discard(reviewer_id)
            if remaining_reviewer_ids:
                updated_targets.append(
                    MessageBackfillTarget(
                        document_id=target.document_id,
                        message_index=target.message_index,
                        content=target.content,
                        missing_reviewer_ids=remaining_reviewer_ids,
                    )
                )

        self.targets = updated_targets


class FlakyWatchRepository(StubRepository):
    """Repository whose change stream fails once, then reconnects."""

    def __init__(self):
        super().__init__()
        self.calls_count = 0
        self.events: list[str] = []

    def watch(self, max_await_time_ms: int = 1000):
        _ = max_await_time_ms
        self.calls_count += 1

        @contextmanager
        def stream_context():
            if self.calls_count == 1:
                raise RuntimeError("stream down")
            self.events.append("watch-opened")
            yield iter(())
            raise KeyboardInterrupt()

        return stream_context()


def _message(
    content: str,
    reviewer_flags: list[dict[str, Any]],
    *,
    role: str = "assistant",
):
    return {
        "content": content,
        "role": role,
        "timestamp": "2026-03-02T00:00:00Z",
        "type": role,
        "flagged": False,
        "flagged_at": None,
        "flagged_by": None,
        "flag_category": None,
        "flag_other_reason": None,
        "user_flag": {
            "category": "",
            "category_other": "",
            "reviews": [],
        },
        "reviewer_flags": reviewer_flags,
        "duplicate_flags": [],
    }


def _conversation_document(
    document_id: ObjectId,
    conversation_id: str,
    messages: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "_id": document_id,
        "conversation_id": conversation_id,
        "participant_id": "p1",
        "model": "test-model",
        "experiment_id": "exp-1",
        "project_id": "2026_03_08",
        "created_at": "2026-03-02T00:00:00Z",
        "custom_system_message_id": None,
        "multi_rounds": True,
        "messages": messages,
        "opened_by": [],
        "assigned_messages": [],
        "reviewed_messages": [],
    }


def test_normalize_openai_categories_handles_flagged_and_safe_values():
    categories, other = normalize_openai_categories([0, "0", "hate", "violence"])
    assert categories == ["hate", "violence"]
    assert other == ""

    categories, other = normalize_openai_categories([0, "0", 0])
    assert categories == []
    assert other == ""


def test_normalize_llama_guard_categories_handles_safe_known_and_unknown():
    categories, other = normalize_llama_guard_categories("0")
    assert categories == []
    assert other == ""

    categories, other = normalize_llama_guard_categories("Hate")
    assert categories == ["Hate"]
    assert other == ""

    categories, other = normalize_llama_guard_categories("unexpected output")
    assert categories == []
    assert other == "unexpected output"


def test_extract_candidate_message_indexes_from_update_paths():
    event = {
        "operationType": "update",
        "updateDescription": {
            "updatedFields": {
                "messages.0.content": "new",
                "messages.2.reviewer_flags": [],
                "participant_id": "p2",
            },
            "removedFields": ["messages.1.user_flag"],
        },
    }

    indexes = RealtimeConversationWatcher.extract_candidate_message_indexes(
        event, [{}, {}, {}]
    )
    assert indexes == {0, 1, 2}


def test_process_message_target_persists_both_provider_flags_when_missing():
    repository = StubRepository()

    watcher = RealtimeConversationWatcher(
        repository=repository,
        openai_rater=lambda _text: [0, "violence"],
        llama_guard_rater=lambda _text: "Hate",
    )

    document_id = ObjectId()
    target = MessageBackfillTarget(
        document_id=document_id,
        message_index=2,
        content="test",
        missing_reviewer_ids={
            SYSTEM_OPENAI_REVIEWER_ID,
            SYSTEM_LLAMA_REVIEWER_ID,
        },
    )

    watcher.process_message_target(target)

    assert len(repository.calls) == 2

    by_reviewer_id = {call.reviewer_id: call for call in repository.calls}

    assert by_reviewer_id[SYSTEM_OPENAI_REVIEWER_ID].document_id == document_id
    assert by_reviewer_id[SYSTEM_OPENAI_REVIEWER_ID].categories == ["violence"]
    assert by_reviewer_id[SYSTEM_OPENAI_REVIEWER_ID].category_other == ""

    assert by_reviewer_id[SYSTEM_LLAMA_REVIEWER_ID].categories == ["Hate"]
    assert by_reviewer_id[SYSTEM_LLAMA_REVIEWER_ID].category_other == ""


def test_handle_change_event_only_processes_messages_missing_system_flags():
    repository = StubRepository()

    watcher = RealtimeConversationWatcher(
        repository=repository,
        openai_rater=lambda _text: [0, "hate"],
        llama_guard_rater=lambda _text: "0",
    )

    document_id = ObjectId()
    event = {
        "operationType": "insert",
        "fullDocument": _conversation_document(
            document_id,
            "conversation-2",
            [
                _message(
                    "already rated",
                    [
                        {
                            "reviewer_id": SYSTEM_OPENAI_REVIEWER_ID,
                            "reviewer_by_username": SYSTEM_OPENAI_REVIEWER_ID,
                            "created_at": "2026-03-02T00:00:00Z",
                            "categories": [],
                            "category_other": "",
                            "comment": "",
                        },
                        {
                            "reviewer_id": SYSTEM_LLAMA_REVIEWER_ID,
                            "reviewer_by_username": SYSTEM_LLAMA_REVIEWER_ID,
                            "created_at": "2026-03-02T00:00:00Z",
                            "categories": [],
                            "category_other": "",
                            "comment": "",
                        },
                    ],
                ),
                _message("needs rating", []),
            ],
        ),
    }

    watcher.handle_change_event(event)

    assert len(repository.calls) == 2
    assert all(call.document_id == document_id for call in repository.calls)
    assert all(call.message_index == 1 for call in repository.calls)


@pytest.mark.parametrize(
    ("operation_type", "update_description"),
    [
        ("insert", None),
        (
            "update",
            {
                "updatedFields": {"messages.0.content": "updated system prompt"},
                "removedFields": [],
            },
        ),
    ],
)
def test_handle_change_event_skips_system_role_messages(
    operation_type: str, update_description: dict[str, Any] | None
):
    repository = StubRepository()

    watcher = RealtimeConversationWatcher(
        repository=repository,
        openai_rater=lambda _text: [0, "hate"],
        llama_guard_rater=lambda _text: "0",
    )

    document_id = ObjectId()
    event: dict[str, Any] = {
        "operationType": operation_type,
        "fullDocument": _conversation_document(
            document_id,
            "conversation-system-only",
            [_message("system prompt", [], role="system")],
        ),
    }
    if update_description is not None:
        event["updateDescription"] = update_description

    watcher.handle_change_event(event)

    assert repository.calls == []


def test_handle_change_event_processes_only_non_system_messages_in_mixed_conversation():
    repository = StubRepository()

    watcher = RealtimeConversationWatcher(
        repository=repository,
        openai_rater=lambda _text: [0, "hate"],
        llama_guard_rater=lambda _text: "0",
    )

    document_id = ObjectId()
    event = {
        "operationType": "insert",
        "fullDocument": _conversation_document(
            document_id,
            "conversation-mixed-roles",
            [
                _message("system prompt", [], role="system"),
                _message("needs rating", [], role="user"),
            ],
        ),
    }

    watcher.handle_change_event(event)

    assert len(repository.calls) == 2
    assert all(call.document_id == document_id for call in repository.calls)
    assert all(call.message_index == 1 for call in repository.calls)


def test_handle_change_event_skips_invalid_document_from_change_stream():
    repository = StubRepository()

    watcher = RealtimeConversationWatcher(
        repository=repository,
        openai_rater=lambda _text: [0, "hate"],
        llama_guard_rater=lambda _text: "0",
    )

    document = _conversation_document(
        ObjectId(), "conversation-invalid", [_message("x", [])]
    )
    del document["assigned_messages"]

    watcher.handle_change_event({"operationType": "insert", "fullDocument": document})

    assert repository.calls == []


def test_run_startup_backfill_processes_targets_until_empty():
    document_id = ObjectId()
    repository = StubRepository(
        backfill_batches=[
            [
                MessageBackfillTarget(
                    document_id=document_id,
                    message_index=0,
                    content="one",
                    missing_reviewer_ids={
                        SYSTEM_OPENAI_REVIEWER_ID,
                        SYSTEM_LLAMA_REVIEWER_ID,
                    },
                )
            ],
            [],
        ]
    )

    watcher = RealtimeConversationWatcher(
        repository=repository,
        openai_rater=lambda _text: [0, "hate"],
        llama_guard_rater=lambda _text: "0",
    )

    watcher.run_startup_backfill()

    assert len(repository.calls) == 2
    assert {call.reviewer_id for call in repository.calls} == {
        SYSTEM_OPENAI_REVIEWER_ID,
        SYSTEM_LLAMA_REVIEWER_ID,
    }


def test_run_startup_backfill_retries_transient_provider_failure():
    openai_calls = {"count": 0}

    def flaky_openai_rater(_text: str):
        openai_calls["count"] += 1
        if openai_calls["count"] == 1:
            raise RuntimeError("temporary failure")
        return [0, "hate"]

    document_id = ObjectId()
    repository = StubRepository(
        backfill_batches=[
            [
                MessageBackfillTarget(
                    document_id=document_id,
                    message_index=0,
                    content="one",
                    missing_reviewer_ids={
                        SYSTEM_OPENAI_REVIEWER_ID,
                        SYSTEM_LLAMA_REVIEWER_ID,
                    },
                )
            ],
            [
                MessageBackfillTarget(
                    document_id=document_id,
                    message_index=0,
                    content="one",
                    missing_reviewer_ids={SYSTEM_OPENAI_REVIEWER_ID},
                )
            ],
            [],
        ]
    )

    watcher = RealtimeConversationWatcher(
        repository=repository,
        openai_rater=flaky_openai_rater,
        llama_guard_rater=lambda _text: "0",
        backfill_retry_sleep_seconds=0.0,
    )

    watcher.run_startup_backfill()

    openai_writes = [
        call
        for call in repository.calls
        if call.reviewer_id == SYSTEM_OPENAI_REVIEWER_ID
    ]
    llama_writes = [
        call
        for call in repository.calls
        if call.reviewer_id == SYSTEM_LLAMA_REVIEWER_ID
    ]

    assert openai_calls["count"] == 2
    assert len(openai_writes) == 1
    assert len(llama_writes) == 1


def test_run_startup_backfill_stops_after_retry_budget_exhausted():
    def always_failing_openai(_text: str):
        raise RuntimeError("always fail")

    document_id = ObjectId()
    target = MessageBackfillTarget(
        document_id=document_id,
        message_index=0,
        content="one",
        missing_reviewer_ids={SYSTEM_OPENAI_REVIEWER_ID},
    )
    repository = StubRepository(
        backfill_batches=[
            [target],
            [target],
            [target],
        ]
    )

    watcher = RealtimeConversationWatcher(
        repository=repository,
        openai_rater=always_failing_openai,
        llama_guard_rater=lambda _text: "0",
        backfill_max_retries=2,
        backfill_retry_sleep_seconds=0.0,
    )

    watcher.run_startup_backfill()

    assert len(repository.calls) == 0
    assert watcher._unresolved_backfill_targets == 1  # pylint: disable=protected-access
    assert len(repository.backfill_batches) == 0


def test_run_startup_backfill_skips_exhausted_keys_while_progress_continues():
    openai_calls = {"count": 0}
    failing_document_id = ObjectId()

    def always_failing_openai(_text: str):
        openai_calls["count"] += 1
        raise RuntimeError("always fail")

    repository = StubRepository(
        backfill_batches=[
            [
                MessageBackfillTarget(
                    document_id=failing_document_id,
                    message_index=0,
                    content="fail",
                    missing_reviewer_ids={SYSTEM_OPENAI_REVIEWER_ID},
                ),
                MessageBackfillTarget(
                    document_id=ObjectId(),
                    message_index=0,
                    content="ok",
                    missing_reviewer_ids={SYSTEM_LLAMA_REVIEWER_ID},
                ),
            ],
            [
                MessageBackfillTarget(
                    document_id=failing_document_id,
                    message_index=0,
                    content="fail",
                    missing_reviewer_ids={SYSTEM_OPENAI_REVIEWER_ID},
                ),
                MessageBackfillTarget(
                    document_id=ObjectId(),
                    message_index=0,
                    content="ok",
                    missing_reviewer_ids={SYSTEM_LLAMA_REVIEWER_ID},
                ),
            ],
            [
                MessageBackfillTarget(
                    document_id=failing_document_id,
                    message_index=0,
                    content="fail",
                    missing_reviewer_ids={SYSTEM_OPENAI_REVIEWER_ID},
                ),
                MessageBackfillTarget(
                    document_id=ObjectId(),
                    message_index=0,
                    content="ok",
                    missing_reviewer_ids={SYSTEM_LLAMA_REVIEWER_ID},
                ),
            ],
            [],
        ]
    )

    watcher = RealtimeConversationWatcher(
        repository=repository,
        openai_rater=always_failing_openai,
        llama_guard_rater=lambda _text: "0",
        backfill_max_retries=2,
        backfill_retry_sleep_seconds=0.0,
    )

    watcher.run_startup_backfill()

    assert openai_calls["count"] == 2
    assert watcher._unresolved_backfill_targets == 1  # pylint: disable=protected-access
    llama_calls = [
        call
        for call in repository.calls
        if call.reviewer_id == SYSTEM_LLAMA_REVIEWER_ID
    ]
    assert len(llama_calls) == 3


def test_run_startup_backfill_batch_size_one_processes_later_targets_after_exhaustion():
    openai_calls = {"count": 0}

    def always_failing_openai(_text: str):
        openai_calls["count"] += 1
        raise RuntimeError("always fail")

    fail_document_id = ObjectId()
    ok_document_id = ObjectId()
    repository = FilteringStubRepository(
        targets=[
            MessageBackfillTarget(
                document_id=fail_document_id,
                message_index=0,
                content="fail",
                missing_reviewer_ids={SYSTEM_OPENAI_REVIEWER_ID},
            ),
            MessageBackfillTarget(
                document_id=ok_document_id,
                message_index=1,
                content="ok",
                missing_reviewer_ids={SYSTEM_LLAMA_REVIEWER_ID},
            ),
        ]
    )

    watcher = RealtimeConversationWatcher(
        repository=repository,
        openai_rater=always_failing_openai,
        llama_guard_rater=lambda _text: "0",
        backfill_batch_size=1,
        backfill_max_retries=1,
        backfill_retry_sleep_seconds=0.0,
    )

    watcher.run_startup_backfill()

    assert openai_calls["count"] == 1
    assert watcher._unresolved_backfill_targets == 1  # pylint: disable=protected-access
    assert any(
        call.document_id == ok_document_id
        and call.reviewer_id == SYSTEM_LLAMA_REVIEWER_ID
        for call in repository.calls
    )


def test_failure_is_retried_on_later_attempt():
    repository = StubRepository()
    openai_calls = {"count": 0}

    def flaky_openai_rater(_text: str):
        openai_calls["count"] += 1
        if openai_calls["count"] == 1:
            raise RuntimeError("temporary failure")
        return [0, "violence"]

    watcher = RealtimeConversationWatcher(
        repository=repository,
        openai_rater=flaky_openai_rater,
        llama_guard_rater=lambda _text: "0",
    )

    document_id = ObjectId()
    target = MessageBackfillTarget(
        document_id=document_id,
        message_index=1,
        content="retry me",
        missing_reviewer_ids={
            SYSTEM_OPENAI_REVIEWER_ID,
            SYSTEM_LLAMA_REVIEWER_ID,
        },
    )

    watcher.process_message_target(target)
    watcher.process_message_target(target)

    openai_writes = [
        call
        for call in repository.calls
        if call.reviewer_id == SYSTEM_OPENAI_REVIEWER_ID
    ]
    llama_writes = [
        call
        for call in repository.calls
        if call.reviewer_id == SYSTEM_LLAMA_REVIEWER_ID
    ]
    assert len(openai_writes) == 1
    assert openai_writes[0].document_id == document_id
    assert openai_writes[0].categories == ["violence"]
    assert len(llama_writes) == 2


def test_compute_reconnect_sleep_seconds_grows_and_caps(monkeypatch):
    monkeypatch.setattr(
        "hpms.monitoring.realtime_watcher.random.uniform", lambda *_: 0.0
    )

    watcher = RealtimeConversationWatcher(
        repository=StubRepository(),
        openai_rater=lambda _text: [0],
        llama_guard_rater=lambda _text: "0",
        reconnect_backoff_base_seconds=1.0,
        reconnect_backoff_max_seconds=4.0,
        reconnect_backoff_jitter_seconds=0.0,
    )

    assert watcher._compute_reconnect_sleep_seconds(1) == 1.0  # pylint: disable=protected-access
    assert watcher._compute_reconnect_sleep_seconds(2) == 2.0  # pylint: disable=protected-access
    assert watcher._compute_reconnect_sleep_seconds(3) == 4.0  # pylint: disable=protected-access
    assert watcher._compute_reconnect_sleep_seconds(8) == 4.0  # pylint: disable=protected-access


def test_compute_reconnect_sleep_seconds_large_failure_number_is_capped(monkeypatch):
    monkeypatch.setattr(
        "hpms.monitoring.realtime_watcher.random.uniform", lambda *_: 0.0
    )

    watcher = RealtimeConversationWatcher(
        repository=StubRepository(),
        openai_rater=lambda _text: [0],
        llama_guard_rater=lambda _text: "0",
        reconnect_backoff_base_seconds=1.0,
        reconnect_backoff_max_seconds=30.0,
        reconnect_backoff_jitter_seconds=0.0,
    )

    assert watcher._compute_reconnect_sleep_seconds(5000) == 30.0  # pylint: disable=protected-access


def test_monitoring_package_import_does_not_require_moderation_credentials():
    env = os.environ.copy()
    env.pop("OPENAI_MODERATION_API_KEY", None)
    env.pop("LLAMA_GUARD_API_KEY", None)
    env.pop("LLAMA_GUARD_ENDPOINT", None)

    command = (
        "import hpms.monitoring\n"
        "assert hasattr(hpms.monitoring, 'RateMessagesProcessor')\n"
        "from hpms.monitoring import RealtimeConversationWatcher\n"
        "assert RealtimeConversationWatcher.__name__ == 'RealtimeConversationWatcher'\n"
        "print('ok')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", command],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_run_reconnects_after_stream_failure(monkeypatch):
    sleeps: list[float] = []
    repository = FlakyWatchRepository()
    watcher = RealtimeConversationWatcher(
        repository=repository,
        openai_rater=lambda _text: [0],
        llama_guard_rater=lambda _text: "0",
        reconnect_backoff_base_seconds=1.0,
        reconnect_backoff_max_seconds=1.0,
        reconnect_backoff_jitter_seconds=0.0,
    )

    monkeypatch.setattr("hpms.monitoring.realtime_watcher.time.sleep", sleeps.append)

    with pytest.raises(KeyboardInterrupt):
        watcher.run()

    assert sleeps == [1.0]
    assert repository.events == ["watch-opened"]
