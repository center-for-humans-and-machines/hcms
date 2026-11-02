"""Tests for realtime watcher normalization and event handling."""

# pylint: disable=missing-function-docstring,missing-class-docstring
# pylint: disable=too-many-arguments,too-many-positional-arguments
# pylint: disable=use-implicit-booleaness-not-comparison,duplicate-code
# pylint: disable=protected-access,too-many-lines

from __future__ import annotations

import os
import subprocess
import sys
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable

import pytest
from bson import ObjectId
from pydantic import ValidationError

from hcms.database.models import ConversationDocument
from hcms.database.repository import (
    SYSTEM_LLAMA_REVIEWER_ID,
    SYSTEM_LLM_JUDGE_REVIEWER_ID,
    SYSTEM_OPENAI_REVIEWER_ID,
    BackfillBatch,
    MessageBackfillTarget,
    MongoConversationRepository,
)
from hcms.monitoring.realtime_watcher import (
    MAX_COMMENT_LENGTH,
    RealtimeConversationWatcher,
    normalize_ktb_judge_verdict,
    normalize_llama_guard_categories,
    normalize_openai_categories,
)
from hcms.rating.ktb_judge import KtbJudgeVerdict


@dataclass
class _Call:
    document_id: Any
    message_index: int
    reviewer_id: str
    categories: list[str]
    category_other: str
    comment: str = ""
    expected_content: str | None = None


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
    def missing_system_reviewers(
        message: dict[str, Any], reviewer_ids: set[str] | None = None
    ):
        return MongoConversationRepository.missing_system_reviewers(
            message, reviewer_ids
        )

    @staticmethod
    def validate_conversation_document(conversation: dict[str, Any]):
        try:
            return ConversationDocument.model_validate(conversation).model_dump(
                by_alias=True
            )
        except ValidationError:
            return None

    def next_backfill_batch(
        self,
        batch_size: int = 200,
        excluded_provider_keys: set[tuple[Any, int, str]] | None = None,
        reviewer_ids: set[str] | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        after_id: Any = None,
    ) -> BackfillBatch:
        _ = (batch_size, excluded_provider_keys, reviewer_ids, since, until, after_id)
        if not self.backfill_batches:
            return BackfillBatch(targets=[], resume_after=None)
        return BackfillBatch(targets=self.backfill_batches.pop(0), resume_after=None)

    def upsert_system_reviewer_flag(
        self,
        document_id: Any,
        message_index: int,
        reviewer_id: str,
        categories: Iterable[str],
        category_other: str,
        created_at: datetime | None = None,
        comment: str = "",
        expected_content: str | None = None,
    ) -> bool:
        _ = created_at
        self.calls.append(
            _Call(
                document_id=document_id,
                message_index=message_index,
                reviewer_id=reviewer_id,
                categories=list(categories),
                category_other=category_other,
                comment=comment,
                expected_content=expected_content,
            )
        )
        return True


@dataclass
class FilteringStubRepository(StubRepository):
    """In-memory repository that filters exhausted provider keys per fetch."""

    targets: list[MessageBackfillTarget] = field(default_factory=list)

    def next_backfill_batch(
        self,
        batch_size: int = 200,
        excluded_provider_keys: set[tuple[Any, int, str]] | None = None,
        reviewer_ids: set[str] | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        after_id: Any = None,
    ) -> BackfillBatch:
        _ = (since, until, after_id)
        excluded_provider_keys = excluded_provider_keys or set()
        eligible_targets: list[MessageBackfillTarget] = []

        for target in self.targets:
            remaining_reviewer_ids = {
                reviewer_id
                for reviewer_id in target.missing_reviewer_ids
                if (reviewer_ids is None or reviewer_id in reviewer_ids)
                and (
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

        return BackfillBatch(targets=eligible_targets, resume_after=None)

    def upsert_system_reviewer_flag(
        self,
        document_id: Any,
        message_index: int,
        reviewer_id: str,
        categories: Iterable[str],
        category_other: str,
        created_at: datetime | None = None,
        comment: str = "",
        expected_content: str | None = None,
    ) -> bool:
        super().upsert_system_reviewer_flag(
            document_id=document_id,
            message_index=message_index,
            reviewer_id=reviewer_id,
            categories=categories,
            category_other=category_other,
            created_at=created_at,
            comment=comment,
            expected_content=expected_content,
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
        return True


class FakeChangeStream:  # pylint: disable=too-few-public-methods
    """Change stream stand-in: `try_next` hands out events, then idles."""

    def __init__(self, events: list[dict[str, Any]]):
        self.events = list(events)
        self.polls = 0
        self.alive = True

    def try_next(self) -> dict[str, Any] | None:
        self.polls += 1
        if self.events:
            return self.events.pop(0)
        return None


def _judge_verdict(*categories: str) -> KtbJudgeVerdict:
    return KtbJudgeVerdict(categories=list(categories), rationale="because")


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

    categories, other = normalize_llama_guard_categories("safe")
    assert categories == []
    assert other == ""

    categories, other = normalize_llama_guard_categories("Safe")
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
        llama_guard_rater=lambda _text, **_context: "Hate",
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
        llama_guard_rater=lambda _text, **_context: "0",
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
        llama_guard_rater=lambda _text, **_context: "0",
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
        llama_guard_rater=lambda _text, **_context: "0",
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
        llama_guard_rater=lambda _text, **_context: "0",
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
        llama_guard_rater=lambda _text, **_context: "0",
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
        llama_guard_rater=lambda _text, **_context: "0",
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
        llama_guard_rater=lambda _text, **_context: "0",
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
        llama_guard_rater=lambda _text, **_context: "0",
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
        llama_guard_rater=lambda _text, **_context: "0",
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
        llama_guard_rater=lambda _text, **_context: "0",
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
        "hcms.monitoring.realtime_watcher.random.uniform", lambda *_: 0.0
    )

    watcher = RealtimeConversationWatcher(
        repository=StubRepository(),
        openai_rater=lambda _text: [0],
        llama_guard_rater=lambda _text, **_context: "0",
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
        "hcms.monitoring.realtime_watcher.random.uniform", lambda *_: 0.0
    )

    watcher = RealtimeConversationWatcher(
        repository=StubRepository(),
        openai_rater=lambda _text: [0],
        llama_guard_rater=lambda _text, **_context: "0",
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
        "import hcms.monitoring\n"
        "assert hasattr(hcms.monitoring, 'RateMessagesProcessor')\n"
        "from hcms.monitoring import RealtimeConversationWatcher\n"
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
        llama_guard_rater=lambda _text, **_context: "0",
        reconnect_backoff_base_seconds=1.0,
        reconnect_backoff_max_seconds=1.0,
        reconnect_backoff_jitter_seconds=0.0,
    )

    monkeypatch.setattr("hcms.monitoring.realtime_watcher.time.sleep", sleeps.append)

    with pytest.raises(KeyboardInterrupt):
        watcher.run()

    assert sleeps == [1.0]
    assert repository.events == ["watch-opened"]


def test_watcher_with_one_provider_only_writes_that_provider():
    repository = StubRepository()
    watcher = RealtimeConversationWatcher(
        repository=repository,
        llama_guard_rater=lambda _text, **_context: "Hate",
        reviewer_ids={SYSTEM_LLAMA_REVIEWER_ID},
    )

    target = MessageBackfillTarget(
        document_id=ObjectId(),
        message_index=0,
        content="test",
        missing_reviewer_ids={SYSTEM_OPENAI_REVIEWER_ID, SYSTEM_LLAMA_REVIEWER_ID},
    )
    results = watcher.process_message_target(target)

    # The OpenAI rater was never resolved, so no OpenAI credentials are needed.
    assert watcher.openai_rater is None
    assert results == {SYSTEM_LLAMA_REVIEWER_ID: True}
    assert [call.reviewer_id for call in repository.calls] == [SYSTEM_LLAMA_REVIEWER_ID]


@pytest.mark.parametrize("reviewer_ids", [set(), {"system_unknown"}])
def test_watcher_rejects_invalid_reviewer_ids(reviewer_ids):
    with pytest.raises(ValueError, match="reviewer_ids"):
        RealtimeConversationWatcher(
            repository=StubRepository(),
            openai_rater=lambda _text: [0],
            llama_guard_rater=lambda _text, **_context: "0",
            reviewer_ids=reviewer_ids,
        )


def test_handle_change_event_ignores_other_providers_missing_entries():
    repository = StubRepository()
    watcher = RealtimeConversationWatcher(
        repository=repository,
        openai_rater=lambda _text: [0],
        reviewer_ids={SYSTEM_OPENAI_REVIEWER_ID},
    )
    openai_flag = {
        "reviewer_id": SYSTEM_OPENAI_REVIEWER_ID,
        "reviewer_by_username": SYSTEM_OPENAI_REVIEWER_ID,
        "created_at": "2026-03-02T00:00:00Z",
        "categories": [],
        "category_other": "",
        "comment": "",
    }
    event = {
        "operationType": "insert",
        "fullDocument": _conversation_document(
            ObjectId(),
            "conversation-openai-only",
            # Missing only the Llama Guard entry, which this watcher doesn't run.
            [_message("rated by OpenAI", [openai_flag])],
        ),
    }

    watcher.handle_change_event(event)

    assert repository.calls == []


def test_llama_guard_rates_assistant_reply_with_its_user_message():
    calls: list[tuple[str, dict[str, Any]]] = []

    def llama_guard_rater(text: str, **context: Any) -> str:
        calls.append((text, context))
        return "0"

    watcher = RealtimeConversationWatcher(
        repository=StubRepository(),
        llama_guard_rater=llama_guard_rater,
        reviewer_ids={SYSTEM_LLAMA_REVIEWER_ID},
    )
    event = {
        "operationType": "insert",
        "fullDocument": _conversation_document(
            ObjectId(),
            "conversation-reply-context",
            [
                _message("How do I pick a lock?", [], role="user"),
                _message("Use a tension wrench.", []),
            ],
        ),
    }

    watcher.handle_change_event(event)

    assert calls == [
        ("How do I pick a lock?", {}),
        (
            "Use a tension wrench.",
            {"role": "assistant", "previous_user_content": "How do I pick a lock?"},
        ),
    ]


def test_run_startup_backfill_ends_when_only_other_providers_are_missing():
    document_id = ObjectId()
    repository = FilteringStubRepository(
        targets=[
            MessageBackfillTarget(
                document_id=document_id,
                message_index=0,
                content="needs OpenAI only",
                missing_reviewer_ids={SYSTEM_OPENAI_REVIEWER_ID},
            )
        ]
    )
    watcher = RealtimeConversationWatcher(
        repository=repository,
        llama_guard_rater=lambda _text, **_context: "0",
        reviewer_ids={SYSTEM_LLAMA_REVIEWER_ID},
    )

    watcher.run_startup_backfill()

    assert repository.calls == []


def _insert_event(content: str = "needs rating", role: str = "user") -> dict[str, Any]:
    return {
        "operationType": "insert",
        "fullDocument": _conversation_document(
            ObjectId(), "conversation-live", [_message(content, [], role=role)]
        ),
    }


def _watcher_with_recording_judge(
    repository: StubRepository, **kwargs: Any
) -> tuple[RealtimeConversationWatcher, list[tuple[str, dict[str, Any]]]]:
    calls: list[tuple[str, dict[str, Any]]] = []

    def llm_judge_rater(text: str, **context: Any) -> KtbJudgeVerdict:
        calls.append((text, context))
        return _judge_verdict("Suicidal ideation")

    watcher = RealtimeConversationWatcher(
        repository=repository,
        llm_judge_rater=llm_judge_rater,
        reviewer_ids={SYSTEM_LLM_JUDGE_REVIEWER_ID},
        **kwargs,
    )
    return watcher, calls


def test_normalize_ktb_judge_verdict_uses_categories_and_rationale():
    verdict = KtbJudgeVerdict(
        categories=["Suicidal ideation", "Self-harm"],
        rationale="  states a wish to die ",
    )

    categories, other, comment = normalize_ktb_judge_verdict(verdict)

    assert categories == ["Suicidal ideation", "Self-harm"]
    assert other == ""
    assert comment == "states a wish to die"


def test_normalize_ktb_judge_verdict_caps_the_comment_length():
    verdict = KtbJudgeVerdict(categories=[], rationale="x" * (MAX_COMMENT_LENGTH + 50))

    _categories, _other, comment = normalize_ktb_judge_verdict(verdict)

    assert len(comment) == MAX_COMMENT_LENGTH


@pytest.mark.parametrize("raw", ["unsafe", {"categories": []}, None, 3])
def test_normalize_ktb_judge_verdict_rejects_anything_but_a_verdict(raw):
    # Malformed output must stay an inference failure: a non-empty
    # category_other would count as a safety finding on the dashboard.
    with pytest.raises(TypeError):
        normalize_ktb_judge_verdict(raw)


def test_process_message_target_persists_all_three_provider_flags():
    repository = StubRepository()
    watcher = RealtimeConversationWatcher(
        repository=repository,
        openai_rater=lambda _text: [0, "violence"],
        llama_guard_rater=lambda _text, **_context: "Hate",
        llm_judge_rater=lambda _text, **_context: _judge_verdict("Self-harm"),
        reviewer_ids={
            SYSTEM_OPENAI_REVIEWER_ID,
            SYSTEM_LLAMA_REVIEWER_ID,
            SYSTEM_LLM_JUDGE_REVIEWER_ID,
        },
    )
    target = MessageBackfillTarget(
        document_id=ObjectId(),
        message_index=0,
        content="test",
        missing_reviewer_ids={
            SYSTEM_OPENAI_REVIEWER_ID,
            SYSTEM_LLAMA_REVIEWER_ID,
            SYSTEM_LLM_JUDGE_REVIEWER_ID,
        },
        role="user",
    )

    results = watcher.process_message_target(target)

    assert results == {
        SYSTEM_OPENAI_REVIEWER_ID: True,
        SYSTEM_LLAMA_REVIEWER_ID: True,
        SYSTEM_LLM_JUDGE_REVIEWER_ID: True,
    }
    by_reviewer_id = {call.reviewer_id: call for call in repository.calls}
    assert by_reviewer_id[SYSTEM_LLM_JUDGE_REVIEWER_ID].categories == ["Self-harm"]
    assert by_reviewer_id[SYSTEM_LLM_JUDGE_REVIEWER_ID].category_other == ""
    assert by_reviewer_id[SYSTEM_LLM_JUDGE_REVIEWER_ID].comment == "because"
    assert by_reviewer_id[SYSTEM_LLAMA_REVIEWER_ID].comment == ""
    # Every write is conditional on the content that was rated.
    assert {call.expected_content for call in repository.calls} == {"test"}


def test_llm_judge_only_watcher_writes_only_its_entry():
    repository = StubRepository()
    watcher, _calls = _watcher_with_recording_judge(repository)
    target = MessageBackfillTarget(
        document_id=ObjectId(),
        message_index=0,
        content="I want to die",
        missing_reviewer_ids={
            SYSTEM_OPENAI_REVIEWER_ID,
            SYSTEM_LLAMA_REVIEWER_ID,
            SYSTEM_LLM_JUDGE_REVIEWER_ID,
        },
        role="user",
    )

    results = watcher.process_message_target(target)

    assert watcher.openai_rater is None
    assert watcher.llama_guard_rater is None
    assert results == {SYSTEM_LLM_JUDGE_REVIEWER_ID: True}
    assert [call.reviewer_id for call in repository.calls] == [
        SYSTEM_LLM_JUDGE_REVIEWER_ID
    ]


def test_default_providers_do_not_include_the_llm_judge():
    watcher = RealtimeConversationWatcher(
        repository=StubRepository(),
        openai_rater=lambda _text: [0],
        llama_guard_rater=lambda _text, **_context: "0",
    )

    assert watcher.reviewer_ids == {SYSTEM_OPENAI_REVIEWER_ID, SYSTEM_LLAMA_REVIEWER_ID}
    assert watcher.llm_judge_rater is None


def test_llm_judge_receives_role_and_context_turns_from_change_event():
    repository = StubRepository()
    watcher, calls = _watcher_with_recording_judge(repository)
    event = {
        "operationType": "insert",
        "fullDocument": _conversation_document(
            ObjectId(),
            "conversation-judge-context",
            [
                _message("system prompt", [], role="system"),
                _message("Hello", [], role="user"),
                _message("Hi, how are you?", []),
                _message("I want to die", [], role="user"),
            ],
        ),
    }

    watcher.handle_change_event(event)

    assert calls == [
        ("Hello", {"role": "user", "context_messages": ()}),
        (
            "Hi, how are you?",
            {"role": "assistant", "context_messages": (("user", "Hello"),)},
        ),
        (
            "I want to die",
            {
                "role": "user",
                "context_messages": (
                    ("user", "Hello"),
                    ("assistant", "Hi, how are you?"),
                ),
            },
        ),
    ]
    assert [call.message_index for call in repository.calls] == [1, 2, 3]


def test_invalid_judge_output_writes_nothing(caplog):
    repository = StubRepository()

    def rejecting_judge(_text: str, **_context: Any) -> KtbJudgeVerdict:
        return KtbJudgeVerdict.model_validate(
            {"categories": ["not a category"], "rationale": "x"}
        )

    watcher = RealtimeConversationWatcher(
        repository=repository,
        llm_judge_rater=rejecting_judge,
        reviewer_ids={SYSTEM_LLM_JUDGE_REVIEWER_ID},
    )

    with caplog.at_level("ERROR"):
        watcher.handle_change_event(_insert_event())

    assert repository.calls == []
    assert watcher._provider_failure_counts[SYSTEM_LLM_JUDGE_REVIEWER_ID] == 1
    assert "Failed LLM judge rating" in caplog.text
    assert len(watcher._retry_queue) == 1


def test_failed_live_rating_is_retried_without_a_new_event():
    repository = StubRepository()
    attempts = {"count": 0}

    def flaky_judge(_text: str, **_context: Any) -> KtbJudgeVerdict:
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RuntimeError("model busy")
        return _judge_verdict("Suicidal ideation")

    watcher = RealtimeConversationWatcher(
        repository=repository,
        llm_judge_rater=flaky_judge,
        reviewer_ids={SYSTEM_LLM_JUDGE_REVIEWER_ID},
        backfill_retry_sleep_seconds=0.0,
    )

    watcher.handle_change_event(_insert_event())
    assert repository.calls == []
    assert len(watcher._retry_queue) == 1

    watcher._run_due_retries()

    assert attempts["count"] == 2
    assert [call.categories for call in repository.calls] == [["Suicidal ideation"]]
    assert watcher._retry_queue == {}


def test_live_retries_stop_after_the_retry_budget(caplog):
    repository = StubRepository()

    def failing_judge(_text: str, **_context: Any) -> KtbJudgeVerdict:
        raise RuntimeError("always fails")

    watcher = RealtimeConversationWatcher(
        repository=repository,
        llm_judge_rater=failing_judge,
        reviewer_ids={SYSTEM_LLM_JUDGE_REVIEWER_ID},
        backfill_max_retries=2,
        backfill_retry_sleep_seconds=0.0,
    )

    watcher.handle_change_event(_insert_event())
    with caplog.at_level("ERROR"):
        watcher._run_due_retries()

    assert repository.calls == []
    assert watcher._retry_queue == {}
    assert watcher._unresolved_live_targets == 1
    assert "Retries exhausted" in caplog.text


def test_retry_waits_for_its_backoff_delay():
    repository = StubRepository()

    def failing_judge(_text: str, **_context: Any) -> KtbJudgeVerdict:
        raise RuntimeError("always fails")

    watcher = RealtimeConversationWatcher(
        repository=repository,
        llm_judge_rater=failing_judge,
        reviewer_ids={SYSTEM_LLM_JUDGE_REVIEWER_ID},
        backfill_retry_sleep_seconds=100.0,
    )

    watcher.handle_change_event(_insert_event())
    watcher._run_due_retries()

    retry_key = next(iter(watcher._retry_queue))
    assert watcher._retry_attempts[retry_key] == 1  # not due yet, so not retried


def test_content_change_event_rates_an_already_flagged_message_again():
    repository = StubRepository()
    watcher = RealtimeConversationWatcher(
        repository=repository,
        openai_rater=lambda _text: [0, "hate"],
        llama_guard_rater=lambda _text, **_context: "Hate",
    )
    rated = [
        {
            "reviewer_id": reviewer_id,
            "reviewer_by_username": reviewer_id,
            "created_at": "2026-03-02T00:00:00Z",
            "categories": [],
            "category_other": "",
            "comment": "",
        }
        for reviewer_id in (SYSTEM_OPENAI_REVIEWER_ID, SYSTEM_LLAMA_REVIEWER_ID)
    ]
    document_id = ObjectId()
    event = {
        "operationType": "update",
        "updateDescription": {
            "updatedFields": {"messages.1.content": "merged text"},
            "removedFields": [],
        },
        "fullDocument": _conversation_document(
            document_id,
            "conversation-merged",
            [
                _message("first", list(rated), role="user"),
                _message("merged text", list(rated), role="user"),
            ],
        ),
    }

    watcher.handle_change_event(event)

    assert sorted(call.reviewer_id for call in repository.calls) == sorted(
        [SYSTEM_OPENAI_REVIEWER_ID, SYSTEM_LLAMA_REVIEWER_ID]
    )
    assert {call.message_index for call in repository.calls} == {1}
    assert {call.expected_content for call in repository.calls} == {"merged text"}


def test_replaced_messages_array_only_rates_messages_without_flags():
    repository = StubRepository()
    watcher = RealtimeConversationWatcher(
        repository=repository,
        openai_rater=lambda _text: [0],
        llama_guard_rater=lambda _text, **_context: "0",
    )
    rated = [
        {
            "reviewer_id": reviewer_id,
            "reviewer_by_username": reviewer_id,
            "created_at": "2026-03-02T00:00:00Z",
            "categories": [],
            "category_other": "",
            "comment": "",
        }
        for reviewer_id in (SYSTEM_OPENAI_REVIEWER_ID, SYSTEM_LLAMA_REVIEWER_ID)
    ]
    event = {
        "operationType": "update",
        "updateDescription": {"updatedFields": {"messages": []}, "removedFields": []},
        "fullDocument": _conversation_document(
            ObjectId(),
            "conversation-replaced",
            [_message("rated", rated, role="user"), _message("new", [], role="user")],
        ),
    }

    watcher.handle_change_event(event)

    assert {call.message_index for call in repository.calls} == {1}


def test_cycle_bounds_live_events_then_serves_retries_and_one_backfill_batch():
    backfill_batches = [
        [
            MessageBackfillTarget(
                document_id=ObjectId(),
                message_index=0,
                content=f"backfill {index}",
                missing_reviewer_ids={SYSTEM_LLM_JUDGE_REVIEWER_ID},
                role="user",
            )
        ]
        for index in range(5)
    ]
    repository = StubRepository(backfill_batches=backfill_batches)
    watcher, calls = _watcher_with_recording_judge(
        repository,
        live_max_events_per_cycle=3,
        backfill_batch_size=1,
        backfill_retry_sleep_seconds=0.0,
    )
    retry_target = MessageBackfillTarget(
        document_id=ObjectId(),
        message_index=0,
        content="retry me",
        missing_reviewer_ids={SYSTEM_LLM_JUDGE_REVIEWER_ID},
        role="user",
    )
    watcher._schedule_retry(retry_target, {SYSTEM_LLM_JUDGE_REVIEWER_ID})
    stream = FakeChangeStream([_insert_event(f"live {index}") for index in range(10)])

    assert watcher._run_cycle(stream) is True

    rated = [text for text, _context in calls]
    # Three live events, then the due retry, then exactly one backfill batch,
    # even though seven live events are still waiting.
    assert rated == ["live 0", "live 1", "live 2", "retry me", "backfill 0"]
    assert len(stream.events) == 7
    assert len(repository.backfill_batches) == 4

    assert watcher._run_cycle(stream) is True

    rated = [text for text, _context in calls]
    assert rated[5:] == ["live 3", "live 4", "live 5", "backfill 1"]


def test_cycle_processes_backfill_when_no_live_event_is_pending():
    target = MessageBackfillTarget(
        document_id=ObjectId(),
        message_index=0,
        content="old message",
        missing_reviewer_ids={SYSTEM_LLM_JUDGE_REVIEWER_ID},
        role="user",
    )
    repository = StubRepository(backfill_batches=[[target], []])
    watcher, calls = _watcher_with_recording_judge(repository, backfill_batch_size=1)
    stream = FakeChangeStream([])

    assert watcher._run_cycle(stream) is True
    assert watcher._run_cycle(stream) is True

    assert [text for text, _context in calls] == ["old message"]
    assert watcher._backfill.complete is True
    assert stream.polls == 2  # one idle poll per cycle, nothing spins


def test_cycle_pauses_backfill_after_a_failed_batch():
    target = MessageBackfillTarget(
        document_id=ObjectId(),
        message_index=0,
        content="fails",
        missing_reviewer_ids={SYSTEM_LLM_JUDGE_REVIEWER_ID},
        role="user",
    )
    repository = StubRepository(backfill_batches=[[target], [target], []])

    def failing_judge(_text: str, **_context: Any) -> KtbJudgeVerdict:
        raise RuntimeError("model down")

    watcher = RealtimeConversationWatcher(
        repository=repository,
        llm_judge_rater=failing_judge,
        reviewer_ids={SYSTEM_LLM_JUDGE_REVIEWER_ID},
        backfill_batch_size=1,
        backfill_retry_sleep_seconds=100.0,
    )
    stream = FakeChangeStream([])

    watcher._run_cycle(stream)
    watcher._run_cycle(stream)

    # The second cycle did not ask for the same failing target again.
    assert len(repository.backfill_batches) == 2


def test_cycle_reports_a_closed_stream():
    watcher, _calls = _watcher_with_recording_judge(StubRepository())
    stream = FakeChangeStream([])
    stream.alive = False

    assert watcher._run_cycle(stream) is False


def test_summary_log_lists_every_enabled_provider(caplog):
    watcher = RealtimeConversationWatcher(
        repository=StubRepository(),
        openai_rater=lambda _text: [0],
        llama_guard_rater=lambda _text, **_context: "0",
        summary_log_interval_events=1,
    )

    with caplog.at_level("INFO"):
        watcher.handle_change_event(_insert_event())

    summary = next(
        record for record in caplog.records if "Watcher summary" in record.message
    )
    assert f"{SYSTEM_OPENAI_REVIEWER_ID}_success=1" in summary.message
    assert f"{SYSTEM_LLAMA_REVIEWER_ID}_success=1" in summary.message
    assert "retry_queue=0" in summary.message


class ContentGuardStubRepository(StubRepository):
    """Stub whose writes honour `expected_content` against the message's current text."""

    def __init__(self) -> None:
        super().__init__()
        self.current_content: dict[tuple[Any, int], str] = {}
        self.skipped: list[str] = []

    def upsert_system_reviewer_flag(self, *args: Any, **kwargs: Any) -> bool:
        expected = kwargs.get("expected_content")
        key = (kwargs["document_id"], kwargs["message_index"])
        if expected is not None and self.current_content.get(key) != expected:
            self.skipped.append(expected)
            return False
        return super().upsert_system_reviewer_flag(*args, **kwargs)


def test_content_change_during_in_flight_rating_rates_the_new_text():
    """Codex review on PR 97: a content-change event that arrives while the
    message is being rated must not be dropped, and its retry must rate the
    new text, since the in-flight rating's guarded write is rejected."""
    repository = ContentGuardStubRepository()
    started = threading.Event()
    release = threading.Event()
    rated: list[str] = []

    def llm_judge_rater(text: str, **_context: Any) -> KtbJudgeVerdict:
        rated.append(text)
        if len(rated) == 1:
            started.set()
            assert release.wait(timeout=5)
        return _judge_verdict("Suicidal ideation")

    watcher = RealtimeConversationWatcher(
        repository=repository,
        llm_judge_rater=llm_judge_rater,
        reviewer_ids={SYSTEM_LLM_JUDGE_REVIEWER_ID},
        worker_threads=2,
    )
    document_id = ObjectId()
    repository.current_content[(document_id, 0)] = "Hello"
    watcher.handle_change_event(
        {
            "operationType": "insert",
            "fullDocument": _conversation_document(
                document_id, "c-merge", [_message("Hello", [], role="user")]
            ),
        }
    )
    assert started.wait(timeout=5)

    # The app appends to the message while it is being rated.
    repository.current_content[(document_id, 0)] = "Hello\nI want to die"
    watcher.handle_change_event(
        {
            "operationType": "update",
            "updateDescription": {
                "updatedFields": {"messages.0.content": "Hello\nI want to die"},
                "removedFields": [],
            },
            "fullDocument": _conversation_document(
                document_id,
                "c-merge",
                [_message("Hello\nI want to die", [], role="user")],
            ),
        }
    )
    assert rated == ["Hello"]  # deferred: the first rating is still in flight

    release.set()
    assert watcher.wait_for_pending(timeout=5)
    watcher.shutdown()

    assert rated == ["Hello", "Hello\nI want to die"]
    assert repository.skipped == ["Hello"]
    assert [call.expected_content for call in repository.calls] == [
        "Hello\nI want to die"
    ]
    assert not watcher._in_flight
    assert not watcher._pending_targets
    assert not watcher._retry_requested


@dataclass
class ResumingStubRepository(StubRepository):
    """Stub scan that resumes after the last complete conversation, like Mongo."""

    targets: list[MessageBackfillTarget] = field(default_factory=list)
    fetches: list[Any] = field(default_factory=list)

    def next_backfill_batch(
        self,
        batch_size: int = 200,
        excluded_provider_keys: set[tuple[Any, int, str]] | None = None,
        reviewer_ids: set[str] | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        after_id: Any = None,
    ) -> BackfillBatch:
        _ = (reviewer_ids, since, until)
        self.fetches.append(after_id)
        excluded_provider_keys = excluded_provider_keys or set()
        batch = [
            target
            for target in self.targets
            if (after_id is None or target.document_id > after_id)
            and not all(
                (target.document_id, target.message_index, reviewer_id)
                in excluded_provider_keys
                for reviewer_id in target.missing_reviewer_ids
            )
        ][:batch_size]
        # A full batch stops inside its last conversation, so the scan resumes
        # with that conversation again.
        resume_after = batch[-2].document_id if len(batch) == batch_size > 1 else None
        return BackfillBatch(targets=batch, resume_after=resume_after)

    def upsert_system_reviewer_flag(self, *args: Any, **kwargs: Any) -> bool:
        super().upsert_system_reviewer_flag(*args, **kwargs)
        self.targets = [
            target
            for target in self.targets
            if (target.document_id, target.message_index)
            != (kwargs["document_id"], kwargs["message_index"])
        ]
        return True


def test_backfill_rescans_from_the_start_before_completing():
    """Codex review on PR 97: a failed target behind the resume cursor must be
    retried before the pass is declared complete."""
    first_id, second_id = sorted([ObjectId(), ObjectId()])
    repository = ResumingStubRepository(
        targets=[
            MessageBackfillTarget(
                document_id=first_id,
                message_index=0,
                content="fails once",
                missing_reviewer_ids={SYSTEM_LLM_JUDGE_REVIEWER_ID},
                role="user",
            ),
            MessageBackfillTarget(
                document_id=second_id,
                message_index=0,
                content="fine",
                missing_reviewer_ids={SYSTEM_LLM_JUDGE_REVIEWER_ID},
                role="user",
            ),
        ]
    )
    attempts = {"count": 0}

    def flaky_judge(text: str, **_context: Any) -> KtbJudgeVerdict:
        if text == "fails once":
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise RuntimeError("model busy")
        return _judge_verdict()

    watcher = RealtimeConversationWatcher(
        repository=repository,
        llm_judge_rater=flaky_judge,
        reviewer_ids={SYSTEM_LLM_JUDGE_REVIEWER_ID},
        backfill_batch_size=2,
        backfill_retry_sleep_seconds=0.0,
    )

    watcher.run_startup_backfill()

    assert attempts["count"] == 2
    assert sorted(call.document_id for call in repository.calls) == [
        first_id,
        second_id,
    ]
    assert watcher._unresolved_backfill_targets == 0
    assert repository.targets == []
    # Scan from the start, continue after the first conversation, run dry,
    # rescan from the start, then confirm from the start that nothing is left.
    assert repository.fetches == [None, first_id, None, None]
