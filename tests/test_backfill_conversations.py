"""Tests for the one-off historical backfill job."""

# pylint: disable=missing-function-docstring,protected-access,too-few-public-methods

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from hcms.database.repository import SYSTEM_LLM_JUDGE_REVIEWER_ID
from hcms.monitoring import backfill_conversations


def test_parse_args_requires_a_known_provider():
    with pytest.raises(SystemExit):
        backfill_conversations.parse_args(["--provider", "unknown"])


def test_parse_args_parses_the_message_window():
    args = backfill_conversations.parse_args(
        [
            "--provider",
            "llm_judge",
            "--since",
            "2026-10-01",
            "--until",
            "2026-11-03T08:00:00Z",
        ]
    )

    assert args.provider == "llm_judge"
    assert args.since == datetime(2026, 10, 1, tzinfo=timezone.utc)
    assert args.until == datetime(2026, 11, 3, 8, tzinfo=timezone.utc)
    assert args.batch_size == 20


class _Repository:
    closed = False

    def close(self) -> None:
        self.closed = True


class _Watcher:
    """Records how the job configures the watcher and fakes its outcome."""

    created: list[dict[str, Any]] = []
    unresolved = 0

    def __init__(self, **kwargs: Any) -> None:
        _Watcher.created.append(kwargs)
        self._unresolved_backfill_targets = _Watcher.unresolved

    def run_startup_backfill(self) -> None:
        return None


@pytest.fixture(name="fake_watcher")
def fixture_fake_watcher(monkeypatch):
    _Watcher.created = []
    _Watcher.unresolved = 0
    repository = _Repository()
    monkeypatch.setattr(backfill_conversations, "create_repository", lambda: repository)
    monkeypatch.setattr(backfill_conversations, "RealtimeConversationWatcher", _Watcher)
    return repository


def test_run_backfills_one_provider_in_the_window(fake_watcher):
    args = backfill_conversations.parse_args(
        ["--provider", "llm_judge", "--since", "2026-10-01", "--batch-size", "5"]
    )

    exit_code = backfill_conversations.run(args)

    assert exit_code == 0
    assert len(_Watcher.created) == 1
    kwargs = _Watcher.created[0]
    assert kwargs["reviewer_ids"] == {SYSTEM_LLM_JUDGE_REVIEWER_ID}
    assert kwargs["backfill_since"] == datetime(2026, 10, 1, tzinfo=timezone.utc)
    assert kwargs["backfill_until"] is None
    assert kwargs["backfill_batch_size"] == 5
    assert fake_watcher.closed is True


def test_run_exits_non_zero_when_messages_stay_unrated(fake_watcher):
    _Watcher.unresolved = 2
    args = backfill_conversations.parse_args(["--provider", "llm_judge"])

    assert backfill_conversations.run(args) == 1
    assert fake_watcher.closed is True
