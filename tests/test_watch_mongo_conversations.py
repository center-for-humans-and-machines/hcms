"""Tests for the realtime watcher entrypoint's environment settings."""

# pylint: disable=missing-function-docstring,protected-access

from __future__ import annotations

import logging
from datetime import datetime, timezone

import pytest

from hcms.database.repository import (
    SYSTEM_LLAMA_REVIEWER_ID,
    SYSTEM_LLM_JUDGE_REVIEWER_ID,
    SYSTEM_OPENAI_REVIEWER_ID,
)
from hcms.monitoring import watch_mongo_conversations


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        # The LLM judge is opt-in: an unset variable never requires its endpoint.
        (None, {SYSTEM_OPENAI_REVIEWER_ID, SYSTEM_LLAMA_REVIEWER_ID}),
        ("llama_guard", {SYSTEM_LLAMA_REVIEWER_ID}),
        (" openai_moderation ", {SYSTEM_OPENAI_REVIEWER_ID}),
        (
            "openai_moderation,llama_guard",
            {SYSTEM_OPENAI_REVIEWER_ID, SYSTEM_LLAMA_REVIEWER_ID},
        ),
        ("llm_judge", {SYSTEM_LLM_JUDGE_REVIEWER_ID}),
        (
            "llama_guard,llm_judge",
            {SYSTEM_LLAMA_REVIEWER_ID, SYSTEM_LLM_JUDGE_REVIEWER_ID},
        ),
    ],
)
def test_get_reviewer_ids_reads_moderation_providers(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("HCMS_MODERATION_PROVIDERS", raising=False)
    else:
        monkeypatch.setenv("HCMS_MODERATION_PROVIDERS", value)

    assert watch_mongo_conversations._get_reviewer_ids() == expected


@pytest.mark.parametrize("value", ["", ",", "llama_guard,unknown"])
def test_get_reviewer_ids_rejects_invalid_providers(monkeypatch, value):
    monkeypatch.setenv("HCMS_MODERATION_PROVIDERS", value)

    with pytest.raises(ValueError, match="HCMS_MODERATION_PROVIDERS"):
        watch_mongo_conversations._get_reviewer_ids()


def test_configure_logging_reads_hcms_log_level(monkeypatch):
    levels: list[int] = []
    monkeypatch.setenv("HCMS_LOG_LEVEL", "debug")
    monkeypatch.setattr(
        watch_mongo_conversations.logging,
        "basicConfig",
        lambda **kwargs: levels.append(kwargs["level"]),
    )

    watch_mongo_conversations._configure_logging()

    assert levels == [logging.DEBUG]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        ("", None),
        ("2026-11-03", datetime(2026, 11, 3, tzinfo=timezone.utc)),
        ("2026-11-03T08:30:00Z", datetime(2026, 11, 3, 8, 30, tzinfo=timezone.utc)),
        (
            "2026-11-03T08:30:00+02:00",
            datetime(2026, 11, 3, 6, 30, tzinfo=timezone.utc),
        ),
    ],
)
def test_get_datetime_env_parses_backfill_since(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("MONGODB_BACKFILL_SINCE", raising=False)
    else:
        monkeypatch.setenv("MONGODB_BACKFILL_SINCE", value)

    assert (
        watch_mongo_conversations._get_datetime_env("MONGODB_BACKFILL_SINCE")
        == expected
    )


def test_get_datetime_env_rejects_invalid_timestamps(monkeypatch):
    monkeypatch.setenv("MONGODB_BACKFILL_SINCE", "yesterday")

    with pytest.raises(ValueError, match="MONGODB_BACKFILL_SINCE"):
        watch_mongo_conversations._get_datetime_env("MONGODB_BACKFILL_SINCE")
