"""Tests for the realtime watcher entrypoint's environment settings."""

# pylint: disable=missing-function-docstring,protected-access

from __future__ import annotations

import logging

import pytest

from hcms.database.repository import SYSTEM_LLAMA_REVIEWER_ID, SYSTEM_OPENAI_REVIEWER_ID
from hcms.monitoring import watch_mongo_conversations


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, {SYSTEM_OPENAI_REVIEWER_ID, SYSTEM_LLAMA_REVIEWER_ID}),
        ("llama_guard", {SYSTEM_LLAMA_REVIEWER_ID}),
        (" openai_moderation ", {SYSTEM_OPENAI_REVIEWER_ID}),
        (
            "openai_moderation,llama_guard",
            {SYSTEM_OPENAI_REVIEWER_ID, SYSTEM_LLAMA_REVIEWER_ID},
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
