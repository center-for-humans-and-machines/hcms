"""Integration smoke tests for realtime monitoring with a real MongoDB replica set."""

# pylint: disable=missing-function-docstring,redefined-outer-name,duplicate-code

from __future__ import annotations

from datetime import datetime, timezone
import os
from typing import Any
import uuid

import pytest

from hpms.database.repository import (
    SYSTEM_LLAMA_REVIEWER_ID,
    SYSTEM_OPENAI_REVIEWER_ID,
    MongoConversationRepository,
)
from hpms.monitoring.realtime_watcher import RealtimeConversationWatcher

try:
    from pymongo import MongoClient as PYMONGO_CLIENT
except ModuleNotFoundError:  # pragma: no cover - optional local dependency
    PYMONGO_CLIENT = None

MONGO_URI = os.getenv("HPMS_INTEGRATION_MONGODB_URI")
pytestmark = [
    pytest.mark.skipif(
        not MONGO_URI,
        reason="Set HPMS_INTEGRATION_MONGODB_URI to run Mongo integration tests.",
    ),
    pytest.mark.skipif(
        PYMONGO_CLIENT is None,
        reason="Install pymongo to run Mongo integration tests.",
    ),
]


def _conversation_doc(
    conversation_id: str,
    reviewer_flags: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    now = datetime.now(timezone.utc)
    return {
        "_id": conversation_id,
        "participant_id": "p1",
        "model": "test-model",
        "experiment_id": "exp-1",
        "conversation_id": conversation_id,
        "project_id": "2026_03_08",
        "created_at": now,
        "messages": [
            {
                "content": "needs moderation",
                "role": "assistant",
                "timestamp": now,
                "type": "assistant",
                "user_flag": {
                    "category": "",
                    "category_other": "",
                    "reviews": [],
                },
                "reviewer_flags": reviewer_flags or [],
            }
        ],
        "opened_by": [],
        "reviewed_by": [],
        "assigned_to": [],
    }


@pytest.fixture()
def collection():
    """Provide isolated real Mongo collection for each integration test."""
    assert PYMONGO_CLIENT is not None
    client = PYMONGO_CLIENT(MONGO_URI, serverSelectionTimeoutMS=5000)
    database_name = f"hpms_integration_{uuid.uuid4().hex}"
    db = client[database_name]
    conversations = db["Conversations"]
    yield conversations
    client.drop_database(database_name)
    client.close()


def test_startup_backfill_writes_both_system_flags(collection):
    collection.insert_one(_conversation_doc("conversation-int-1"))
    repository = MongoConversationRepository.from_collection(collection)
    watcher = RealtimeConversationWatcher(
        repository=repository,
        openai_rater=lambda _text: [0, "hate"],
        llama_guard_rater=lambda _text: "Hate",
        backfill_retry_sleep_seconds=0.0,
    )

    watcher.run_startup_backfill()

    stored = collection.find_one({"_id": "conversation-int-1"})
    assert stored is not None
    reviewer_ids = {
        flag["reviewer_id"] for flag in stored["messages"][0]["reviewer_flags"]
    }
    assert reviewer_ids == {SYSTEM_OPENAI_REVIEWER_ID, SYSTEM_LLAMA_REVIEWER_ID}


def test_change_event_writes_missing_system_flag(collection):
    now = datetime.now(timezone.utc)
    openai_flag = {
        "reviewer_id": SYSTEM_OPENAI_REVIEWER_ID,
        "created_at": now,
        "categories": ["hate"],
        "category_other": "",
    }
    collection.insert_one(_conversation_doc("conversation-int-2", [openai_flag]))

    repository = MongoConversationRepository.from_collection(collection)
    watcher = RealtimeConversationWatcher(
        repository=repository,
        openai_rater=lambda _text: [0, "hate"],
        llama_guard_rater=lambda _text: "Hate",
    )

    full_document = collection.find_one({"_id": "conversation-int-2"})
    assert full_document is not None
    event: dict[str, Any] = {"operationType": "insert", "fullDocument": full_document}
    watcher.handle_change_event(event)

    stored = collection.find_one({"_id": "conversation-int-2"})
    assert stored is not None
    reviewer_ids = {
        flag["reviewer_id"] for flag in stored["messages"][0]["reviewer_flags"]
    }
    assert reviewer_ids == {SYSTEM_OPENAI_REVIEWER_ID, SYSTEM_LLAMA_REVIEWER_ID}


def test_change_event_does_not_create_duplicate_provider_entries(collection):
    collection.insert_one(_conversation_doc("conversation-int-3"))

    repository = MongoConversationRepository.from_collection(collection)
    watcher = RealtimeConversationWatcher(
        repository=repository,
        openai_rater=lambda _text: [0, "hate"],
        llama_guard_rater=lambda _text: "Hate",
    )

    stale_document = collection.find_one({"_id": "conversation-int-3"})
    assert stale_document is not None
    event = {"operationType": "insert", "fullDocument": stale_document}

    watcher.handle_change_event(event)
    watcher.handle_change_event(event)

    stored = collection.find_one({"_id": "conversation-int-3"})
    assert stored is not None
    reviewer_flags = stored["messages"][0]["reviewer_flags"]
    assert len(reviewer_flags) == 2
    assert sorted(flag["reviewer_id"] for flag in reviewer_flags) == sorted(
        [SYSTEM_OPENAI_REVIEWER_ID, SYSTEM_LLAMA_REVIEWER_ID]
    )
