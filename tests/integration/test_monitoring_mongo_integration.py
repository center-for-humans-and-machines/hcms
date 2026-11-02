"""Integration smoke tests for realtime monitoring with a real MongoDB replica set."""

# pylint: disable=missing-function-docstring,redefined-outer-name,duplicate-code

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from typing import Any

import pytest
from bson import ObjectId

from hcms.database.repository import (
    SYSTEM_LLAMA_REVIEWER_ID,
    SYSTEM_OPENAI_REVIEWER_ID,
    MongoConversationRepository,
)
from hcms.monitoring.realtime_watcher import RealtimeConversationWatcher

try:
    from pymongo import MongoClient as PYMONGO_CLIENT
except ModuleNotFoundError:  # pragma: no cover - optional local dependency
    PYMONGO_CLIENT = None

# The dev cluster the watchers use (GitHub secret MONGODB_URI_DEV); each test
# works in its own throwaway database there. A local replica set works too.
MONGO_URI = os.getenv("MONGODB_URI_DEV") or os.getenv("HCMS_INTEGRATION_MONGODB_URI")
pytestmark = [
    pytest.mark.skipif(
        not MONGO_URI,
        reason=(
            "Set MONGODB_URI_DEV (or HCMS_INTEGRATION_MONGODB_URI) to run Mongo "
            "integration tests."
        ),
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
        "_id": ObjectId(),
        "conversation_id": conversation_id,
        "participant_id": "p1",
        "model": "test-model",
        "experiment_id": "exp-1",
        "project_id": "2026_03_08",
        "created_at": now,
        "custom_system_message_id": None,
        "multi_rounds": True,
        "messages": [
            {
                "content": "needs moderation",
                "role": "assistant",
                "timestamp": now,
                "type": "assistant",
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
                "reviewer_flags": reviewer_flags or [],
                "duplicate_flags": [],
            }
        ],
        "opened_by": [],
        "assigned_messages": [],
        "reviewed_messages": [],
    }


@pytest.fixture()
def collection():
    """Provide isolated real Mongo collection for each integration test."""
    assert PYMONGO_CLIENT is not None
    client = PYMONGO_CLIENT(MONGO_URI, serverSelectionTimeoutMS=5000)
    database_name = f"hcms_integration_{uuid.uuid4().hex}"
    db = client[database_name]
    conversations = db["Conversations"]
    yield conversations
    client.drop_database(database_name)
    client.close()


def test_startup_backfill_writes_both_system_flags(collection):
    document = _conversation_doc("conversation-int-1")
    collection.insert_one(document)
    repository = MongoConversationRepository.from_collection(collection)
    watcher = RealtimeConversationWatcher(
        repository=repository,
        openai_rater=lambda _text: [0, "hate"],
        llama_guard_rater=lambda _text, **_context: "Hate",
        backfill_retry_sleep_seconds=0.0,
    )

    watcher.run_startup_backfill()

    stored = collection.find_one({"_id": document["_id"]})
    assert stored is not None
    reviewer_ids = {
        flag["reviewer_id"] for flag in stored["messages"][0]["reviewer_flags"]
    }
    assert reviewer_ids == {SYSTEM_OPENAI_REVIEWER_ID, SYSTEM_LLAMA_REVIEWER_ID}


def test_change_event_writes_missing_system_flag(collection):
    now = datetime.now(timezone.utc)
    document = _conversation_doc(
        "conversation-int-2",
        [
            {
                "reviewer_id": SYSTEM_OPENAI_REVIEWER_ID,
                "reviewer_by_username": SYSTEM_OPENAI_REVIEWER_ID,
                "created_at": now,
                "categories": ["hate"],
                "category_other": "",
                "comment": "",
            }
        ],
    )
    collection.insert_one(document)

    repository = MongoConversationRepository.from_collection(collection)
    watcher = RealtimeConversationWatcher(
        repository=repository,
        openai_rater=lambda _text: [0, "hate"],
        llama_guard_rater=lambda _text, **_context: "Hate",
    )

    full_document = collection.find_one({"_id": document["_id"]})
    assert full_document is not None
    event: dict[str, Any] = {"operationType": "insert", "fullDocument": full_document}
    watcher.handle_change_event(event)

    stored = collection.find_one({"_id": document["_id"]})
    assert stored is not None
    reviewer_ids = {
        flag["reviewer_id"] for flag in stored["messages"][0]["reviewer_flags"]
    }
    assert reviewer_ids == {SYSTEM_OPENAI_REVIEWER_ID, SYSTEM_LLAMA_REVIEWER_ID}


def test_change_event_does_not_create_duplicate_provider_entries(collection):
    document = _conversation_doc("conversation-int-3")
    collection.insert_one(document)

    repository = MongoConversationRepository.from_collection(collection)
    watcher = RealtimeConversationWatcher(
        repository=repository,
        openai_rater=lambda _text: [0, "hate"],
        llama_guard_rater=lambda _text, **_context: "Hate",
    )

    stale_document = collection.find_one({"_id": document["_id"]})
    assert stale_document is not None
    event = {"operationType": "insert", "fullDocument": stale_document}

    watcher.handle_change_event(event)
    watcher.handle_change_event(event)

    stored = collection.find_one({"_id": document["_id"]})
    assert stored is not None
    reviewer_flags = stored["messages"][0]["reviewer_flags"]
    assert len(reviewer_flags) == 2
    assert sorted(flag["reviewer_id"] for flag in reviewer_flags) == sorted(
        [SYSTEM_OPENAI_REVIEWER_ID, SYSTEM_LLAMA_REVIEWER_ID]
    )


def _user_message(content: str, timestamp: datetime) -> dict[str, object]:
    return {
        "content": content,
        "role": "user",
        "timestamp": timestamp,
        "type": "user",
        "flagged": False,
        "flagged_at": None,
        "flagged_by": None,
        "flag_category": None,
        "flag_other_reason": None,
        "user_flag": {"category": "", "category_other": "", "reviews": []},
        "reviewer_flags": [],
        "duplicate_flags": [],
    }


def _judge_verdict(*categories: str) -> Any:
    # pylint: disable=import-outside-toplevel
    from hcms.rating.ktb_judge import KtbJudgeVerdict

    return KtbJudgeVerdict(categories=list(categories), rationale="because")


def test_backfill_since_rates_a_new_message_in_an_old_conversation(collection):
    # pylint: disable=import-outside-toplevel
    from datetime import timedelta

    from hcms.database.repository import SYSTEM_LLM_JUDGE_REVIEWER_ID

    document = _conversation_doc("conversation-int-4")
    collection.insert_one(document)
    since = datetime.now(timezone.utc) + timedelta(seconds=1)
    collection.update_one(
        {"_id": document["_id"]},
        {
            "$push": {
                "messages": _user_message("I want to die", since + timedelta(seconds=1))
            }
        },
    )

    repository = MongoConversationRepository.from_collection(collection)
    watcher = RealtimeConversationWatcher(
        repository=repository,
        llm_judge_rater=lambda _text, **_context: _judge_verdict("Suicidal ideation"),
        reviewer_ids={SYSTEM_LLM_JUDGE_REVIEWER_ID},
        backfill_since=since,
        backfill_retry_sleep_seconds=0.0,
    )

    # A restart after go-live: the old message stays unrated, the new one is found.
    watcher.run_startup_backfill()

    stored = collection.find_one({"_id": document["_id"]})
    assert stored is not None
    assert stored["messages"][0]["reviewer_flags"] == []
    (flag,) = stored["messages"][1]["reviewer_flags"]
    assert flag["reviewer_id"] == SYSTEM_LLM_JUDGE_REVIEWER_ID
    assert flag["categories"] == ["Suicidal ideation"]
    assert flag["comment"] == "because"


def test_flag_is_not_written_for_content_changed_while_rating(collection):
    # pylint: disable=import-outside-toplevel
    from hcms.database.repository import SYSTEM_LLM_JUDGE_REVIEWER_ID

    document = _conversation_doc("conversation-int-5")
    collection.insert_one(document)
    repository = MongoConversationRepository.from_collection(collection)
    rated_texts: list[str] = []

    def judge_while_message_is_merged(text: str, **_context: Any) -> Any:
        rated_texts.append(text)
        if len(rated_texts) == 1:
            # The app appends to the message between our read and our write.
            collection.update_one(
                {"_id": document["_id"]},
                {"$set": {"messages.0.content": "needs moderation\nand more"}},
            )
        return _judge_verdict("Self-harm")

    watcher = RealtimeConversationWatcher(
        repository=repository,
        llm_judge_rater=judge_while_message_is_merged,
        reviewer_ids={SYSTEM_LLM_JUDGE_REVIEWER_ID},
    )
    stale_document = collection.find_one({"_id": document["_id"]})
    assert stale_document is not None

    watcher.handle_change_event(
        {"operationType": "insert", "fullDocument": stale_document}
    )

    stored = collection.find_one({"_id": document["_id"]})
    assert stored is not None
    assert stored["messages"][0]["reviewer_flags"] == []

    # The content-change event rates the final text.
    watcher.handle_change_event(
        {
            "operationType": "update",
            "updateDescription": {
                "updatedFields": {"messages.0.content": "needs moderation\nand more"},
                "removedFields": [],
            },
            "fullDocument": stored,
        }
    )

    stored = collection.find_one({"_id": document["_id"]})
    assert stored is not None
    (flag,) = stored["messages"][0]["reviewer_flags"]
    assert flag["reviewer_id"] == SYSTEM_LLM_JUDGE_REVIEWER_ID
    assert rated_texts == ["needs moderation", "needs moderation\nand more"]
