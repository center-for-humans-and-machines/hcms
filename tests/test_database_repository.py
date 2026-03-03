"""Tests for Mongo conversation repository helpers."""

# pylint: disable=missing-function-docstring,too-few-public-methods,too-many-locals,duplicate-code

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from hpms.database.repository import (
    SYSTEM_LLAMA_REVIEWER_ID,
    SYSTEM_OPENAI_REVIEWER_ID,
    MongoConversationRepository,
)


@dataclass
class _UpdateResult:
    matched_count: int


class FakeCollection:
    """Minimal in-memory collection for repository unit tests."""

    def __init__(self, docs: list[dict[str, Any]]):
        self.docs = {doc["_id"]: deepcopy(doc) for doc in docs}
        self.update_calls = 0

    def find(self, *_args, **_kwargs):
        return [deepcopy(doc) for doc in self.docs.values()]

    def find_one(self, query: dict[str, Any], projection: dict[str, Any] | None = None):
        _ = projection
        doc = self.docs.get(query.get("_id"))
        if doc is None:
            return None
        return deepcopy(doc)

    def update_one(
        self,
        query: dict[str, Any],
        update: Any,
        array_filters: list[dict[str, Any]] | None = None,
    ):
        _ = array_filters
        self.update_calls += 1

        doc = self.docs.get(query.get("_id"))
        if doc is None:
            return _UpdateResult(0)

        message_index = None
        for key in query:
            if not key.startswith("messages."):
                continue
            parts = key.split(".")
            if len(parts) >= 2 and parts[1].isdigit():
                message_index = int(parts[1])
                break

        if message_index is None:
            raise AssertionError("Missing message index in query")
        if message_index < 0 or message_index >= len(doc["messages"]):
            return _UpdateResult(0)

        message = doc["messages"][message_index]
        reviewer_flags = message.setdefault("reviewer_flags", [])

        if isinstance(update, list):
            assert len(update) == 1
            stage = update[0]
            set_key = f"messages.{message_index}.reviewer_flags"
            new_flag = deepcopy(stage["$set"][set_key]["$let"]["vars"]["new_flag"])
            reviewer_id = new_flag["reviewer_id"]
            for index, flag in enumerate(reviewer_flags):
                if flag.get("reviewer_id") == reviewer_id:
                    reviewer_flags[index] = new_flag
                    return _UpdateResult(1)
            reviewer_flags.append(new_flag)
            return _UpdateResult(1)

        raise AssertionError("Unsupported update operation in fake collection")


def _conversation_doc() -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    return {
        "_id": "conversation-1",
        "participant_id": "p1",
        "model": "test-model",
        "experiment_id": "exp-1",
        "created_at": now,
        "messages": [
            {
                "content": "message one",
                "role": "assistant",
                "timestamp": now,
                "type": "assistant",
                "user_flag": {
                    "category": "",
                    "category_other": "",
                    "reviews": [],
                },
                "reviewer_flags": [],
            },
            {
                "content": "message two",
                "role": "assistant",
                "timestamp": now,
                "type": "assistant",
                "user_flag": {
                    "category": "",
                    "category_other": "",
                    "reviews": [],
                },
                "reviewer_flags": [
                    {
                        "reviewer_id": SYSTEM_OPENAI_REVIEWER_ID,
                        "created_at": now,
                        "categories": ["violence"],
                        "category_other": "",
                    }
                ],
            },
        ],
        "opened_by": [],
        "reviewed_by": [],
        "assigned_to": [],
    }


def test_get_backfill_targets_includes_only_messages_missing_system_flags():
    repository = MongoConversationRepository.from_collection(
        FakeCollection([_conversation_doc()])
    )

    targets = repository.get_backfill_targets(batch_size=10)

    assert len(targets) == 2

    first = targets[0]
    assert first.conversation_id == "conversation-1"
    assert first.message_index == 0
    assert first.missing_reviewer_ids == {
        SYSTEM_OPENAI_REVIEWER_ID,
        SYSTEM_LLAMA_REVIEWER_ID,
    }

    second = targets[1]
    assert second.message_index == 1
    assert second.missing_reviewer_ids == {SYSTEM_LLAMA_REVIEWER_ID}


def test_get_backfill_targets_excludes_exhausted_provider_keys():
    repository = MongoConversationRepository.from_collection(
        FakeCollection([_conversation_doc()])
    )
    excluded_provider_keys = {
        ("conversation-1", 0, SYSTEM_OPENAI_REVIEWER_ID),
        ("conversation-1", 0, SYSTEM_LLAMA_REVIEWER_ID),
    }

    targets = repository.get_backfill_targets(
        batch_size=1, excluded_provider_keys=excluded_provider_keys
    )

    assert len(targets) == 1
    assert targets[0].conversation_id == "conversation-1"
    assert targets[0].message_index == 1
    assert targets[0].missing_reviewer_ids == {SYSTEM_LLAMA_REVIEWER_ID}


def test_upsert_system_reviewer_flag_is_idempotent_per_message_and_provider():
    collection = FakeCollection([_conversation_doc()])
    repository = MongoConversationRepository.from_collection(collection)

    repository.upsert_system_reviewer_flag(
        conversation_id="conversation-1",
        message_index=0,
        reviewer_id=SYSTEM_OPENAI_REVIEWER_ID,
        categories=["harassment"],
        category_other="",
        created_at=datetime(2026, 3, 2, 12, 0, 0),
    )
    repository.upsert_system_reviewer_flag(
        conversation_id="conversation-1",
        message_index=0,
        reviewer_id=SYSTEM_OPENAI_REVIEWER_ID,
        categories=["hate"],
        category_other="",
        created_at=datetime(2026, 3, 2, 12, 1, 0),
    )

    message_flags = collection.docs["conversation-1"]["messages"][0]["reviewer_flags"]
    assert len(message_flags) == 1
    assert message_flags[0]["reviewer_id"] == SYSTEM_OPENAI_REVIEWER_ID
    assert message_flags[0]["categories"] == ["hate"]
    assert collection.update_calls == 2


def test_get_backfill_targets_skips_invalid_conversations():
    invalid = {
        "_id": "invalid",
        # Missing required top-level schema fields
        "messages": [],
    }

    repository = MongoConversationRepository.from_collection(
        FakeCollection([invalid, _conversation_doc()])
    )

    targets = repository.get_backfill_targets(batch_size=10)

    assert len(targets) == 2
    assert all(target.conversation_id == "conversation-1" for target in targets)


def test_get_backfill_targets_defaults_missing_user_flag():
    now = datetime.now(timezone.utc)
    conversation_missing_user_flag = {
        "_id": "conversation-missing-user-flag",
        "participant_id": "p2",
        "model": "test-model",
        "experiment_id": "exp-2",
        "created_at": now,
        "messages": [
            {
                "content": "message without user flag",
                "role": "assistant",
                "timestamp": now,
                "type": "assistant",
                "reviewer_flags": [],
            }
        ],
        "opened_by": [],
        "reviewed_by": [],
        "assigned_to": [],
    }

    repository = MongoConversationRepository.from_collection(
        FakeCollection([conversation_missing_user_flag])
    )

    targets = repository.get_backfill_targets(batch_size=10)

    assert len(targets) == 1
    assert targets[0].conversation_id == "conversation-missing-user-flag"
    assert targets[0].message_index == 0
    assert targets[0].missing_reviewer_ids == {
        SYSTEM_OPENAI_REVIEWER_ID,
        SYSTEM_LLAMA_REVIEWER_ID,
    }


def test_get_backfill_targets_defaults_missing_optional_lists():
    now = datetime.now(timezone.utc)
    conversation_missing_optional_lists = {
        "_id": "conversation-missing-lists",
        "participant_id": "p3",
        "model": "test-model",
        "experiment_id": "exp-3",
        "created_at": now,
        "messages": [
            {
                "content": "message without optional lists",
                "role": "assistant",
                "timestamp": now,
                "type": "assistant",
            }
        ],
    }

    repository = MongoConversationRepository.from_collection(
        FakeCollection([conversation_missing_optional_lists])
    )

    targets = repository.get_backfill_targets(batch_size=10)

    assert len(targets) == 1
    assert targets[0].conversation_id == "conversation-missing-lists"
    assert targets[0].message_index == 0
    assert targets[0].missing_reviewer_ids == {
        SYSTEM_OPENAI_REVIEWER_ID,
        SYSTEM_LLAMA_REVIEWER_ID,
    }
