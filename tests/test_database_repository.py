"""Tests for Mongo conversation repository helpers."""

# pylint: disable=missing-function-docstring,too-few-public-methods,too-many-locals

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
        update: dict[str, Any],
        array_filters: list[dict[str, Any]] | None = None,
    ):
        doc = self.docs.get(query.get("_id"))
        if doc is None:
            return _UpdateResult(0)

        message_key = next(
            key
            for key in query.keys()
            if key.startswith("messages.") and key.endswith("reviewer_id")
        )
        message_index = int(message_key.split(".")[1])
        reviewer_condition = query[message_key]

        message = doc["messages"][message_index]
        reviewer_flags = message.setdefault("reviewer_flags", [])

        if isinstance(reviewer_condition, str):
            if not any(
                flag.get("reviewer_id") == reviewer_condition for flag in reviewer_flags
            ):
                return _UpdateResult(0)

        if isinstance(reviewer_condition, dict) and "$ne" in reviewer_condition:
            reviewer_id = reviewer_condition["$ne"]
            if any(flag.get("reviewer_id") == reviewer_id for flag in reviewer_flags):
                return _UpdateResult(0)

        if "$set" in update:
            assert array_filters is not None
            reviewer_id = array_filters[0]["flag.reviewer_id"]
            for flag in reviewer_flags:
                if flag.get("reviewer_id") != reviewer_id:
                    continue
                for path, value in update["$set"].items():
                    field_name = path.split(".")[-1]
                    flag[field_name] = value
                return _UpdateResult(1)
            return _UpdateResult(0)

        if "$push" in update:
            push_key = f"messages.{message_index}.reviewer_flags"
            reviewer_flags.append(deepcopy(update["$push"][push_key]))
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
