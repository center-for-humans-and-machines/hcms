"""Tests for Mongo conversation repository helpers."""

# pylint: disable=missing-function-docstring,too-few-public-methods,duplicate-code

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from bson import ObjectId
import pytest
from pydantic import ValidationError

from hpms.database.models import ConversationDocument
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

    # pylint: disable=too-many-locals
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
        reviewer_flags = message["reviewer_flags"]

        if isinstance(update, list):
            assert len(update) == 1
            stage = update[0]
            set_messages = stage["$set"]["messages"]
            new_flag = deepcopy(set_messages["$let"]["vars"]["new_flag"])
            target_index = set_messages["$let"]["vars"]["target_index"]
            assert target_index == message_index
            reviewer_id = new_flag["reviewer_id"]
            for index, flag in enumerate(reviewer_flags):
                if flag.get("reviewer_id") == reviewer_id:
                    reviewer_flags[index] = new_flag
                    return _UpdateResult(1)
            reviewer_flags.append(new_flag)
            return _UpdateResult(1)

        raise AssertionError("Unsupported update operation in fake collection")


def _message_doc(
    now: datetime,
    *,
    content: str,
    role: str = "assistant",
    reviewer_flags: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "content": content,
        "role": role,
        "timestamp": now,
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
        "reviewer_flags": reviewer_flags or [],
        "duplicate_flags": [],
    }


def _conversation_doc() -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    document_id = ObjectId()
    return {
        "_id": document_id,
        "conversation_id": "conversation-1",
        "participant_id": "p1",
        "model": "test-model",
        "experiment_id": "exp-1",
        "project_id": "2026_03_08",
        "created_at": now,
        "custom_system_message_id": None,
        "multi_rounds": True,
        "messages": [
            _message_doc(now, content="message one"),
            _message_doc(
                now,
                content="message two",
                reviewer_flags=[
                    {
                        "reviewer_id": SYSTEM_OPENAI_REVIEWER_ID,
                        "reviewer_by_username": SYSTEM_OPENAI_REVIEWER_ID,
                        "created_at": now,
                        "categories": ["violence"],
                        "category_other": "",
                        "comment": "",
                    }
                ],
            ),
        ],
        "opened_by": [],
        "assigned_messages": [],
        "reviewed_messages": [],
    }


def test_get_backfill_targets_includes_only_messages_missing_system_flags():
    conversation = _conversation_doc()
    repository = MongoConversationRepository.from_collection(
        FakeCollection([conversation])
    )

    targets = repository.get_backfill_targets(batch_size=10)

    assert len(targets) == 2

    first = targets[0]
    assert first.document_id == conversation["_id"]
    assert first.message_index == 0
    assert first.missing_reviewer_ids == {
        SYSTEM_OPENAI_REVIEWER_ID,
        SYSTEM_LLAMA_REVIEWER_ID,
    }

    second = targets[1]
    assert second.document_id == conversation["_id"]
    assert second.message_index == 1
    assert second.missing_reviewer_ids == {SYSTEM_LLAMA_REVIEWER_ID}


def test_get_backfill_targets_excludes_exhausted_provider_keys():
    conversation = _conversation_doc()
    repository = MongoConversationRepository.from_collection(
        FakeCollection([conversation])
    )
    excluded_provider_keys = {
        (conversation["_id"], 0, SYSTEM_OPENAI_REVIEWER_ID),
        (conversation["_id"], 0, SYSTEM_LLAMA_REVIEWER_ID),
    }

    targets = repository.get_backfill_targets(
        batch_size=1, excluded_provider_keys=excluded_provider_keys
    )

    assert len(targets) == 1
    assert targets[0].document_id == conversation["_id"]
    assert targets[0].message_index == 1
    assert targets[0].missing_reviewer_ids == {SYSTEM_LLAMA_REVIEWER_ID}


def test_missing_system_reviewers_returns_empty_for_system_role_message():
    now = datetime.now(timezone.utc)
    message = _message_doc(now, content="system prompt", role="system")

    missing = MongoConversationRepository.missing_system_reviewers(message)

    assert missing == set()


def test_upsert_system_reviewer_flag_is_idempotent_per_message_and_provider():
    conversation = _conversation_doc()
    collection = FakeCollection([conversation])
    repository = MongoConversationRepository.from_collection(collection)

    repository.upsert_system_reviewer_flag(
        document_id=conversation["_id"],
        message_index=0,
        reviewer_id=SYSTEM_OPENAI_REVIEWER_ID,
        categories=["harassment"],
        category_other="",
        created_at=datetime(2026, 3, 2, 12, 0, 0, tzinfo=timezone.utc),
    )
    repository.upsert_system_reviewer_flag(
        document_id=conversation["_id"],
        message_index=0,
        reviewer_id=SYSTEM_OPENAI_REVIEWER_ID,
        categories=["hate"],
        category_other="",
        created_at=datetime(2026, 3, 2, 12, 1, 0, tzinfo=timezone.utc),
    )

    message_flags = collection.docs[conversation["_id"]]["messages"][0][
        "reviewer_flags"
    ]
    assert len(message_flags) == 1
    assert message_flags[0]["reviewer_id"] == SYSTEM_OPENAI_REVIEWER_ID
    assert message_flags[0]["reviewer_by_username"] == SYSTEM_OPENAI_REVIEWER_ID
    assert message_flags[0]["categories"] == ["hate"]
    assert message_flags[0]["comment"] == ""
    assert collection.update_calls == 2


def test_get_backfill_targets_skips_invalid_conversations():
    invalid = {
        "_id": ObjectId(),
        "messages": [],
    }
    valid = _conversation_doc()

    repository = MongoConversationRepository.from_collection(
        FakeCollection([invalid, valid])
    )

    targets = repository.get_backfill_targets(batch_size=10)

    assert len(targets) == 2
    assert all(target.document_id == valid["_id"] for target in targets)


def test_get_backfill_targets_accepts_messages_with_empty_flag_state():
    conversation = _conversation_doc()
    conversation["messages"][0]["flagged_by"] = ""
    conversation["messages"][0]["user_flag"] = None
    repository = MongoConversationRepository.from_collection(
        FakeCollection([conversation])
    )

    targets = repository.get_backfill_targets(batch_size=10)

    assert len(targets) == 2
    assert all(target.document_id == conversation["_id"] for target in targets)


@pytest.mark.parametrize("participant_id", ["", " "])
def test_get_backfill_targets_accepts_blank_participant_id(participant_id):
    conversation = _conversation_doc()
    conversation["participant_id"] = participant_id
    repository = MongoConversationRepository.from_collection(
        FakeCollection([conversation])
    )

    targets = repository.get_backfill_targets(batch_size=10)

    assert len(targets) == 2
    assert all(target.document_id == conversation["_id"] for target in targets)


def test_conversation_document_accepts_canonical_shape():
    now = datetime.now(timezone.utc)
    document_id = ObjectId()
    document = ConversationDocument.model_validate(
        {
            "_id": document_id,
            "conversation_id": "conversation-canonical",
            "participant_id": "p5",
            "model": "test-model",
            "experiment_id": "exp-5",
            "project_id": "2026_03_08",
            "created_at": now,
            "custom_system_message_id": None,
            "multi_rounds": True,
            "messages": [
                {
                    "content": "message with all fields",
                    "role": "assistant",
                    "timestamp": now,
                    "type": "assistant",
                    "flagged": True,
                    "flagged_at": now,
                    "flagged_by": "participant-1",
                    "flag_category": "harassment",
                    "flag_other_reason": None,
                    "user_flag": {
                        "category": "harassment",
                        "category_other": "",
                        "reviews": [
                            {
                                "reviewer_id": "reviewer-1",
                                "reviewer_username": "alice",
                                "approved": True,
                                "comment": "Looks valid",
                                "reviewed_at": now,
                            }
                        ],
                    },
                    "reviewer_flags": [
                        {
                            "reviewer_id": "reviewer-2",
                            "reviewer_by_username": "bob",
                            "created_at": now,
                            "categories": ["hate"],
                            "category_other": "",
                            "comment": "Escalate",
                        }
                    ],
                    "duplicate_flags": [
                        {
                            "reviewer_id": "reviewer-3",
                            "reviewer_username": "carol",
                            "flagged_at": now,
                        }
                    ],
                }
            ],
            "opened_by": [
                {
                    "reviewer_id": "reviewer-4",
                    "opened_at": now,
                }
            ],
            "assigned_messages": [
                {
                    "reviewer_id": "reviewer-5",
                    "message_index": 0,
                    "reason": "participant_flag",
                    "assigned_at": now,
                }
            ],
            "reviewed_messages": [
                {
                    "reviewer_id": "reviewer-6",
                    "message_index": 0,
                    "reviewed_at": now,
                }
            ],
            "naturalness_ratings": [
                {
                    "reviewer_id": "reviewer-7",
                    "rated_at": now,
                    "coherence": 4,
                    "topic_progression": 3,
                }
            ],
            "realism_ratings": [
                {
                    "reviewer_id": "reviewer-8",
                    "rated_at": now,
                    "rating": 10,
                }
            ],
        }
    )

    assert document.id == document_id
    assert document.messages[0].user_flag.reviews[0].reviewer_username == "alice"
    assert document.messages[0].reviewer_flags[0].reviewer_by_username == "bob"
    assert document.messages[0].duplicate_flags[0].reviewer_username == "carol"
    assert document.assigned_messages[0].reason == "participant_flag"
    assert document.reviewed_messages[0].message_index == 0
    assert document.naturalness_ratings[0].coherence == 4
    assert document.realism_ratings[0].rating == 10


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        (
            lambda document: document.update({"reviewed_by": []}),
            "reviewed_by",
        ),
        (
            lambda document: document.update({"assigned_to": []}),
            "assigned_to",
        ),
        (
            lambda document: document["messages"][0]["user_flag"].update(
                {"created_at": datetime.now(timezone.utc)}
            ),
            "created_at",
        ),
        (
            lambda document: document["messages"][0]["user_flag"].update(
                {"created_by": "participant"}
            ),
            "created_by",
        ),
    ],
)
def test_conversation_document_rejects_legacy_fields(mutator, match):
    now = datetime.now(timezone.utc)
    document = _conversation_doc()
    document["messages"] = [_message_doc(now, content="strict message")]

    mutator(document)

    with pytest.raises(ValidationError, match=match):
        ConversationDocument.model_validate(document)


@pytest.mark.parametrize(
    "missing_field",
    [
        "reviewer_flags",
        "duplicate_flags",
    ],
)
def test_conversation_document_requires_message_review_fields(missing_field):
    document = _conversation_doc()
    del document["messages"][0][missing_field]

    with pytest.raises(ValidationError, match=missing_field):
        ConversationDocument.model_validate(document)


def test_conversation_document_requires_user_flag_field():
    document = _conversation_doc()
    del document["messages"][0]["user_flag"]

    with pytest.raises(ValidationError, match="user_flag"):
        ConversationDocument.model_validate(document)


@pytest.mark.parametrize("user_flag", [None, {}])
def test_conversation_document_accepts_empty_user_flag_states(user_flag):
    document = _conversation_doc()
    document["messages"][0]["user_flag"] = user_flag

    validated = ConversationDocument.model_validate(document)

    assert validated.messages[0].user_flag is None


def test_conversation_document_accepts_empty_flagged_by_string():
    document = _conversation_doc()
    document["messages"][0]["flagged_by"] = ""

    validated = ConversationDocument.model_validate(document)

    assert validated.messages[0].flagged_by == ""


@pytest.mark.parametrize("participant_id", ["", " "])
def test_conversation_document_normalizes_blank_participant_id(participant_id):
    document = _conversation_doc()
    document["participant_id"] = participant_id

    validated = ConversationDocument.model_validate(document)

    assert validated.participant_id is None


@pytest.mark.parametrize("missing_field", ["assigned_messages", "reviewed_messages"])
def test_conversation_document_requires_top_level_message_state_lists(missing_field):
    document = _conversation_doc()
    del document[missing_field]

    with pytest.raises(ValidationError, match=missing_field):
        ConversationDocument.model_validate(document)


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        (
            lambda document: document["assigned_messages"][0].update(
                {"message_index": -1}
            ),
            "message_index",
        ),
        (
            lambda document: document["reviewed_messages"][0].update(
                {"message_index": -1}
            ),
            "message_index",
        ),
        (
            lambda document: document["assigned_messages"][0].update(
                {"reason": "legacy_reason"}
            ),
            "reason",
        ),
    ],
)
def test_conversation_document_rejects_invalid_message_level_review_state(
    mutator, match
):
    now = datetime.now(timezone.utc)
    document = _conversation_doc()
    document["assigned_messages"] = [
        {
            "reviewer_id": "reviewer-1",
            "message_index": 0,
            "reason": "random_sample",
            "assigned_at": now,
        }
    ]
    document["reviewed_messages"] = [
        {
            "reviewer_id": "reviewer-2",
            "message_index": 0,
            "reviewed_at": now,
        }
    ]

    mutator(document)

    with pytest.raises(ValidationError, match=match):
        ConversationDocument.model_validate(document)


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        (
            lambda document: document["messages"][0]["user_flag"]["reviews"][0].update(
                {"reviewer_username": " "}
            ),
            "reviewer_username",
        ),
        (
            lambda document: document["messages"][0]["reviewer_flags"][0].update(
                {"reviewer_by_username": " "}
            ),
            "reviewer_by_username",
        ),
        (
            lambda document: document["messages"][0].update({"flagged_by": " "}),
            "flagged_by",
        ),
    ],
)
def test_conversation_document_rejects_blank_identity_fields(mutator, match):
    now = datetime.now(timezone.utc)
    document = _conversation_doc()
    document["messages"] = [
        {
            **_message_doc(now, content="message"),
            "flagged": True,
            "flagged_at": now,
            "flagged_by": "participant-1",
            "user_flag": {
                "category": "harassment",
                "category_other": "",
                "reviews": [
                    {
                        "reviewer_id": "reviewer-1",
                        "reviewer_username": "alice",
                        "approved": True,
                        "comment": "",
                        "reviewed_at": now,
                    }
                ],
            },
            "reviewer_flags": [
                {
                    "reviewer_id": "reviewer-2",
                    "reviewer_by_username": "bob",
                    "created_at": now,
                    "categories": [],
                    "category_other": "",
                    "comment": "",
                }
            ],
            "duplicate_flags": [
                {
                    "reviewer_id": "reviewer-3",
                    "reviewer_username": "carol",
                    "flagged_at": now,
                }
            ],
        }
    ]

    mutator(document)

    with pytest.raises(ValidationError, match=match):
        ConversationDocument.model_validate(document)


def test_conversation_document_requires_object_id():
    document = _conversation_doc()
    document["_id"] = "conversation-1"

    with pytest.raises(ValidationError, match="_id"):
        ConversationDocument.model_validate(document)
