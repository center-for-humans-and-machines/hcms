"""Tests for Mongo conversation repository helpers."""

# pylint: disable=missing-function-docstring,too-few-public-methods,duplicate-code
# pylint: disable=too-many-lines

from __future__ import annotations

import operator
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import pytest
from bson import ObjectId
from pydantic import ValidationError

from hcms.database.models import ConversationDocument
from hcms.database.repository import (
    SYSTEM_LLAMA_REVIEWER_ID,
    SYSTEM_LLM_JUDGE_REVIEWER_ID,
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
        self.find_queries: list[dict[str, Any]] = []

    def find(
        self,
        query: dict[str, Any] | None = None,
        projection: dict[str, Any] | None = None,
        sort: list[tuple[str, int]] | None = None,
    ):
        _ = projection
        self.find_queries.append(deepcopy(query or {}))
        docs = [deepcopy(doc) for doc in self.docs.values()]
        if query:
            docs = [doc for doc in docs if self._matches(doc, query)]
        if sort:
            assert sort == [("_id", 1)], "only an ascending _id sort is supported"
            docs.sort(key=lambda doc: doc["_id"])
        return docs

    @classmethod
    def _matches(cls, doc: dict[str, Any], query: dict[str, Any]) -> bool:
        """Evaluate the subset of MongoDB query syntax the repository uses."""
        return all(
            cls._key_matches(doc, key, condition) for key, condition in query.items()
        )

    @classmethod
    def _key_matches(cls, doc: dict[str, Any], key: str, condition: Any) -> bool:
        if key == "$or":
            return any(cls._matches(doc, branch) for branch in condition)
        if key == "messages":
            assert set(condition) == {"$elemMatch"}
            return any(
                isinstance(message, dict)
                and cls._matches(message, condition["$elemMatch"])
                for message in doc.get("messages", [])
            )
        if key == "reviewer_flags.reviewer_id":
            reviewer_ids = [
                flag.get("reviewer_id")
                for flag in doc.get("reviewer_flags", [])
                if isinstance(flag, dict)
            ]
            return cls._field_matches(reviewer_ids, condition)
        return cls._field_matches(doc.get(key), condition)

    @classmethod
    def _field_matches(cls, value: Any, condition: Any) -> bool:
        if not isinstance(condition, dict):
            return value == condition
        return all(
            cls._operator_matches(value, query_operator, operand)
            for query_operator, operand in condition.items()
        )

    @staticmethod
    def _operator_matches(value: Any, query_operator: str, operand: Any) -> bool:
        if query_operator == "$ne":
            if isinstance(value, list):
                return operand not in value
            return value != operand
        if query_operator == "$type":
            assert operand == "string"
            return isinstance(value, str)
        comparisons = {"$gt": operator.gt, "$gte": operator.ge, "$lt": operator.lt}
        if query_operator not in comparisons:
            raise AssertionError(f"Unsupported query operator {query_operator}")
        return value is not None and comparisons[query_operator](value, operand)

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
        expected_content = query.get(f"messages.{message_index}.content")
        if expected_content is not None and message["content"] != expected_content:
            return _UpdateResult(0)
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


def test_get_backfill_targets_only_counts_requested_reviewers():
    conversation = _conversation_doc()
    repository = MongoConversationRepository.from_collection(
        FakeCollection([conversation])
    )

    # Message two already has its OpenAI entry, so an OpenAI-only watcher
    # must not see it, or its backfill would never run out of targets.
    openai_targets = repository.get_backfill_targets(
        batch_size=10, reviewer_ids={SYSTEM_OPENAI_REVIEWER_ID}
    )
    llama_targets = repository.get_backfill_targets(
        batch_size=10, reviewer_ids={SYSTEM_LLAMA_REVIEWER_ID}
    )

    assert [(t.message_index, t.missing_reviewer_ids) for t in openai_targets] == [
        (0, {SYSTEM_OPENAI_REVIEWER_ID})
    ]
    assert [(t.message_index, t.missing_reviewer_ids) for t in llama_targets] == [
        (0, {SYSTEM_LLAMA_REVIEWER_ID}),
        (1, {SYSTEM_LLAMA_REVIEWER_ID}),
    ]


def test_get_backfill_targets_carries_reply_context():
    now = datetime.now(timezone.utc)
    conversation = _conversation_doc()
    conversation["messages"] = [
        _message_doc(now, content="system prompt", role="system"),
        _message_doc(now, content="Hello!"),
        _message_doc(now, content="How do I pick a lock?", role="user"),
        _message_doc(now, content="Use a tension wrench."),
    ]
    repository = MongoConversationRepository.from_collection(
        FakeCollection([conversation])
    )

    targets = repository.get_backfill_targets(batch_size=10)

    assert [(t.message_index, t.role, t.previous_user_content) for t in targets] == [
        (1, "assistant", ""),
        (2, "user", ""),
        (3, "assistant", "How do I pick a lock?"),
    ]


@pytest.mark.parametrize(
    ("message_index", "expected"),
    [
        (0, ""),  # user message: no context
        (1, "first question"),
        (2, "first question"),  # follow-up still answers the same question
        (5, "second question"),  # skips the blank user message
        (6, ""),  # out of range
    ],
)
def test_previous_user_content_finds_closest_earlier_user_message(
    message_index, expected
):
    messages = [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first reply"},
        {"role": "assistant", "content": "follow-up"},
        {"role": "user", "content": "second question"},
        {"role": "user", "content": "   "},
        {"role": "assistant", "content": "second reply"},
    ]

    assert (
        MongoConversationRepository.previous_user_content(messages, message_index)
        == expected
    )


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


def test_conversation_document_backfills_legacy_naturalness_topic_progression():
    now = datetime.now(timezone.utc)
    document = _conversation_doc()
    document["naturalness_ratings"] = [
        {
            "reviewer_id": "reviewer-7",
            "rated_at": now,
            "coherence": 3,
        }
    ]

    validated = ConversationDocument.model_validate(document)

    assert validated.naturalness_ratings[0].coherence == 3
    assert validated.naturalness_ratings[0].topic_progression == 3


def test_validate_conversation_document_accepts_legacy_naturalness_rating():
    now = datetime.now(timezone.utc)
    document = _conversation_doc()
    document["naturalness_ratings"] = [
        {
            "reviewer_id": "reviewer-7",
            "rated_at": now,
            "coherence": 3,
        }
    ]

    validated = MongoConversationRepository.validate_conversation_document(document)

    assert validated is not None
    assert validated["naturalness_ratings"][0]["coherence"] == 3
    assert validated["naturalness_ratings"][0]["topic_progression"] == 3


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


def test_conversation_document_accepts_llm_flag_assignment_reason():
    now = datetime.now(timezone.utc)
    document = _conversation_doc()
    document["assigned_messages"] = [
        {
            "reviewer_id": "reviewer-1",
            "message_index": 0,
            "reason": "llm_flag",
            "assigned_at": now,
        }
    ]

    validated = ConversationDocument.model_validate(document)

    assert validated.assigned_messages[0].reason == "llm_flag"


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


@pytest.mark.parametrize("category_other", [None, ""])
def test_reviewer_flag_accepts_null_and_empty_category_other(category_other):
    now = datetime.now(timezone.utc)
    document = _conversation_doc()
    document["messages"] = [
        {
            **_message_doc(now, content="message"),
            "reviewer_flags": [
                {
                    "reviewer_id": "reviewer-1",
                    "reviewer_by_username": "alice",
                    "created_at": now,
                    "categories": [],
                    "category_other": category_other,
                    "comment": "",
                }
            ],
        }
    ]

    validated = ConversationDocument.model_validate(document)

    assert validated.messages[0].reviewer_flags[0].category_other == category_other


def test_reviewer_flag_category_other_defaults_to_none_when_omitted():
    now = datetime.now(timezone.utc)
    document = _conversation_doc()
    document["messages"] = [
        {
            **_message_doc(now, content="message"),
            "reviewer_flags": [
                {
                    "reviewer_id": "reviewer-1",
                    "reviewer_by_username": "alice",
                    "created_at": now,
                    "categories": [],
                    "comment": "",
                }
            ],
        }
    ]

    validated = ConversationDocument.model_validate(document)

    assert validated.messages[0].reviewer_flags[0].category_other is None


@pytest.mark.parametrize("comment", [None, ""])
def test_reviewer_flag_accepts_null_and_empty_comment(comment):
    now = datetime.now(timezone.utc)
    document = _conversation_doc()
    document["messages"] = [
        {
            **_message_doc(now, content="message"),
            "reviewer_flags": [
                {
                    "reviewer_id": "reviewer-1",
                    "reviewer_by_username": "alice",
                    "created_at": now,
                    "categories": [],
                    "category_other": "",
                    "comment": comment,
                }
            ],
        }
    ]

    validated = ConversationDocument.model_validate(document)

    assert validated.messages[0].reviewer_flags[0].comment == comment


def _all_system_flags(now: datetime) -> list[dict[str, Any]]:
    return [
        {
            "reviewer_id": reviewer_id,
            "reviewer_by_username": reviewer_id,
            "created_at": now,
            "categories": [],
            "category_other": "",
            "comment": "",
        }
        for reviewer_id in (SYSTEM_OPENAI_REVIEWER_ID, SYSTEM_LLAMA_REVIEWER_ID)
    ]


def test_missing_system_reviewers_default_leaves_the_llm_judge_opt_in():
    now = datetime.now(timezone.utc)
    message = _message_doc(now, content="hello", reviewer_flags=_all_system_flags(now))

    assert MongoConversationRepository.missing_system_reviewers(message) == set()
    assert MongoConversationRepository.missing_system_reviewers(
        message, {SYSTEM_LLM_JUDGE_REVIEWER_ID}
    ) == {SYSTEM_LLM_JUDGE_REVIEWER_ID}


@pytest.mark.parametrize(
    ("message_index", "limit", "expected"),
    [
        (0, 4, ()),
        (1, 4, (("user", "first question"),)),
        # system turns and blank turns are skipped, order is kept
        (4, 4, (("user", "first question"), ("assistant", "first reply"))),
        # user targets get context too, capped at the limit, most recent kept
        (6, 2, (("user", "second question"), ("assistant", "second reply"))),
        (6, 0, ()),
        (7, 4, ()),  # out of range
    ],
)
def test_context_messages_returns_earlier_non_system_turns(
    message_index, limit, expected
):
    messages = [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first reply"},
        {"role": "system", "content": "system note"},
        {"role": "assistant", "content": "   "},
        {"role": "user", "content": "second question"},
        {"role": "assistant", "content": "second reply"},
        {"role": "user", "content": "third question"},
    ]

    assert (
        MongoConversationRepository.context_messages(messages, message_index, limit)
        == expected
    )


def test_get_backfill_targets_carries_context_messages():
    now = datetime.now(timezone.utc)
    conversation = _conversation_doc()
    conversation["messages"] = [
        _message_doc(now, content="system prompt", role="system"),
        _message_doc(now, content="Hello", role="user"),
        _message_doc(now, content="Hi!"),
        _message_doc(now, content="I want to die", role="user"),
    ]
    repository = MongoConversationRepository.from_collection(
        FakeCollection([conversation])
    )

    targets = repository.get_backfill_targets(batch_size=10)

    assert [(t.message_index, t.context_messages) for t in targets] == [
        (1, ()),
        (2, (("user", "Hello"),)),
        (3, (("user", "Hello"), ("assistant", "Hi!"))),
    ]


def test_backfill_window_uses_message_timestamps_not_conversation_creation():
    since = datetime(2026, 11, 3, tzinfo=timezone.utc)
    before = datetime(2026, 10, 1, tzinfo=timezone.utc)
    after = datetime(2026, 11, 4, tzinfo=timezone.utc)

    old_conversation = _conversation_doc()
    old_conversation["created_at"] = before
    old_conversation["messages"] = [
        _message_doc(before, content="old message", role="user"),
        # A message added to an old conversation after go-live.
        _message_doc(after, content="new message", role="user"),
    ]
    only_old = _conversation_doc()
    only_old["created_at"] = before
    only_old["messages"] = [_message_doc(before, content="older message", role="user")]
    collection = FakeCollection([old_conversation, only_old])
    repository = MongoConversationRepository.from_collection(collection)

    targets = repository.get_backfill_targets(
        batch_size=10, reviewer_ids={SYSTEM_LLM_JUDGE_REVIEWER_ID}, since=since
    )

    assert [(t.document_id, t.content) for t in targets] == [
        (old_conversation["_id"], "new message")
    ]
    assert collection.find_queries[-1]["$or"][0]["messages"]["$elemMatch"][
        "timestamp"
    ] == {"$gte": since}


def test_backfill_window_until_excludes_later_messages():
    until = datetime(2026, 11, 3, tzinfo=timezone.utc)
    conversation = _conversation_doc()
    conversation["messages"] = [
        _message_doc(datetime(2026, 11, 2, tzinfo=timezone.utc), content="before"),
        _message_doc(datetime(2026, 11, 3, tzinfo=timezone.utc), content="at"),
    ]
    repository = MongoConversationRepository.from_collection(
        FakeCollection([conversation])
    )

    targets = repository.get_backfill_targets(batch_size=10, until=until)

    assert [t.content for t in targets] == ["before"]


def test_backfill_query_skips_conversations_rated_by_the_requested_providers():
    now = datetime.now(timezone.utc)
    rated = _conversation_doc()
    rated["messages"] = [
        _message_doc(now, content="rated", reviewer_flags=_all_system_flags(now))
    ]
    unrated = _conversation_doc()
    collection = FakeCollection([rated, unrated])
    repository = MongoConversationRepository.from_collection(collection)

    batch = repository.next_backfill_batch(batch_size=10)

    assert {t.document_id for t in batch.targets} == {unrated["_id"]}
    query = collection.find_queries[-1]
    assert "_id" not in query
    assert [
        branch["messages"]["$elemMatch"]["reviewer_flags.reviewer_id"]
        for branch in query["$or"]
    ] == [{"$ne": SYSTEM_LLAMA_REVIEWER_ID}, {"$ne": SYSTEM_OPENAI_REVIEWER_ID}]
    assert query["$or"][0]["messages"]["$elemMatch"]["role"] == {"$ne": "system"}


def test_next_backfill_batch_resumes_after_the_last_complete_conversation():
    now = datetime.now(timezone.utc)
    conversations = []
    for name in ("a", "b", "c"):
        conversation = _conversation_doc()
        conversation["messages"] = [_message_doc(now, content=name, role="user")]
        conversations.append(conversation)
    collection = FakeCollection(conversations)
    repository = MongoConversationRepository.from_collection(collection)
    reviewer_ids = {SYSTEM_LLAMA_REVIEWER_ID}

    first = repository.next_backfill_batch(batch_size=2, reviewer_ids=reviewer_ids)

    assert [t.content for t in first.targets] == ["a", "b"]
    # The batch filled up inside "b", so the scan resumes with "b" again.
    assert first.resume_after == conversations[0]["_id"]

    for target in first.targets:
        repository.upsert_system_reviewer_flag(
            document_id=target.document_id,
            message_index=target.message_index,
            reviewer_id=SYSTEM_LLAMA_REVIEWER_ID,
            categories=[],
            category_other="",
        )

    second = repository.next_backfill_batch(
        batch_size=2, reviewer_ids=reviewer_ids, after_id=first.resume_after
    )

    assert [t.content for t in second.targets] == ["c"]
    assert second.resume_after is None
    assert collection.find_queries[-1]["_id"] == {"$gt": conversations[0]["_id"]}

    repository.upsert_system_reviewer_flag(
        document_id=second.targets[0].document_id,
        message_index=0,
        reviewer_id=SYSTEM_LLAMA_REVIEWER_ID,
        categories=[],
        category_other="",
    )
    third = repository.next_backfill_batch(batch_size=2, reviewer_ids=reviewer_ids)

    assert not third.targets
    assert third.resume_after is None


def test_upsert_system_reviewer_flag_writes_comment():
    conversation = _conversation_doc()
    collection = FakeCollection([conversation])
    repository = MongoConversationRepository.from_collection(collection)

    written = repository.upsert_system_reviewer_flag(
        document_id=conversation["_id"],
        message_index=0,
        reviewer_id=SYSTEM_LLM_JUDGE_REVIEWER_ID,
        categories=["Suicidal ideation"],
        category_other="",
        comment="states a wish to die",
        expected_content="message one",
    )

    assert written is True
    (flag,) = collection.docs[conversation["_id"]]["messages"][0]["reviewer_flags"]
    assert flag["reviewer_id"] == SYSTEM_LLM_JUDGE_REVIEWER_ID
    assert flag["categories"] == ["Suicidal ideation"]
    assert flag["comment"] == "states a wish to die"


def test_upsert_system_reviewer_flag_skips_a_message_whose_content_changed():
    conversation = _conversation_doc()
    collection = FakeCollection([conversation])
    repository = MongoConversationRepository.from_collection(collection)

    written = repository.upsert_system_reviewer_flag(
        document_id=conversation["_id"],
        message_index=0,
        reviewer_id=SYSTEM_LLM_JUDGE_REVIEWER_ID,
        categories=["Suicidal ideation"],
        category_other="",
        expected_content="message one, as it was before the merge",
    )

    assert written is False
    assert collection.docs[conversation["_id"]]["messages"][0]["reviewer_flags"] == []
