"""MongoDB repository for conversation documents."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import logging
from typing import Any, Iterable, Optional, Set

from pydantic import ValidationError

from hpms.database.models import ConversationDocument, MessageDocument

SYSTEM_OPENAI_REVIEWER_ID = "system_openai_moderation"
SYSTEM_LLAMA_REVIEWER_ID = "system_llama_guard"
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class MessageBackfillTarget:
    """Message that still requires one or more system moderation writes."""

    conversation_id: Any
    message_index: int
    content: str
    missing_reviewer_ids: Set[str]


class MongoConversationRepository:
    """Encapsulates MongoDB reads/writes used by the realtime watcher."""

    def __init__(
        self,
        mongo_uri: str,
        database_name: str,
        collection_name: str = "Conversations",
    ) -> None:
        # Import lazily so tests can run without pymongo installed.
        # pylint: disable=import-outside-toplevel
        from pymongo import MongoClient

        self._client = MongoClient(mongo_uri)
        self.collection = self._client[database_name][collection_name]

    @classmethod
    def from_collection(cls, collection: Any) -> "MongoConversationRepository":
        """Construct repository from an already-created collection (tests)."""
        repo = cls.__new__(cls)
        repo._client = None
        repo.collection = collection
        return repo

    def close(self) -> None:
        """Close underlying MongoDB client if owned by this repository."""
        if self._client is not None:
            self._client.close()

    def watch(self, max_await_time_ms: int = 1000) -> Any:
        """Open a MongoDB change stream for conversation documents."""
        return self.collection.watch(
            full_document="updateLookup",
            max_await_time_ms=max_await_time_ms,
        )

    @staticmethod
    def validate_conversation_document(
        conversation: dict[str, Any],
    ) -> Optional[dict[str, Any]]:
        """Validate one conversation against the canonical schema."""
        try:
            validated = ConversationDocument.model_validate(conversation)
        except ValidationError as error:
            LOGGER.warning(
                "Skipping invalid conversation document %s: %s",
                conversation.get("_id"),
                error,
            )
            return None
        return validated.model_dump(by_alias=True)

    @staticmethod
    def missing_system_reviewers(message: dict[str, Any]) -> Set[str]:
        """Return system reviewer IDs that have not yet been persisted."""
        reviewer_flags = message.get("reviewer_flags", [])
        existing_reviewer_ids = {
            str(flag.get("reviewer_id", ""))
            for flag in reviewer_flags
            if isinstance(flag, dict)
        }

        missing: Set[str] = set()
        if SYSTEM_OPENAI_REVIEWER_ID not in existing_reviewer_ids:
            missing.add(SYSTEM_OPENAI_REVIEWER_ID)
        if SYSTEM_LLAMA_REVIEWER_ID not in existing_reviewer_ids:
            missing.add(SYSTEM_LLAMA_REVIEWER_ID)
        return missing

    def get_backfill_targets(
        self,
        batch_size: int = 200,
        excluded_provider_keys: Optional[Set[tuple[Any, int, str]]] = None,
    ) -> list[MessageBackfillTarget]:
        """Find messages that still need one or both system moderation writes."""
        targets: list[MessageBackfillTarget] = []
        excluded_provider_keys = excluded_provider_keys or set()

        cursor = self.collection.find(
            {},
            projection={
                "_id": 1,
                "participant_id": 1,
                "model": 1,
                "experiment_id": 1,
                "created_at": 1,
                "messages": 1,
                "opened_by": 1,
                "reviewed_by": 1,
                "assigned_to": 1,
            },
        )

        for raw_conversation in cursor:
            validated_conversation = self.validate_conversation_document(
                raw_conversation
            )
            if validated_conversation is None:
                continue

            conversation_id = validated_conversation.get("_id")
            messages = validated_conversation.get("messages", [])
            if not isinstance(messages, list):
                continue

            for message_index, message in enumerate(messages):
                if not isinstance(message, dict):
                    continue
                content = message.get("content", "")
                if not isinstance(content, str) or not content.strip():
                    continue

                missing = self.missing_system_reviewers(message)
                missing = {
                    reviewer_id
                    for reviewer_id in missing
                    if (conversation_id, message_index, reviewer_id)
                    not in excluded_provider_keys
                }
                if not missing:
                    continue

                targets.append(
                    MessageBackfillTarget(
                        conversation_id=conversation_id,
                        message_index=message_index,
                        content=content,
                        missing_reviewer_ids=missing,
                    )
                )
                if len(targets) >= batch_size:
                    return targets

        return targets

    def get_message(
        self, conversation_id: Any, message_index: int
    ) -> Optional[dict[str, Any]]:
        """Fetch a single message from a conversation document by index."""
        conversation = self.collection.find_one(
            {"_id": conversation_id},
            projection={"messages": 1},
        )
        if not conversation:
            return None

        messages = conversation.get("messages", [])
        if not isinstance(messages, list):
            return None

        if message_index < 0 or message_index >= len(messages):
            return None

        message = messages[message_index]
        if not isinstance(message, dict):
            return None

        try:
            validated_message = MessageDocument.model_validate(message)
        except ValidationError as error:
            LOGGER.warning(
                "Skipping invalid message document conversation=%s index=%s: %s",
                conversation_id,
                message_index,
                error,
            )
            return None

        return validated_message.model_dump()

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def upsert_system_reviewer_flag(
        self,
        conversation_id: Any,
        message_index: int,
        reviewer_id: str,
        categories: Iterable[str],
        category_other: str,
        created_at: Optional[datetime] = None,
    ) -> None:
        """Upsert one system reviewer flag per provider for one message."""
        timestamp = created_at or datetime.now(timezone.utc)
        normalized_categories = [str(category) for category in categories]

        reviewer_flag = {
            "reviewer_id": reviewer_id,
            "created_at": timestamp,
            "categories": normalized_categories,
            "category_other": category_other,
        }

        self.collection.update_one(
            {
                "_id": conversation_id,
                f"messages.{message_index}": {"$exists": True},
            },
            self._build_reviewer_flag_pipeline(
                message_index=message_index,
                reviewer_id=reviewer_id,
                reviewer_flag=reviewer_flag,
            ),
        )

    @staticmethod
    # pylint: disable=line-too-long
    def _build_reviewer_flag_pipeline(
        message_index: int,
        reviewer_id: str,
        reviewer_flag: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Build atomic pipeline to upsert one reviewer flag in messages[index]."""
        return [
            {
                "$set": {
                    # Use full-array mapping to avoid dotted array-index writes
                    # in pipeline updates (which can corrupt nested shape).
                    "messages": {
                        "$let": {
                            "vars": {
                                "target_index": message_index,
                                "new_flag": reviewer_flag,
                            },
                            "in": {
                                "$map": {
                                    "input": {"$range": [0, {"$size": "$messages"}]},
                                    "as": "idx",
                                    "in": {
                                        "$let": {
                                            "vars": {
                                                "message": {
                                                    "$arrayElemAt": [
                                                        "$messages",
                                                        "$$idx",
                                                    ]
                                                }
                                            },
                                            "in": {
                                                "$cond": [
                                                    {
                                                        "$eq": [
                                                            "$$idx",
                                                            "$$target_index",
                                                        ]
                                                    },
                                                    {
                                                        "$mergeObjects": [
                                                            "$$message",
                                                            {
                                                                "reviewer_flags": {
                                                                    "$let": {
                                                                        "vars": {
                                                                            "existing_flags": {
                                                                                    "$ifNull": [
                                                                                        "$$message.reviewer_flags",
                                                                                        [],
                                                                                    ]
                                                                            }
                                                                        },
                                                                        "in": {
                                                                            "$cond": [
                                                                                {
                                                                                    "$in": [
                                                                                        reviewer_id,
                                                                                        {
                                                                                            "$map": {
                                                                                                "input": "$$existing_flags",
                                                                                                "as": "flag",
                                                                                                "in": (
                                                                                                    "$$flag.reviewer_id"
                                                                                                ),
                                                                                            }
                                                                                        },
                                                                                    ]
                                                                                },
                                                                                {
                                                                                    "$map": {
                                                                                        "input": (
                                                                                            "$$existing_flags"
                                                                                        ),
                                                                                        "as": (
                                                                                            "flag"
                                                                                        ),
                                                                                        "in": {
                                                                                            "$cond": [
                                                                                                {
                                                                                                    "$eq": [
                                                                                                        "$$flag.reviewer_id",
                                                                                                        reviewer_id,
                                                                                                    ]
                                                                                                },
                                                                                                "$$new_flag",
                                                                                                "$$flag",
                                                                                            ]
                                                                                        },
                                                                                    }
                                                                                },
                                                                                {
                                                                                    "$concatArrays": [
                                                                                        "$$existing_flags",
                                                                                        ["$$new_flag"],
                                                                                    ]
                                                                                },
                                                                            ]
                                                                        },
                                                                    }
                                                                }
                                                            },
                                                        ]
                                                    },
                                                    "$$message",
                                                ]
                                            },
                                        }
                                    },
                                }
                            },
                        }
                    },
                }
            }
        ]
    # pylint: enable=line-too-long
