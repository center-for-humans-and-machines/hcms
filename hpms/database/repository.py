"""MongoDB repository for conversation documents."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Optional, Set

from pydantic import ValidationError

from hpms.database.models import ConversationDocument, MessageDocument

SYSTEM_OPENAI_REVIEWER_ID = "system_openai_moderation"
SYSTEM_LLAMA_REVIEWER_ID = "system_llama_guard"


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
            print(
                "Skipping invalid conversation document "
                f"{conversation.get('_id')}: {error}"
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
        self, batch_size: int = 200
    ) -> list[MessageBackfillTarget]:
        """Find messages that still need one or both system moderation writes."""
        targets: list[MessageBackfillTarget] = []

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
            print(
                "Skipping invalid message document "
                f"conversation={conversation_id} index={message_index}: {error}"
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

        # Update existing entry if present.
        update_existing = self.collection.update_one(
            {
                "_id": conversation_id,
                f"messages.{message_index}.reviewer_flags.reviewer_id": reviewer_id,
            },
            {
                "$set": {
                    "messages."
                    f"{message_index}.reviewer_flags.$[flag].created_at": timestamp,
                    "messages."
                    f"{message_index}.reviewer_flags.$[flag].categories": normalized_categories,
                    "messages."
                    f"{message_index}.reviewer_flags.$[flag].category_other": category_other,
                }
            },
            array_filters=[{"flag.reviewer_id": reviewer_id}],
        )

        if update_existing.matched_count > 0:
            return

        # Insert only if there is no existing flag for this reviewer_id at message index.
        self.collection.update_one(
            {
                "_id": conversation_id,
                f"messages.{message_index}.reviewer_flags.reviewer_id": {
                    "$ne": reviewer_id
                },
            },
            {"$push": {f"messages.{message_index}.reviewer_flags": reviewer_flag}},
        )
