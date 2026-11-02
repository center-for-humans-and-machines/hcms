"""MongoDB repository for conversation documents."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Optional, Set

from pydantic import ValidationError

from hcms.constants import ASSISTANT, SYSTEM, USER
from hcms.database.models import ConversationDocument, MessageDocument

SYSTEM_OPENAI_REVIEWER_ID = "system_openai_moderation"
SYSTEM_LLAMA_REVIEWER_ID = "system_llama_guard"
SYSTEM_LLM_JUDGE_REVIEWER_ID = "system_llm_judge"
# Every system reviewer a watcher may run.
SYSTEM_REVIEWER_IDS: frozenset[str] = frozenset(
    {SYSTEM_OPENAI_REVIEWER_ID, SYSTEM_LLAMA_REVIEWER_ID, SYSTEM_LLM_JUDGE_REVIEWER_ID}
)
# Reviewers run when none are named. The LLM judge is opt-in, so a watcher
# that predates it never needs the judge's endpoint or credentials.
DEFAULT_SYSTEM_REVIEWER_IDS: frozenset[str] = frozenset(
    {SYSTEM_OPENAI_REVIEWER_ID, SYSTEM_LLAMA_REVIEWER_ID}
)
# Earlier turns handed to raters that judge a message in its conversation.
CONTEXT_MESSAGE_LIMIT = 4
LOGGER = logging.getLogger(__name__)

# (role, content) of earlier turns, oldest first.
ContextMessages = tuple[tuple[str, str], ...]

_BACKFILL_PROJECTION = {
    "_id": 1,
    "participant_id": 1,
    "model": 1,
    "experiment_id": 1,
    "conversation_id": 1,
    "project_id": 1,
    "created_at": 1,
    "custom_system_message_id": 1,
    "multi_rounds": 1,
    "messages": 1,
    "opened_by": 1,
    "reviewed_messages": 1,
    "assigned_messages": 1,
}


def _as_utc(value: Any) -> Optional[datetime]:
    """Return a timezone-aware UTC datetime, treating naive values as UTC."""
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


@dataclass(frozen=True)
class MessageBackfillTarget:  # pylint: disable=too-many-instance-attributes
    """Message that still requires one or more system moderation writes.

    `content` is the exact text that gets rated; the flag is only written while
    the message still has it. For assistant messages, `previous_user_content`
    holds the user message the reply answers (Llama Guard rates the pair), and
    `context_messages` holds the previous turns for raters that judge any
    message in context. `message_timestamp` is when the message was written,
    so the watcher can report how long its rating took to land.
    """

    document_id: Any
    message_index: int
    content: str
    missing_reviewer_ids: Set[str]
    role: str = ""
    previous_user_content: str = ""
    message_timestamp: Optional[datetime] = None
    context_messages: ContextMessages = ()


@dataclass(frozen=True)
class BackfillBatch:
    """One page of a resumable backfill scan.

    `resume_after` is the conversation `_id` to continue from on the next call;
    `None` means the scan reached the end and the next call starts over. An
    empty `targets` list means nothing is left to rate.
    """

    targets: list[MessageBackfillTarget]
    resume_after: Any = None


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
    def missing_system_reviewers(
        message: dict[str, Any],
        reviewer_ids: Optional[Set[str]] = None,
    ) -> Set[str]:
        """Return system reviewer IDs that have not yet been persisted.

        Only `reviewer_ids` are considered (default: the providers run without
        configuration, see `DEFAULT_SYSTEM_REVIEWER_IDS`), so a watcher running
        a subset of providers ignores the others' entries.
        """
        if not MongoConversationRepository.is_message_eligible_for_system_reviewers(
            message
        ):
            return set()

        reviewer_flags = message.get("reviewer_flags", [])
        existing_reviewer_ids = {
            str(flag.get("reviewer_id", ""))
            for flag in reviewer_flags
            if isinstance(flag, dict)
        }

        expected_reviewer_ids = (
            DEFAULT_SYSTEM_REVIEWER_IDS if reviewer_ids is None else reviewer_ids
        )
        return {
            reviewer_id
            for reviewer_id in expected_reviewer_ids
            if reviewer_id not in existing_reviewer_ids
        }

    @staticmethod
    def context_messages(
        messages: list[Any], message_index: int, limit: int = CONTEXT_MESSAGE_LIMIT
    ) -> ContextMessages:
        """Return up to `limit` earlier turns of any role but system, oldest first.

        Blank turns are skipped. Raters that judge a message in context get these
        together with the message itself, whatever its role.
        """
        if limit <= 0 or not 0 <= message_index < len(messages):
            return ()

        turns: list[tuple[str, str]] = []
        for earlier in messages[:message_index]:
            if not isinstance(earlier, dict):
                continue
            role = earlier.get("role")
            content = earlier.get("content")
            if not isinstance(role, str) or not role.strip() or role.strip() == SYSTEM:
                continue
            if not isinstance(content, str) or not content.strip():
                continue
            turns.append((role, content))
        return tuple(turns[-limit:])

    @staticmethod
    def previous_user_content(messages: list[Any], message_index: int) -> str:
        """Return the user message an assistant reply answers, or "".

        Only assistant messages get context: the closest earlier non-empty user
        message. Other roles, and replies with no user message before them,
        return "".
        """
        if not 0 <= message_index < len(messages):
            return ""
        message = messages[message_index]
        if not isinstance(message, dict) or message.get("role") != ASSISTANT:
            return ""

        for earlier in reversed(messages[:message_index]):
            if not isinstance(earlier, dict) or earlier.get("role") != USER:
                continue
            content = earlier.get("content", "")
            if isinstance(content, str) and content.strip():
                return content
        return ""

    @staticmethod
    def is_message_eligible_for_system_reviewers(message: dict[str, Any]) -> bool:
        """Return whether a message should be sent to automated reviewers."""
        content = message.get("content", "")
        if not isinstance(content, str) or not content.strip():
            return False

        role = message.get("role", "")
        if not isinstance(role, str):
            return False

        return role.strip() != SYSTEM

    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    def get_backfill_targets(
        self,
        batch_size: int = 200,
        excluded_provider_keys: Optional[Set[tuple[Any, int, str]]] = None,
        reviewer_ids: Optional[Set[str]] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
    ) -> list[MessageBackfillTarget]:
        """Find messages that still need system moderation writes, from the start.

        Convenience wrapper around `next_backfill_batch` for callers that don't
        resume a scan.
        """
        return self.next_backfill_batch(
            batch_size=batch_size,
            excluded_provider_keys=excluded_provider_keys,
            reviewer_ids=reviewer_ids,
            since=since,
            until=until,
        ).targets

    def next_backfill_batch(
        self,
        batch_size: int = 200,
        excluded_provider_keys: Optional[Set[tuple[Any, int, str]]] = None,
        reviewer_ids: Optional[Set[str]] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        after_id: Any = None,
    ) -> BackfillBatch:
        """Return the next batch of messages missing system moderation writes.

        MongoDB only returns conversations that still hold an eligible message
        without an entry of one of `reviewer_ids` (default:
        `DEFAULT_SYSTEM_REVIEWER_IDS`), ordered by `_id`, and `after_id` continues
        a scan where the previous batch stopped, so a batch costs about as much
        as the messages it returns instead of a scan of the whole collection.
        `since`/`until` bound the message `timestamp` (not the conversation's
        creation), so a new message in an old conversation is still found.
        """
        excluded_provider_keys = excluded_provider_keys or set()
        expected_reviewer_ids = (
            set(DEFAULT_SYSTEM_REVIEWER_IDS)
            if reviewer_ids is None
            else set(reviewer_ids)
        )
        targets: list[MessageBackfillTarget] = []

        cursor = self.collection.find(
            self._backfill_query(expected_reviewer_ids, since, until, after_id),
            projection=_BACKFILL_PROJECTION,
            sort=[("_id", 1)],
        )

        # The `_id` of the last conversation scanned completely: a batch that
        # fills up in the middle of a conversation resumes with that conversation.
        previous_document_id = after_id
        for raw_conversation in cursor:
            document_id = raw_conversation.get("_id")
            validated_conversation = self.validate_conversation_document(
                raw_conversation
            )
            if validated_conversation is None:
                previous_document_id = document_id
                continue

            document_id = validated_conversation.get("_id")
            messages = validated_conversation.get("messages", [])
            if not isinstance(messages, list):
                previous_document_id = document_id
                continue

            for message_index, message in enumerate(messages):
                if not isinstance(message, dict):
                    continue
                if not self.is_message_eligible_for_system_reviewers(message):
                    continue
                if not self._message_in_window(message, since, until):
                    continue
                content = message["content"]

                missing = self.missing_system_reviewers(message, expected_reviewer_ids)
                missing = {
                    reviewer_id
                    for reviewer_id in missing
                    if (document_id, message_index, reviewer_id)
                    not in excluded_provider_keys
                }
                if not missing:
                    continue

                targets.append(
                    MessageBackfillTarget(
                        document_id=document_id,
                        message_index=message_index,
                        content=content,
                        missing_reviewer_ids=missing,
                        role=message.get("role", ""),
                        previous_user_content=self.previous_user_content(
                            messages, message_index
                        ),
                        message_timestamp=message.get("timestamp"),
                        context_messages=self.context_messages(messages, message_index),
                    )
                )
                if len(targets) >= batch_size:
                    return BackfillBatch(
                        targets=targets, resume_after=previous_document_id
                    )

            previous_document_id = document_id

        return BackfillBatch(targets=targets, resume_after=None)

    # pylint: enable=too-many-arguments,too-many-positional-arguments,too-many-locals

    @staticmethod
    def _backfill_query(
        reviewer_ids: Set[str],
        since: Optional[datetime],
        until: Optional[datetime],
        after_id: Any,
    ) -> dict[str, Any]:
        """Match conversations with an eligible message lacking one of the reviewers."""
        window: dict[str, Any] = {}
        if since is not None:
            window["$gte"] = since
        if until is not None:
            window["$lt"] = until

        unrated_message: dict[str, Any] = {
            "role": {"$ne": SYSTEM},
            "content": {"$type": "string", "$ne": ""},
        }
        if window:
            unrated_message["timestamp"] = window

        query: dict[str, Any] = {
            "$or": [
                {
                    "messages": {
                        "$elemMatch": {
                            **unrated_message,
                            "reviewer_flags.reviewer_id": {"$ne": reviewer_id},
                        }
                    }
                }
                for reviewer_id in sorted(reviewer_ids)
            ]
        }
        if after_id is not None:
            query["_id"] = {"$gt": after_id}
        return query

    @staticmethod
    def _message_in_window(
        message: dict[str, Any],
        since: Optional[datetime],
        until: Optional[datetime],
    ) -> bool:
        """Return whether the message timestamp lies in [since, until)."""
        if since is None and until is None:
            return True
        timestamp = _as_utc(message.get("timestamp"))
        if timestamp is None:
            return False
        since_utc = _as_utc(since)
        until_utc = _as_utc(until)
        if since_utc is not None and timestamp < since_utc:
            return False
        if until_utc is not None and timestamp >= until_utc:
            return False
        return True

    def get_message(
        self, document_id: Any, message_index: int
    ) -> Optional[dict[str, Any]]:
        """Fetch a single message from a conversation document by index."""
        conversation = self.collection.find_one(
            {"_id": document_id},
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
                "Skipping invalid message document document_id=%s index=%s: %s",
                document_id,
                message_index,
                error,
            )
            return None

        return validated_message.model_dump()

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def upsert_system_reviewer_flag(
        self,
        document_id: Any,
        message_index: int,
        reviewer_id: str,
        categories: Iterable[str],
        category_other: str,
        created_at: Optional[datetime] = None,
        comment: str = "",
        expected_content: Optional[str] = None,
    ) -> bool:
        """Upsert one system reviewer flag per provider for one message.

        With `expected_content`, the flag is only written while the message still
        has the content that was rated; a message edited in the meantime keeps
        its old entries and is rated again from its content-change event.

        Returns:
            bool: Whether the message matched and the flag was written.
        """
        timestamp = created_at or datetime.now(timezone.utc)
        normalized_categories = [str(category) for category in categories]

        reviewer_flag = {
            "reviewer_id": reviewer_id,
            "reviewer_by_username": reviewer_id,
            "created_at": timestamp,
            "categories": normalized_categories,
            "category_other": category_other,
            "comment": comment,
        }

        query: dict[str, Any] = {
            "_id": document_id,
            f"messages.{message_index}": {"$exists": True},
        }
        if expected_content is not None:
            query[f"messages.{message_index}.content"] = expected_content

        result = self.collection.update_one(
            query,
            self._build_reviewer_flag_pipeline(
                message_index=message_index,
                reviewer_id=reviewer_id,
                reviewer_flag=reviewer_flag,
            ),
        )
        matched = getattr(result, "matched_count", None)
        if matched == 0:
            LOGGER.warning(
                "Skipped %s flag for document_id=%s message_index=%s: "
                "message missing or changed since it was rated",
                reviewer_id,
                document_id,
                message_index,
            )
            return False
        return True

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
                                                                                        [
                                                                                            "$$new_flag"
                                                                                        ],
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
