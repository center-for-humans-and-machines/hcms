"""Realtime MongoDB watcher that rates newly added messages."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Callable, Optional, Set

from pydantic import ValidationError

from hpms.database.models import ConversationDocument
from hpms.database.repository import (
    SYSTEM_LLAMA_REVIEWER_ID,
    SYSTEM_OPENAI_REVIEWER_ID,
    MessageBackfillTarget,
    MongoConversationRepository,
)
from hpms.rating.llama_guard import _rate_text_with_llama_guard
from hpms.rating.openai_moderation import _rate_text_with_openai_moderation

KNOWN_LLAMA_GUARD_CATEGORIES = {
    "Violent Crimes",
    "Non-Violent Crimes",
    "Sex-Related Crimes",
    "Child Sexual Exploitation",
    "Defamation",
    "Specialized Advice",
    "Privacy",
    "Intellectual Property",
    "Indiscriminate Weapons",
    "Hate",
    "Suicide & Self-Harm",
    "Sexual Content",
    "Elections",
    "Code Interpreter Abuse",
}


def normalize_openai_categories(raw_result: Any) -> tuple[list[str], str]:
    """Normalize OpenAI moderation output into reviewer flag fields."""
    if not isinstance(raw_result, list):
        return [], str(raw_result)

    categories: list[str] = []
    has_unexpected_value = False

    for item in raw_result:
        if isinstance(item, str):
            category = item.strip()
            if category and category != "0":
                categories.append(category)
            continue

        if isinstance(item, int):
            if item != 0:
                has_unexpected_value = True
            continue

        has_unexpected_value = True

    # Deduplicate while preserving order.
    categories = list(dict.fromkeys(categories))

    if categories:
        return categories, ""

    if has_unexpected_value:
        return [], str(raw_result)

    return [], ""


def normalize_llama_guard_categories(raw_result: Any) -> tuple[list[str], str]:
    """Normalize Llama Guard output into reviewer flag fields."""
    if not isinstance(raw_result, str):
        return [], str(raw_result)

    result = raw_result.strip()
    if result in {"", "0"}:
        return [], ""

    if result in KNOWN_LLAMA_GUARD_CATEGORIES:
        return [result], ""

    return [], result


class RealtimeConversationWatcher:
    """Watches MongoDB conversation changes and persists moderation ratings."""

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(
        self,
        repository: MongoConversationRepository,
        openai_rater: Callable[[str], Any] = _rate_text_with_openai_moderation,
        llama_guard_rater: Callable[[str], Any] = _rate_text_with_llama_guard,
        backfill_batch_size: int = 200,
        change_stream_max_await_ms: int = 1000,
        reconnect_sleep_seconds: float = 1.0,
    ) -> None:
        self.repository = repository
        self.openai_rater = openai_rater
        self.llama_guard_rater = llama_guard_rater
        self.backfill_batch_size = backfill_batch_size
        self.change_stream_max_await_ms = change_stream_max_await_ms
        self.reconnect_sleep_seconds = reconnect_sleep_seconds

    def run(self) -> None:
        """Run startup backfill once, then keep processing change events."""
        self.run_startup_backfill()

        while True:
            try:
                with self.repository.watch(
                    max_await_time_ms=self.change_stream_max_await_ms
                ) as stream:
                    for change_event in stream:
                        self.handle_change_event(change_event)
            except Exception as error:  # pylint: disable=broad-exception-caught
                # pragma: no cover - network/runtime path
                print(f"Change stream interrupted: {error}")
                time.sleep(self.reconnect_sleep_seconds)

    def run_startup_backfill(self) -> None:
        """Backfill unrated messages present before the watcher started."""
        seen_targets: Set[tuple[Any, int]] = set()

        while True:
            targets = self.repository.get_backfill_targets(
                batch_size=self.backfill_batch_size
            )
            if not targets:
                return

            progress_made = False
            for target in targets:
                target_key = (target.conversation_id, target.message_index)
                if target_key in seen_targets:
                    continue
                seen_targets.add(target_key)

                self.process_message_target(target)
                progress_made = True

            if not progress_made:
                return

    def handle_change_event(self, change_event: dict[str, Any]) -> None:
        """Handle one MongoDB change stream event."""
        full_document = change_event.get("fullDocument")
        if not isinstance(full_document, dict):
            return

        validated_document = self._validate_full_document(full_document)
        if validated_document is None:
            return

        messages = validated_document.get("messages", [])
        if not isinstance(messages, list):
            return

        message_indexes = self.extract_candidate_message_indexes(change_event, messages)

        for message_index in sorted(message_indexes):
            if message_index < 0 or message_index >= len(messages):
                continue

            message = messages[message_index]
            content = (
                message.get("content", "").strip()
                if isinstance(message, dict)
                and isinstance(message.get("content", ""), str)
                else ""
            )
            if not content:
                continue

            missing_reviewer_ids = self.repository.missing_system_reviewers(message)
            if not missing_reviewer_ids:
                continue

            target = MessageBackfillTarget(
                conversation_id=validated_document.get("_id"),
                message_index=message_index,
                content=content,
                missing_reviewer_ids=missing_reviewer_ids,
            )
            self.process_message_target(target)

    def _validate_full_document(
        self, full_document: dict[str, Any]
    ) -> Optional[dict[str, Any]]:
        """Validate changed conversation document against canonical schema."""
        repository_validator = getattr(
            self.repository, "validate_conversation_document", None
        )
        if callable(repository_validator):
            return repository_validator(full_document)

        try:
            validated = ConversationDocument.model_validate(full_document)
        except ValidationError as error:
            print(
                "Skipping invalid conversation document from change stream "
                f"{full_document.get('_id')}: {error}"
            )
            return None
        return validated.model_dump(by_alias=True)

    def process_message_target(self, target: MessageBackfillTarget) -> None:
        """Rate one message and persist missing system reviewer flags."""
        if SYSTEM_OPENAI_REVIEWER_ID in target.missing_reviewer_ids:
            self._persist_openai_flag(target)

        if SYSTEM_LLAMA_REVIEWER_ID in target.missing_reviewer_ids:
            self._persist_llama_guard_flag(target)

    def _persist_openai_flag(self, target: MessageBackfillTarget) -> None:
        """Rate with OpenAI moderation and write into reviewer_flags."""
        try:
            raw_result = self.openai_rater(target.content)
            categories, category_other = normalize_openai_categories(raw_result)
            self.repository.upsert_system_reviewer_flag(
                conversation_id=target.conversation_id,
                message_index=target.message_index,
                reviewer_id=SYSTEM_OPENAI_REVIEWER_ID,
                categories=categories,
                category_other=category_other,
                created_at=datetime.now(timezone.utc),
            )
        except Exception as error:  # pylint: disable=broad-exception-caught
            # pragma: no cover - network/runtime path
            print(
                "Failed OpenAI moderation rating for "
                f"conversation={target.conversation_id} "
                f"message_index={target.message_index}: {error}"
            )

    def _persist_llama_guard_flag(self, target: MessageBackfillTarget) -> None:
        """Rate with Llama Guard and write into reviewer_flags."""
        try:
            raw_result = self.llama_guard_rater(target.content)
            categories, category_other = normalize_llama_guard_categories(raw_result)
            self.repository.upsert_system_reviewer_flag(
                conversation_id=target.conversation_id,
                message_index=target.message_index,
                reviewer_id=SYSTEM_LLAMA_REVIEWER_ID,
                categories=categories,
                category_other=category_other,
                created_at=datetime.now(timezone.utc),
            )
        except Exception as error:  # pylint: disable=broad-exception-caught
            # pragma: no cover - network/runtime path
            print(
                "Failed Llama Guard rating for "
                f"conversation={target.conversation_id} "
                f"message_index={target.message_index}: {error}"
            )

    @staticmethod
    def extract_candidate_message_indexes(
        change_event: dict[str, Any], messages: list[Any]
    ) -> set[int]:
        """Infer which message indexes were touched by a change event."""
        operation = change_event.get("operationType")

        if operation in {"insert", "replace"}:
            return set(range(len(messages)))

        if operation != "update":
            return set()

        update_description = change_event.get("updateDescription", {})
        updated_fields = update_description.get("updatedFields", {})
        removed_fields = update_description.get("removedFields", [])

        message_indexes: set[int] = set()

        # If full messages array is replaced, evaluate all messages.
        if "messages" in updated_fields:
            return set(range(len(messages)))

        for field_path in list(updated_fields.keys()) + list(removed_fields):
            message_index = RealtimeConversationWatcher._parse_message_index(field_path)
            if message_index is not None:
                message_indexes.add(message_index)

        return message_indexes

    @staticmethod
    def _parse_message_index(field_path: str) -> Optional[int]:
        """Parse message index from paths like `messages.3.content`."""
        if not isinstance(field_path, str):
            return None
        if not field_path.startswith("messages."):
            return None

        parts = field_path.split(".")
        if len(parts) < 2:
            return None

        index_part = parts[1]
        if not index_part.isdigit():
            return None

        return int(index_part)
