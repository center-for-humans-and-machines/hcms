"""Realtime MongoDB watcher that rates newly added messages."""

from __future__ import annotations

import logging
import math
import random
import time
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from hcms.database.models import ConversationDocument
from hcms.database.repository import (
    SYSTEM_LLAMA_REVIEWER_ID,
    SYSTEM_OPENAI_REVIEWER_ID,
    MessageBackfillTarget,
    MongoConversationRepository,
)
from pydantic import ValidationError

LOGGER = logging.getLogger(__name__)

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

ProviderRater = Callable[[str], Any]
ProviderResultMap = dict[str, bool]
BackfillProviderKey = tuple[Any, int, str]


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


def _get_default_openai_rater() -> ProviderRater:
    """Resolve default OpenAI rater lazily to avoid import-time side effects."""
    # pylint: disable=import-outside-toplevel
    from hcms.rating.openai_moderation import _rate_text_with_openai_moderation

    return _rate_text_with_openai_moderation


def _get_default_llama_guard_rater() -> ProviderRater:
    """Resolve default Llama Guard rater lazily to avoid import-time side effects."""
    # pylint: disable=import-outside-toplevel
    from hcms.rating.llama_guard import _rate_text_with_llama_guard

    return _rate_text_with_llama_guard


class RealtimeConversationWatcher:
    """Watches MongoDB conversation changes and persists moderation ratings."""

    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-instance-attributes,too-many-branches
    def __init__(
        self,
        repository: MongoConversationRepository,
        openai_rater: Optional[ProviderRater] = None,
        llama_guard_rater: Optional[ProviderRater] = None,
        backfill_batch_size: int = 200,
        change_stream_max_await_ms: int = 1000,
        backfill_max_retries: int = 10,
        backfill_retry_sleep_seconds: float = 2.0,
        reconnect_backoff_base_seconds: float = 1.0,
        reconnect_backoff_max_seconds: float = 30.0,
        reconnect_backoff_jitter_seconds: float = 0.25,
        summary_log_interval_events: int = 100,
    ) -> None:
        self.repository = repository
        self.openai_rater = openai_rater or _get_default_openai_rater()
        self.llama_guard_rater = llama_guard_rater or _get_default_llama_guard_rater()
        self.backfill_batch_size = backfill_batch_size
        self.change_stream_max_await_ms = change_stream_max_await_ms
        self.backfill_max_retries = max(backfill_max_retries, 1)
        self.backfill_retry_sleep_seconds = max(backfill_retry_sleep_seconds, 0.0)
        self.reconnect_backoff_base_seconds = max(reconnect_backoff_base_seconds, 0.1)
        self.reconnect_backoff_max_seconds = max(
            reconnect_backoff_max_seconds, self.reconnect_backoff_base_seconds
        )
        self.reconnect_backoff_jitter_seconds = max(
            reconnect_backoff_jitter_seconds, 0.0
        )
        self.summary_log_interval_events = max(summary_log_interval_events, 1)

        self._events_consumed = 0
        self._invalid_documents = 0
        self._provider_success_counts = {
            SYSTEM_OPENAI_REVIEWER_ID: 0,
            SYSTEM_LLAMA_REVIEWER_ID: 0,
        }
        self._provider_failure_counts = {
            SYSTEM_OPENAI_REVIEWER_ID: 0,
            SYSTEM_LLAMA_REVIEWER_ID: 0,
        }
        self._unresolved_backfill_targets = 0

    def run(self) -> None:
        """Open change stream first, then backfill and process realtime events."""

        consecutive_failures = 0
        while True:
            try:
                with self.repository.watch(
                    max_await_time_ms=self.change_stream_max_await_ms
                ) as stream:
                    if consecutive_failures > 0:
                        LOGGER.info(
                            "Change stream reconnected after %d failure(s)",
                            consecutive_failures,
                        )
                    consecutive_failures = 0

                    # Start the stream first to avoid a blind startup window between
                    # the final backfill query and stream establishment.
                    self.run_startup_backfill()

                    for change_event in stream:
                        self.handle_change_event(change_event)
            except Exception:  # pylint: disable=broad-exception-caught
                # pragma: no cover - network/runtime path
                consecutive_failures += 1
                sleep_seconds = self._compute_reconnect_sleep_seconds(
                    failure_number=consecutive_failures
                )
                LOGGER.exception(
                    "Change stream interrupted; retrying in %.2fs (failure=%d)",
                    sleep_seconds,
                    consecutive_failures,
                )
                time.sleep(sleep_seconds)

    # pylint: disable=too-many-branches
    def run_startup_backfill(self) -> None:
        """Backfill unrated messages present before the watcher started."""
        attempts: dict[BackfillProviderKey, int] = {}
        unresolved_provider_keys: set[BackfillProviderKey] = set()

        while True:
            exhausted_provider_keys = {
                key
                for key, attempt_count in attempts.items()
                if attempt_count >= self.backfill_max_retries
            }
            unresolved_provider_keys.update(exhausted_provider_keys)

            targets = self.repository.get_backfill_targets(
                batch_size=self.backfill_batch_size,
                excluded_provider_keys=exhausted_provider_keys,
            )
            if not targets:
                self._unresolved_backfill_targets = len(unresolved_provider_keys)
                if self._unresolved_backfill_targets:
                    LOGGER.error(
                        "Startup backfill ended with unresolved provider target(s): %d",
                        self._unresolved_backfill_targets,
                    )
                else:
                    LOGGER.info("Startup backfill completed successfully")
                return

            progress_made = False
            retry_pending = False
            for target in targets:
                eligible_reviewer_ids: set[str] = set()
                for reviewer_id in target.missing_reviewer_ids:
                    key = (target.document_id, target.message_index, reviewer_id)
                    if attempts.get(key, 0) >= self.backfill_max_retries:
                        unresolved_provider_keys.add(key)
                        continue
                    eligible_reviewer_ids.add(reviewer_id)

                if not eligible_reviewer_ids:
                    continue

                eligible_target = MessageBackfillTarget(
                    document_id=target.document_id,
                    message_index=target.message_index,
                    content=target.content,
                    missing_reviewer_ids=eligible_reviewer_ids,
                )
                provider_results = self.process_message_target(eligible_target)
                for reviewer_id in eligible_reviewer_ids:
                    key = (target.document_id, target.message_index, reviewer_id)
                    if provider_results.get(reviewer_id, False):
                        attempts.pop(key, None)
                        unresolved_provider_keys.discard(key)
                        progress_made = True
                        continue

                    attempt_count = attempts.get(key, 0) + 1
                    attempts[key] = attempt_count
                    if attempt_count < self.backfill_max_retries:
                        retry_pending = True
                        continue

                    unresolved_provider_keys.add(key)
                    LOGGER.error(
                        "Startup backfill retries exhausted for "
                        "document_id=%s message_index=%s reviewer_id=%s attempts=%d",
                        target.document_id,
                        target.message_index,
                        reviewer_id,
                        attempt_count,
                    )

            if progress_made:
                continue

            if retry_pending:
                LOGGER.warning(
                    "Startup backfill made no progress; retrying unresolved targets in "
                    "%.2fs",
                    self.backfill_retry_sleep_seconds,
                )
                time.sleep(self.backfill_retry_sleep_seconds)
                continue

            LOGGER.warning(
                "Startup backfill made no progress in current batch; continuing scan "
                "with exhausted provider target(s) excluded"
            )
            continue

    # pylint: enable=too-many-branches

    def handle_change_event(self, change_event: dict[str, Any]) -> None:
        """Handle one MongoDB change stream event."""
        self._events_consumed += 1

        full_document = change_event.get("fullDocument")
        if not isinstance(full_document, dict):
            self._maybe_log_summary()
            return

        validated_document = self._validate_full_document(full_document)
        if validated_document is None:
            self._invalid_documents += 1
            self._maybe_log_summary()
            return

        messages = validated_document.get("messages", [])
        if not isinstance(messages, list):
            self._maybe_log_summary()
            return

        message_indexes = self.extract_candidate_message_indexes(change_event, messages)

        for message_index in sorted(message_indexes):
            if message_index < 0 or message_index >= len(messages):
                continue

            message = messages[message_index]
            if not isinstance(message, dict):
                continue
            if not self.repository.is_message_eligible_for_system_reviewers(message):
                continue
            content = message["content"]

            missing_reviewer_ids = self.repository.missing_system_reviewers(message)
            if not missing_reviewer_ids:
                continue

            target = MessageBackfillTarget(
                document_id=validated_document.get("_id"),
                message_index=message_index,
                content=content,
                missing_reviewer_ids=missing_reviewer_ids,
            )
            self.process_message_target(target)

        self._maybe_log_summary()

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
            LOGGER.warning(
                "Skipping invalid conversation document from change stream %s: %s",
                full_document.get("_id"),
                error,
            )
            return None
        return validated.model_dump(by_alias=True)

    def process_message_target(
        self, target: MessageBackfillTarget
    ) -> ProviderResultMap:
        """Rate one message and persist missing system reviewer flags."""
        provider_results: ProviderResultMap = {}

        if SYSTEM_OPENAI_REVIEWER_ID in target.missing_reviewer_ids:
            provider_results[SYSTEM_OPENAI_REVIEWER_ID] = self._persist_openai_flag(
                target
            )

        if SYSTEM_LLAMA_REVIEWER_ID in target.missing_reviewer_ids:
            provider_results[SYSTEM_LLAMA_REVIEWER_ID] = self._persist_llama_guard_flag(
                target
            )

        return provider_results

    def _persist_openai_flag(self, target: MessageBackfillTarget) -> bool:
        """Rate with OpenAI moderation and write into reviewer_flags."""
        return self._persist_provider_flag(
            target=target,
            reviewer_id=SYSTEM_OPENAI_REVIEWER_ID,
            provider_name="OpenAI moderation",
            rater=self.openai_rater,
            normalizer=normalize_openai_categories,
        )

    def _persist_llama_guard_flag(self, target: MessageBackfillTarget) -> bool:
        """Rate with Llama Guard and write into reviewer_flags."""
        return self._persist_provider_flag(
            target=target,
            reviewer_id=SYSTEM_LLAMA_REVIEWER_ID,
            provider_name="Llama Guard",
            rater=self.llama_guard_rater,
            normalizer=normalize_llama_guard_categories,
        )

    def _persist_provider_flag(
        self,
        target: MessageBackfillTarget,
        reviewer_id: str,
        provider_name: str,
        rater: ProviderRater,
        normalizer: Callable[[Any], tuple[list[str], str]],
    ) -> bool:
        """Run one provider rating and persist one system reviewer flag."""
        try:
            raw_result = rater(target.content)
            categories, category_other = normalizer(raw_result)
            self.repository.upsert_system_reviewer_flag(
                document_id=target.document_id,
                message_index=target.message_index,
                reviewer_id=reviewer_id,
                categories=categories,
                category_other=category_other,
                created_at=datetime.now(timezone.utc),
            )
            self._provider_success_counts[reviewer_id] += 1
            return True
        except Exception:  # pylint: disable=broad-exception-caught
            # pragma: no cover - network/runtime path
            self._provider_failure_counts[reviewer_id] += 1
            LOGGER.exception(
                "Failed %s rating for document_id=%s message_index=%s",
                provider_name,
                target.document_id,
                target.message_index,
            )
            return False

    def _compute_reconnect_sleep_seconds(self, failure_number: int) -> float:
        """Compute reconnect delay using exponential backoff with jitter."""
        raw_exponent = max(failure_number - 1, 0)
        max_ratio = (
            self.reconnect_backoff_max_seconds / self.reconnect_backoff_base_seconds
        )
        if max_ratio <= 1:
            max_exponent = 0
        else:
            max_exponent = math.ceil(math.log2(max_ratio))

        exponent = min(raw_exponent, max_exponent)
        base_sleep = self.reconnect_backoff_base_seconds * (2**exponent)
        bounded_sleep = min(base_sleep, self.reconnect_backoff_max_seconds)
        jitter = (
            random.uniform(0.0, self.reconnect_backoff_jitter_seconds)
            if self.reconnect_backoff_jitter_seconds > 0
            else 0.0
        )
        return bounded_sleep + jitter

    def _maybe_log_summary(self) -> None:
        """Log watcher runtime counters at a fixed event interval."""
        if self._events_consumed % self.summary_log_interval_events != 0:
            return

        LOGGER.info(
            "Watcher summary events=%d invalid_documents=%d "
            "openai_success=%d openai_failure=%d "
            "llama_success=%d llama_failure=%d "
            "unresolved_backfill_targets=%d",
            self._events_consumed,
            self._invalid_documents,
            self._provider_success_counts[SYSTEM_OPENAI_REVIEWER_ID],
            self._provider_failure_counts[SYSTEM_OPENAI_REVIEWER_ID],
            self._provider_success_counts[SYSTEM_LLAMA_REVIEWER_ID],
            self._provider_failure_counts[SYSTEM_LLAMA_REVIEWER_ID],
            self._unresolved_backfill_targets,
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
