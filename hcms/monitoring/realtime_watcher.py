"""Realtime MongoDB watcher that rates newly added messages."""

# pylint: disable=too-many-lines

from __future__ import annotations

import concurrent.futures
import dataclasses
import functools
import logging
import math
import random
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Optional, Sequence

from pydantic import ValidationError

from hcms.constants import ASSISTANT
from hcms.database.models import ConversationDocument
from hcms.database.repository import (
    DEFAULT_SYSTEM_REVIEWER_IDS,
    SYSTEM_LLAMA_REVIEWER_ID,
    SYSTEM_LLM_JUDGE_REVIEWER_ID,
    SYSTEM_OPENAI_REVIEWER_ID,
    SYSTEM_REVIEWER_IDS,
    MessageBackfillTarget,
    MongoConversationRepository,
)

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

PROVIDER_NAMES = {
    SYSTEM_OPENAI_REVIEWER_ID: "OpenAI moderation",
    SYSTEM_LLAMA_REVIEWER_ID: "Llama Guard",
    SYSTEM_LLM_JUDGE_REVIEWER_ID: "LLM judge",
}

# The dashboard shows the comment next to the flag; keep it readable.
MAX_COMMENT_LENGTH = 500
# Longest pause between two retries of a failed live rating.
MAX_RETRY_DELAY_SECONDS = 60.0

ProviderRater = Callable[[str], Any]
# Called as rater(content), or for assistant replies as
# rater(content, role=..., previous_user_content=...).
LlamaGuardRater = Callable[..., Any]
# Called as rater(content, role=..., context_messages=...).
LlmJudgeRater = Callable[..., Any]
# (categories, category_other) or (categories, category_other, comment).
ReviewerFlagFields = Sequence[Any]
Normalizer = Callable[[Any], ReviewerFlagFields]
ProviderResultMap = dict[str, bool]
BackfillProviderKey = tuple[Any, int, str]
RetryKey = tuple[Any, int]

# Returned by `_next_event` when the change stream can yield nothing more.
_STREAM_CLOSED = object()


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
    if result in {"", "0"} or result.lower() == "safe":
        return [], ""

    if result in KNOWN_LLAMA_GUARD_CATEGORIES:
        return [result], ""

    return [], result


def normalize_ktb_judge_verdict(raw_result: Any) -> tuple[list[str], str, str]:
    """Turn a validated judge verdict into reviewer flag fields.

    The judge's categories alone decide the flag; its rationale becomes the
    comment. Anything but a verdict is an inference failure (the rater raises on
    malformed output), so it raises instead of becoming a safety finding: the
    dashboard treats a non-empty `category_other` as one.
    """
    categories = getattr(raw_result, "categories", None)
    rationale = getattr(raw_result, "rationale", None)
    if not isinstance(categories, list) or not isinstance(rationale, str):
        raise TypeError(
            f"Expected a judge verdict, got {type(raw_result).__name__}: {raw_result!r}"
        )
    return (
        [str(category) for category in categories],
        "",
        rationale.strip()[:MAX_COMMENT_LENGTH],
    )


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


def _get_default_llm_judge_rater() -> LlmJudgeRater:
    """Resolve default LLM judge rater lazily to avoid import-time side effects."""
    # pylint: disable=import-outside-toplevel
    from hcms.rating.ktb_judge import rate_text_with_ktb_judge

    return rate_text_with_ktb_judge


def _unpack_flag_fields(fields: ReviewerFlagFields) -> tuple[list[str], str, str]:
    """Accept two- or three-element normalizer results."""
    categories = [str(category) for category in fields[0]]
    category_other = str(fields[1])
    comment = str(fields[2]) if len(fields) > 2 else ""
    return categories, category_other, comment


@dataclasses.dataclass
class _RetryEntry:
    """A live rating that failed and waits for another attempt."""

    target: MessageBackfillTarget
    due_at: float


@dataclasses.dataclass
class _BackfillPass:
    """Progress of one scan over the collection for unrated messages."""

    attempts: dict[BackfillProviderKey, int] = dataclasses.field(default_factory=dict)
    unresolved_provider_keys: set[BackfillProviderKey] = dataclasses.field(
        default_factory=set
    )
    resume_after: Any = None
    complete: bool = False
    not_before: float = 0.0


class RealtimeConversationWatcher:
    """Watches MongoDB conversation changes and persists moderation ratings.

    Work runs in bounded cycles so nothing starves anything else: a few live
    change events, then a due retry of a failed live rating, then one batch of
    the startup backfill. A live message therefore waits at most for the events
    ahead of it, one retry and one backfill batch. Ratings run on a pool of
    worker threads when several are configured.
    """

    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-instance-attributes,too-many-branches,too-many-locals,too-many-public-methods
    def __init__(
        self,
        repository: MongoConversationRepository,
        openai_rater: Optional[ProviderRater] = None,
        llama_guard_rater: Optional[LlamaGuardRater] = None,
        backfill_batch_size: int = 200,
        change_stream_max_await_ms: int = 1000,
        backfill_max_retries: int = 10,
        backfill_retry_sleep_seconds: float = 2.0,
        reconnect_backoff_base_seconds: float = 1.0,
        reconnect_backoff_max_seconds: float = 30.0,
        reconnect_backoff_jitter_seconds: float = 0.25,
        summary_log_interval_events: int = 100,
        reviewer_ids: Optional[Iterable[str]] = None,
        worker_threads: int = 1,
        rating_lag_warn_seconds: float = 120.0,
        llm_judge_rater: Optional[LlmJudgeRater] = None,
        backfill_since: Optional[datetime] = None,
        backfill_until: Optional[datetime] = None,
        live_max_events_per_cycle: int = 3,
        live_max_seconds_per_cycle: float = 60.0,
        retry_max_per_cycle: int = 1,
    ) -> None:
        self.repository = repository
        # Providers this watcher rates and writes (default: the ones run
        # without configuration). Entries of other providers are left to the
        # watchers that run them.
        self.reviewer_ids = (
            DEFAULT_SYSTEM_REVIEWER_IDS
            if reviewer_ids is None
            else frozenset(reviewer_ids)
        )
        unknown_reviewer_ids = self.reviewer_ids - SYSTEM_REVIEWER_IDS
        if not self.reviewer_ids or unknown_reviewer_ids:
            raise ValueError(
                f"reviewer_ids must be a non-empty subset of {sorted(SYSTEM_REVIEWER_IDS)},"
                f" got {sorted(self.reviewer_ids)}"
            )
        # Resolve default raters only for enabled providers, so a watcher
        # doesn't need a disabled provider's dependencies or credentials.
        self.openai_rater = openai_rater
        if self.openai_rater is None and SYSTEM_OPENAI_REVIEWER_ID in self.reviewer_ids:
            self.openai_rater = _get_default_openai_rater()
        self.llama_guard_rater = llama_guard_rater
        if (
            self.llama_guard_rater is None
            and SYSTEM_LLAMA_REVIEWER_ID in self.reviewer_ids
        ):
            self.llama_guard_rater = _get_default_llama_guard_rater()
        self.llm_judge_rater = llm_judge_rater
        if (
            self.llm_judge_rater is None
            and SYSTEM_LLM_JUDGE_REVIEWER_ID in self.reviewer_ids
        ):
            self.llm_judge_rater = _get_default_llm_judge_rater()
        self.backfill_batch_size = max(backfill_batch_size, 1)
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
        # Messages are rated on this many threads. Each provider write is one
        # atomic update of its own reviewer_flags entry, so ratings of different
        # messages can land in any order without clobbering each other.
        self.worker_threads = max(worker_threads, 1)
        # One rating landing later than this after its message was written logs
        # a warning; 0 disables the warning (the summary still reports lag).
        self.rating_lag_warn_seconds = max(rating_lag_warn_seconds, 0.0)
        # Only messages with a timestamp in [since, until) are backfilled; live
        # events are rated whatever their timestamp.
        self.backfill_since = backfill_since
        self.backfill_until = backfill_until
        self.live_max_events_per_cycle = max(live_max_events_per_cycle, 1)
        self.live_max_seconds_per_cycle = max(live_max_seconds_per_cycle, 0.0)
        self.retry_max_per_cycle = max(retry_max_per_cycle, 0)

        self._lock = threading.Lock()
        self._executor: Optional[ThreadPoolExecutor] = None
        # Provider keys queued or being rated by worker threads, so a change
        # event that arrives again while a rating is in flight is not rated twice.
        self._in_flight: set[BackfillProviderKey] = set()
        # In-flight keys whose change event arrived again meanwhile. If their
        # rating fails, that event is honoured with one more attempt, as serial
        # processing would have retried on it.
        self._retry_requested: set[BackfillProviderKey] = set()
        self._pending_futures: set[Future[None]] = set()
        # Bounds the queue in front of the workers; the change stream cursor
        # keeps the rest of the backlog server-side.
        self._queue_slots = threading.Semaphore(self.worker_threads * 2)
        self._suppress_lag_warnings = False
        self._lag_count = 0
        self._lag_total_seconds = 0.0
        self._lag_max_seconds = 0.0
        # Failed live ratings waiting for another attempt (with backoff), and
        # how many attempts each message has had so far.
        self._retry_queue: dict[RetryKey, _RetryEntry] = {}
        self._retry_attempts: dict[RetryKey, int] = {}
        self._backfill = _BackfillPass()

        self._events_consumed = 0
        self._invalid_documents = 0
        self._provider_success_counts = dict.fromkeys(SYSTEM_REVIEWER_IDS, 0)
        self._provider_failure_counts = dict.fromkeys(SYSTEM_REVIEWER_IDS, 0)
        self._unresolved_backfill_targets = 0
        self._unresolved_live_targets = 0

    def run(self) -> None:
        """Open the change stream, then cycle through live events, retries and backfill."""

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
                    # the final backfill query and stream establishment. Ratings
                    # dispatched before a reconnect must land before the backfill
                    # selects targets, or they would be rated twice. The backfill
                    # pass then runs in batches between live events.
                    self.wait_for_pending()
                    self._backfill = _BackfillPass()
                    while self._run_cycle(stream):
                        pass
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

    def _run_cycle(self, stream: Any) -> bool:
        """Run one bounded round of work; return False once the stream is closed."""
        deadline = time.monotonic() + self.live_max_seconds_per_cycle
        for _ in range(self.live_max_events_per_cycle):
            change_event = self._next_event(stream)
            if change_event is _STREAM_CLOSED:
                return False
            if change_event is None:
                break
            self.handle_change_event(change_event)
            if time.monotonic() >= deadline:
                break

        self._run_due_retries()
        self._run_backfill_batch()
        return True

    @staticmethod
    def _next_event(stream: Any) -> Any:
        """Return the next change event, None when none is pending, or closed."""
        try_next = getattr(stream, "try_next", None)
        if callable(try_next):
            if getattr(stream, "alive", True) is False:
                return _STREAM_CLOSED
            return try_next()
        try:
            return next(stream)
        except StopIteration:
            return _STREAM_CLOSED

    def run_startup_backfill(self) -> None:
        """Backfill unrated messages in one go, retrying transient failures.

        Backfilled ratings are late by definition, so they count towards the
        lag summary but do not log per-message lag warnings.

        On a change stream reconnect, ratings dispatched before the stream
        failed may still be queued or running. Their writes must land before
        targets are selected, or the same message would be rated again and the
        later result would overwrite the first.
        """
        self.wait_for_pending()
        self._suppress_lag_warnings = True
        try:
            self._backfill = _BackfillPass()
            while not self._backfill.complete:
                status = self._backfill_step()
                if status == "retry_pending":
                    LOGGER.warning(
                        "Startup backfill made no progress; retrying unresolved "
                        "targets in %.2fs",
                        self.backfill_retry_sleep_seconds,
                    )
                    time.sleep(self.backfill_retry_sleep_seconds)
                elif status == "no_progress":
                    LOGGER.warning(
                        "Startup backfill made no progress in current batch; "
                        "continuing scan with exhausted provider target(s) excluded"
                    )
        finally:
            self._suppress_lag_warnings = False

    def _run_backfill_batch(self) -> None:
        """Advance the backfill pass by one batch, unless it is done or paused."""
        if self._backfill.complete or time.monotonic() < self._backfill.not_before:
            return
        self._suppress_lag_warnings = True
        try:
            status = self._backfill_step()
        finally:
            self._suppress_lag_warnings = False
        if status in {"retry_pending", "no_progress"}:
            # Let live events through before asking the same targets again.
            self._backfill.not_before = (
                time.monotonic() + self.backfill_retry_sleep_seconds
            )

    # pylint: disable=too-many-branches
    def _backfill_step(self) -> str:
        """Fetch and rate one backfill batch.

        Returns "complete" when nothing is left, "progress" when a flag was
        written, "retry_pending" when a failed target still has attempts left,
        and "no_progress" otherwise.
        """
        state = self._backfill
        exhausted_provider_keys = {
            key
            for key, attempt_count in state.attempts.items()
            if attempt_count >= self.backfill_max_retries
        }
        state.unresolved_provider_keys.update(exhausted_provider_keys)

        batch = self.repository.next_backfill_batch(
            batch_size=self.backfill_batch_size,
            excluded_provider_keys=exhausted_provider_keys,
            reviewer_ids=set(self.reviewer_ids),
            since=self.backfill_since,
            until=self.backfill_until,
            after_id=state.resume_after,
        )
        state.resume_after = batch.resume_after
        if not batch.targets:
            state.complete = True
            self._unresolved_backfill_targets = len(state.unresolved_provider_keys)
            if self._unresolved_backfill_targets:
                LOGGER.error(
                    "Startup backfill ended with unresolved provider target(s): %d",
                    self._unresolved_backfill_targets,
                )
            else:
                LOGGER.info(
                    "Startup backfill completed successfully (max_lag=%.1fs)",
                    self._lag_max_seconds,
                )
            return "complete"

        eligible_targets: list[MessageBackfillTarget] = []
        for target in batch.targets:
            eligible_reviewer_ids: set[str] = set()
            for reviewer_id in target.missing_reviewer_ids:
                key = (target.document_id, target.message_index, reviewer_id)
                if state.attempts.get(key, 0) >= self.backfill_max_retries:
                    state.unresolved_provider_keys.add(key)
                    continue
                eligible_reviewer_ids.add(reviewer_id)

            if not eligible_reviewer_ids:
                continue

            eligible_targets.append(
                dataclasses.replace(target, missing_reviewer_ids=eligible_reviewer_ids)
            )

        progress_made = False
        retry_pending = False
        batch_results = self._process_targets(eligible_targets)
        for target, provider_results in zip(eligible_targets, batch_results):
            for reviewer_id in target.missing_reviewer_ids:
                key = (target.document_id, target.message_index, reviewer_id)
                if provider_results.get(reviewer_id, False):
                    state.attempts.pop(key, None)
                    state.unresolved_provider_keys.discard(key)
                    progress_made = True
                    continue

                attempt_count = state.attempts.get(key, 0) + 1
                state.attempts[key] = attempt_count
                if attempt_count < self.backfill_max_retries:
                    retry_pending = True
                    continue

                state.unresolved_provider_keys.add(key)
                LOGGER.error(
                    "Startup backfill retries exhausted for "
                    "document_id=%s message_index=%s reviewer_id=%s attempts=%d",
                    target.document_id,
                    target.message_index,
                    reviewer_id,
                    attempt_count,
                )

        if progress_made:
            return "progress"
        if retry_pending:
            return "retry_pending"
        return "no_progress"

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
        content_changed_indexes = self.extract_content_changed_message_indexes(
            change_event
        )

        for message_index in sorted(message_indexes):
            if message_index < 0 or message_index >= len(messages):
                continue

            message = messages[message_index]
            if not isinstance(message, dict):
                continue
            if not self.repository.is_message_eligible_for_system_reviewers(message):
                continue
            content = message["content"]

            if message_index in content_changed_indexes:
                # The text changed after it was rated: existing entries are
                # stale, so every provider of this watcher rates it again.
                missing_reviewer_ids = set(self.reviewer_ids)
            else:
                missing_reviewer_ids = self.repository.missing_system_reviewers(
                    message, set(self.reviewer_ids)
                )
            if not missing_reviewer_ids:
                continue

            target = MessageBackfillTarget(
                document_id=validated_document.get("_id"),
                message_index=message_index,
                content=content,
                missing_reviewer_ids=missing_reviewer_ids,
                role=message.get("role", ""),
                previous_user_content=MongoConversationRepository.previous_user_content(
                    messages, message_index
                ),
                message_timestamp=message.get("timestamp"),
                context_messages=MongoConversationRepository.context_messages(
                    messages, message_index
                ),
            )
            self._dispatch_target(target)

        self._maybe_log_summary()

    def _dispatch_target(self, target: MessageBackfillTarget) -> None:
        """Rate a message inline, or on a worker thread when several are configured.

        With worker threads, providers already queued or in flight for the same
        message are skipped, so a change event that arrives again before the
        first rating lands (the watcher's own write, another provider's write)
        does not rate it twice. Should that rating then fail, the skipped event
        earns it one more attempt. Blocks when the queue in front of the
        workers is full, so the backlog stays in the change stream.
        """
        if self.worker_threads == 1:
            provider_results = self.process_message_target(target)
            self._settle_live_result(target, provider_results)
            return

        with self._lock:
            reviewer_ids: set[str] = set()
            for reviewer_id in target.missing_reviewer_ids:
                key = (target.document_id, target.message_index, reviewer_id)
                if key in self._in_flight:
                    self._retry_requested.add(key)
                else:
                    reviewer_ids.add(reviewer_id)
            if not reviewer_ids:
                return
            keys = {
                (target.document_id, target.message_index, reviewer_id)
                for reviewer_id in reviewer_ids
            }
            self._in_flight.update(keys)

        # Released by the worker thread, so a `with` block does not fit here.
        self._queue_slots.acquire()  # pylint: disable=consider-using-with
        self._submit_queued_target(target, reviewer_ids, keys, holds_queue_slot=True)

    def _submit_queued_target(
        self,
        target: MessageBackfillTarget,
        reviewer_ids: set[str],
        keys: set[BackfillProviderKey],
        holds_queue_slot: bool,
    ) -> None:
        queued_target = dataclasses.replace(target, missing_reviewer_ids=reviewer_ids)
        try:
            future = self._get_executor().submit(
                self._process_queued_target, queued_target, keys, holds_queue_slot
            )
        except RuntimeError:
            # The executor was shut down underneath us; give the keys back so
            # a restart's backfill picks the message up.
            with self._lock:
                self._in_flight.difference_update(keys)
            if holds_queue_slot:
                self._queue_slots.release()
            LOGGER.error(
                "Watcher is shutting down; dropped rating of document_id=%s "
                "message_index=%s",
                target.document_id,
                target.message_index,
            )
            return
        with self._lock:
            self._pending_futures.add(future)
        future.add_done_callback(self._discard_pending_future)

    def _process_queued_target(
        self,
        target: MessageBackfillTarget,
        keys: set[BackfillProviderKey],
        holds_queue_slot: bool,
    ) -> None:
        """Worker-thread body: rate one message, then free its keys and queue slot.

        A provider that failed while a change event for it was skipped is
        queued once more. The retry inherits this task's queue slot, so the
        queue bound holds and no worker ever waits for a slot (which could
        deadlock with the queued work it is needed for). Each retry is bounded
        by the skipped event that requested it. Other failures go to the retry
        queue, which the main loop serves with backoff.
        """
        provider_results: ProviderResultMap = {}
        try:
            provider_results = self.process_message_target(target)
        finally:
            with self._lock:
                self._in_flight.difference_update(keys)
                retry_keys = {
                    key
                    for key in keys & self._retry_requested
                    if not provider_results.get(key[2], False)
                }
                self._retry_requested.difference_update(keys)
                self._in_flight.update(retry_keys)
            if retry_keys:
                LOGGER.info(
                    "Retrying %d failed rating(s) for document_id=%s message_index=%s "
                    "requested by a change event received meanwhile",
                    len(retry_keys),
                    target.document_id,
                    target.message_index,
                )
                self._submit_queued_target(
                    target,
                    {key[2] for key in retry_keys},
                    retry_keys,
                    holds_queue_slot=holds_queue_slot,
                )
            elif holds_queue_slot:
                self._queue_slots.release()
            self._settle_live_result(
                target,
                provider_results,
                excluded_reviewer_ids={key[2] for key in retry_keys},
            )

    def _settle_live_result(
        self,
        target: MessageBackfillTarget,
        provider_results: ProviderResultMap,
        excluded_reviewer_ids: Optional[set[str]] = None,
    ) -> None:
        """Queue the providers that failed for a retry, or clear the message's record."""
        failed_reviewer_ids = {
            reviewer_id
            for reviewer_id in target.missing_reviewer_ids & self.reviewer_ids
            if not provider_results.get(reviewer_id, False)
        } - (excluded_reviewer_ids or set())
        retry_key: RetryKey = (target.document_id, target.message_index)
        if failed_reviewer_ids:
            self._schedule_retry(target, failed_reviewer_ids)
            return
        with self._lock:
            if retry_key not in self._retry_queue:
                self._retry_attempts.pop(retry_key, None)

    def _schedule_retry(
        self, target: MessageBackfillTarget, failed_reviewer_ids: set[str]
    ) -> None:
        """Queue a failed live rating for another attempt, with backoff."""
        retry_key: RetryKey = (target.document_id, target.message_index)
        with self._lock:
            attempts = self._retry_attempts.get(retry_key, 0) + 1
            existing = self._retry_queue.get(retry_key)
            if existing is not None:
                failed_reviewer_ids = failed_reviewer_ids | set(
                    existing.target.missing_reviewer_ids
                )

            if attempts >= self.backfill_max_retries:
                self._retry_queue.pop(retry_key, None)
                self._retry_attempts.pop(retry_key, None)
                self._unresolved_live_targets += 1
                exhausted = True
            else:
                self._retry_attempts[retry_key] = attempts
                delay = min(
                    self.backfill_retry_sleep_seconds * (2 ** max(attempts - 1, 0)),
                    MAX_RETRY_DELAY_SECONDS,
                )
                self._retry_queue[retry_key] = _RetryEntry(
                    target=dataclasses.replace(
                        target, missing_reviewer_ids=failed_reviewer_ids
                    ),
                    due_at=time.monotonic() + delay,
                )
                exhausted = False

        if exhausted:
            LOGGER.error(
                "Retries exhausted for document_id=%s message_index=%s "
                "reviewer_ids=%s attempts=%d; left for the next startup backfill",
                target.document_id,
                target.message_index,
                sorted(failed_reviewer_ids),
                attempts,
            )

    def _run_due_retries(self) -> None:
        """Retry at most `retry_max_per_cycle` failed live ratings that are due."""
        if self.retry_max_per_cycle <= 0:
            return
        now = time.monotonic()
        with self._lock:
            due_keys = [
                key for key, entry in self._retry_queue.items() if entry.due_at <= now
            ]
            due_entries = [
                self._retry_queue.pop(key)
                for key in due_keys[: self.retry_max_per_cycle]
            ]
        for entry in due_entries:
            self._dispatch_target(entry.target)

    def _discard_pending_future(self, future: Future[None]) -> None:
        with self._lock:
            self._pending_futures.discard(future)

    def _process_targets(
        self, targets: list[MessageBackfillTarget]
    ) -> list[ProviderResultMap]:
        """Rate targets in order, on the worker pool when there is one."""
        if self.worker_threads == 1 or len(targets) <= 1:
            return [self.process_message_target(target) for target in targets]
        return list(self._get_executor().map(self.process_message_target, targets))

    def _get_executor(self) -> ThreadPoolExecutor:
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=self.worker_threads, thread_name_prefix="hcms-rater"
            )
        return self._executor

    def wait_for_pending(self, timeout: Optional[float] = None) -> bool:
        """Block until queued ratings, and the retries they queue, have finished.

        Returns:
            bool: True once nothing is pending, False if `timeout` ran out first.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            with self._lock:
                pending = [f for f in self._pending_futures if not f.done()]
            if not pending:
                return True
            remaining = (
                None if deadline is None else max(deadline - time.monotonic(), 0.0)
            )
            if remaining == 0.0:
                return False
            concurrent.futures.wait(pending, timeout=remaining)

    def shutdown(self) -> None:
        """Finish queued ratings, retries included, then stop the worker threads."""
        if self._executor is None:
            return
        self.wait_for_pending()
        self._executor.shutdown(wait=True)
        self._executor = None

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
        missing_reviewer_ids = target.missing_reviewer_ids & self.reviewer_ids

        if SYSTEM_OPENAI_REVIEWER_ID in missing_reviewer_ids:
            provider_results[SYSTEM_OPENAI_REVIEWER_ID] = self._persist_openai_flag(
                target
            )

        if SYSTEM_LLAMA_REVIEWER_ID in missing_reviewer_ids:
            provider_results[SYSTEM_LLAMA_REVIEWER_ID] = self._persist_llama_guard_flag(
                target
            )

        if SYSTEM_LLM_JUDGE_REVIEWER_ID in missing_reviewer_ids:
            provider_results[SYSTEM_LLM_JUDGE_REVIEWER_ID] = (
                self._persist_llm_judge_flag(target)
            )

        return provider_results

    def _persist_openai_flag(self, target: MessageBackfillTarget) -> bool:
        """Rate with OpenAI moderation and write into reviewer_flags."""
        assert self.openai_rater is not None  # set whenever the provider is enabled
        return self._persist_provider_flag(
            target=target,
            reviewer_id=SYSTEM_OPENAI_REVIEWER_ID,
            provider_name=PROVIDER_NAMES[SYSTEM_OPENAI_REVIEWER_ID],
            rater=self.openai_rater,
            normalizer=normalize_openai_categories,
        )

    def _persist_llama_guard_flag(self, target: MessageBackfillTarget) -> bool:
        """Rate with Llama Guard and write into reviewer_flags.

        Assistant replies are rated together with the user message they answer,
        so Llama Guard applies its response policy instead of judging the reply
        as if a user had written it.
        """
        assert (
            self.llama_guard_rater is not None
        )  # set whenever the provider is enabled
        rater: ProviderRater = self.llama_guard_rater
        if target.role == ASSISTANT:
            rater = functools.partial(
                self.llama_guard_rater,
                role=ASSISTANT,
                previous_user_content=target.previous_user_content,
            )
        return self._persist_provider_flag(
            target=target,
            reviewer_id=SYSTEM_LLAMA_REVIEWER_ID,
            provider_name=PROVIDER_NAMES[SYSTEM_LLAMA_REVIEWER_ID],
            rater=rater,
            normalizer=normalize_llama_guard_categories,
        )

    def _persist_llm_judge_flag(self, target: MessageBackfillTarget) -> bool:
        """Rate with the LLM judge, in context, and write into reviewer_flags."""
        assert self.llm_judge_rater is not None  # set whenever the provider is enabled
        rater: ProviderRater = functools.partial(
            self.llm_judge_rater,
            role=target.role,
            context_messages=target.context_messages,
        )
        return self._persist_provider_flag(
            target=target,
            reviewer_id=SYSTEM_LLM_JUDGE_REVIEWER_ID,
            provider_name=PROVIDER_NAMES[SYSTEM_LLM_JUDGE_REVIEWER_ID],
            rater=rater,
            normalizer=normalize_ktb_judge_verdict,
        )

    def _persist_provider_flag(
        self,
        target: MessageBackfillTarget,
        reviewer_id: str,
        provider_name: str,
        rater: ProviderRater,
        normalizer: Normalizer,
    ) -> bool:
        """Run one provider rating and persist one system reviewer flag.

        Any exception, including a rater rejecting malformed model output, is a
        failure: nothing is written, the failure is counted and logged, and the
        caller retries. A message whose content changed since it was read is
        not written either; its content-change event rates it again.
        """
        try:
            raw_result = rater(target.content)
            categories, category_other, comment = _unpack_flag_fields(
                normalizer(raw_result)
            )
            rated_at = datetime.now(timezone.utc)
            persisted = self.repository.upsert_system_reviewer_flag(
                document_id=target.document_id,
                message_index=target.message_index,
                reviewer_id=reviewer_id,
                categories=categories,
                category_other=category_other,
                created_at=rated_at,
                comment=comment,
                expected_content=target.content,
            )
            if persisted is False:
                LOGGER.info(
                    "Skipped %s rating for document_id=%s message_index=%s: "
                    "message changed since it was read",
                    provider_name,
                    target.document_id,
                    target.message_index,
                )
            with self._lock:
                self._provider_success_counts[reviewer_id] += 1
            # Lag counts the write too: the rating is late until it is stored.
            self._record_rating_lag(target, provider_name, datetime.now(timezone.utc))
            return True
        except Exception:  # pylint: disable=broad-exception-caught
            # pragma: no cover - network/runtime path
            with self._lock:
                self._provider_failure_counts[reviewer_id] += 1
            LOGGER.exception(
                "Failed %s rating for document_id=%s message_index=%s",
                provider_name,
                target.document_id,
                target.message_index,
            )
            return False

    def _record_rating_lag(
        self, target: MessageBackfillTarget, provider_name: str, stored_at: datetime
    ) -> None:
        """Track how long after a message was written its rating was stored.

        The lag is the end-to-end delay a dashboard user sees: it includes time
        spent waiting in the change stream and writing to MongoDB, not only the
        provider call.
        """
        if target.message_timestamp is None:
            return
        message_time = target.message_timestamp
        if message_time.tzinfo is None:
            message_time = message_time.replace(tzinfo=timezone.utc)
        lag_seconds = max((stored_at - message_time).total_seconds(), 0.0)

        with self._lock:
            self._lag_count += 1
            self._lag_total_seconds += lag_seconds
            self._lag_max_seconds = max(self._lag_max_seconds, lag_seconds)

        if (
            self.rating_lag_warn_seconds
            and lag_seconds > self.rating_lag_warn_seconds
            and not self._suppress_lag_warnings
        ):
            LOGGER.warning(
                "Rating lag %.1fs exceeds %.0fs: %s for document_id=%s message_index=%s",
                lag_seconds,
                self.rating_lag_warn_seconds,
                provider_name,
                target.document_id,
                target.message_index,
            )

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

        with self._lock:
            lag_count = self._lag_count
            mean_lag = self._lag_total_seconds / lag_count if lag_count else 0.0
            max_lag = self._lag_max_seconds
            # Lag is reported per summary window so a growing backlog shows up.
            self._lag_count = 0
            self._lag_total_seconds = 0.0
            self._lag_max_seconds = 0.0
            provider_counts = " ".join(
                f"{reviewer_id}_success={self._provider_success_counts[reviewer_id]} "
                f"{reviewer_id}_failure={self._provider_failure_counts[reviewer_id]}"
                for reviewer_id in sorted(self.reviewer_ids)
            )
            LOGGER.info(
                "Watcher summary events=%d invalid_documents=%d %s "
                "unresolved_backfill_targets=%d retry_queue=%d "
                "unresolved_live_targets=%d "
                "rated=%d mean_lag=%.1fs max_lag=%.1fs in_flight=%d",
                self._events_consumed,
                self._invalid_documents,
                provider_counts,
                self._unresolved_backfill_targets,
                len(self._retry_queue),
                self._unresolved_live_targets,
                lag_count,
                mean_lag,
                max_lag,
                len(self._in_flight),
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
    def extract_content_changed_message_indexes(
        change_event: dict[str, Any],
    ) -> set[int]:
        """Return the indexes whose `content` field an update event rewrote.

        Only a targeted `messages.<i>.content` write counts; a replaced
        `messages` array gives no way to tell which texts changed.
        """
        if change_event.get("operationType") != "update":
            return set()

        update_description = change_event.get("updateDescription", {})
        updated_fields = update_description.get("updatedFields", {})

        message_indexes: set[int] = set()
        for field_path in updated_fields:
            if not isinstance(field_path, str):
                continue
            parts = field_path.split(".")
            if (
                len(parts) == 3
                and parts[0] == "messages"
                and parts[1].isdigit()
                and parts[2] == "content"
            ):
                message_indexes.add(int(parts[1]))
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
