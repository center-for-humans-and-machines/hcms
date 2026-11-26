"""Rate generated conversations with the Kill The Bot (KTB) judge.

Every assistant reply is rated in context: up to four earlier user/assistant
turns go to the judge as ``context_messages``. KTB grades the bot's reply, so
user turns are context only and get no rating of their own.

As in the realtime watcher, a failed rating is never written as a safe verdict.
Transport errors, truncated completions and invalid verdicts are recorded as
failures (``ktb_unsafe`` is null and ``ktb_error`` says why), counted, and the
run fails when their share exceeds ``LLM_JUDGE_MAX_FAILED_SHARE``.

The run stops early, without waiting for every reply, once the failures
already exceed that share of all replies to rate, or after
``MAX_CONSECUTIVE_FAILURES`` failed replies in a row: a judge server that
hangs instead of refusing connections would otherwise hold the run for up to
``LLM_JUDGE_MAX_ATTEMPTS`` timeouts per reply.
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import polars as pl
from tqdm import tqdm

from hcms.constants import ASSISTANT, USER
from hcms.loading.constants import DATASET_DIR
from hcms.loading.data_loading import _load_json_with_config
from hcms.rating.ktb_judge import KtbJudgeVerdict, rate_text_with_ktb_judge
from hcms.utils import get_env_variable

KtbJudgeRater = Callable[..., KtbJudgeVerdict]

CONTEXT_TURNS = 4
"""Earlier turns sent with each reply, as the realtime watcher does."""

DEFAULT_MAX_ATTEMPTS = 2
"""Attempts per reply: one retry absorbs a transient judge or network error."""

DEFAULT_MAX_FAILED_SHARE = 0.05
"""Share of replies (0 to 1) that may fail to get a verdict before the run fails."""

MAX_CONSECUTIVE_FAILURES = 10
"""Failed replies in a row after which the judge is taken to be down."""

RETRY_DELAY_SECONDS = 5.0
"""Pause between attempts at one reply, so a transient error can clear."""

OUTPUT_SCHEMA = {
    "conversation_id": pl.String,
    "turn": pl.Int64,
    "role": pl.String,
    "message": pl.String,
    "ktb_unsafe": pl.Boolean,
    "ktb_categories": pl.List(pl.String),
    "ktb_rationale": pl.String,
    "ktb_error": pl.String,
}


class KtbJudgeFailuresError(RuntimeError):
    """Too many replies could not be rated; their verdicts are unknown."""


@dataclass
class KtbJudgeRunSummary:
    """Counts across all files a run rated."""

    rated: int = 0
    unsafe: int = 0
    failed: int = 0
    consecutive_failed: int = 0
    stopped_early: bool = False

    @property
    def total(self) -> int:
        """Replies the judge was asked to rate."""
        return self.rated + self.failed

    @property
    def failed_share(self) -> float:
        """Share of replies without a verdict."""
        return self.failed / self.total if self.total else 0.0


def _context_for(conversation: list[dict[str, Any]], index: int) -> tuple:
    """Return up to CONTEXT_TURNS nonblank user/assistant turns before ``index``."""
    earlier = [
        (message["role"], message["content"])
        for message in conversation[:index]
        if message.get("role") in {USER, ASSISTANT}
        and (message.get("content") or "").strip()
    ]
    return tuple(earlier[-CONTEXT_TURNS:])


def _rate_reply(
    rater: KtbJudgeRater,
    content: str,
    context: tuple,
    max_attempts: int,
) -> dict[str, Any]:
    """Rate one reply; return the verdict fields, or the last error if all attempts fail."""
    error: Optional[Exception] = None
    for attempt in range(max_attempts):
        if attempt:
            time.sleep(RETRY_DELAY_SECONDS)
        try:
            verdict = rater(content, role=ASSISTANT, context_messages=context)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            error = exc
            continue
        if not isinstance(verdict, KtbJudgeVerdict):
            error = TypeError(f"Expected a judge verdict, got {type(verdict).__name__}")
            continue
        return {
            "ktb_unsafe": verdict.unsafe,
            "ktb_categories": list(verdict.categories),
            "ktb_rationale": verdict.rationale,
            "ktb_error": None,
        }
    return {
        "ktb_unsafe": None,
        "ktb_categories": None,
        "ktb_rationale": None,
        "ktb_error": f"{type(error).__name__}: {error}",
    }


def _replies(df: pl.DataFrame):
    """Yield (conversation id, conversation, index) of each nonblank assistant reply."""
    for conversation_id, conversation in zip(
        df["conversation_id"].to_list(), df["conversation"].to_list()
    ):
        for index, message in enumerate(conversation or []):
            content = message.get("content") or ""
            if message.get("role") == ASSISTANT and content.strip():
                yield conversation_id, conversation, index


def _rate_file(  # pylint: disable=too-many-arguments
    df: pl.DataFrame,
    rater: KtbJudgeRater,
    max_attempts: int,
    summary: KtbJudgeRunSummary,
    *,
    max_failed: float,
) -> pl.DataFrame:
    """Rate every nonblank assistant reply in one generated dataset file.

    Stops after the reply that takes the failures above ``max_failed`` or to
    MAX_CONSECUTIVE_FAILURES in a row, and sets ``summary.stopped_early``.
    """
    rows = []
    for conversation_id, conversation, index in _replies(df):
        content = conversation[index]["content"]
        result = _rate_reply(
            rater, content, _context_for(conversation, index), max_attempts
        )
        if result["ktb_error"] is not None:
            summary.failed += 1
            summary.consecutive_failed += 1
        else:
            summary.rated += 1
            summary.unsafe += bool(result["ktb_unsafe"])
            summary.consecutive_failed = 0
        rows.append(
            {
                "conversation_id": str(conversation_id),
                "turn": index,
                "role": ASSISTANT,
                "message": content,
                **result,
            }
        )
        if (
            summary.failed > max_failed
            or summary.consecutive_failed >= MAX_CONSECUTIVE_FAILURES
        ):
            summary.stopped_early = True
            break
    return pl.DataFrame(rows, schema=OUTPUT_SCHEMA)


def rate_conversations_with_ktb_judge(  # pylint: disable=too-many-arguments
    file_pattern: str = "dataset-round-2-*.json",
    input_dir: Path = DATASET_DIR,
    output_dir: Path = DATASET_DIR,
    output_prefix: str = "rated-ktb-judge-",
    *,
    rater: Optional[KtbJudgeRater] = None,
    max_attempts: Optional[int] = None,
    max_failed_share: Optional[float] = None,
) -> KtbJudgeRunSummary:
    """Rate each assistant reply in the matching files with the KTB judge.

    Writes one ``{output_prefix}{input name}`` file per input, with a row per
    assistant reply. Every file is written before the failure check, so the
    failed rows can be inspected. Once the run can no longer pass (failures
    above ``max_failed_share`` of all replies, or MAX_CONSECUTIVE_FAILURES in
    a row) it stops, writes the rows rated so far and raises.

    Args:
        file_pattern: Glob for the generated dataset files.
        input_dir: Directory holding the dataset files.
        output_dir: Directory for the rated files.
        output_prefix: Prefix of the rated files' names.
        rater: Judge call, ``rate_text_with_ktb_judge`` by default.
        max_attempts: Attempts per reply; ``LLM_JUDGE_MAX_ATTEMPTS`` or 2.
        max_failed_share: Allowed share of failed replies;
            ``LLM_JUDGE_MAX_FAILED_SHARE`` or 0.05.

    Returns:
        KtbJudgeRunSummary: Counts of rated, unsafe and failed replies.

    Raises:
        FileNotFoundError: If no file matches ``file_pattern``.
        KtbJudgeFailuresError: If there is no reply to rate, the failed share
            is above ``max_failed_share``, or the run stopped early.
    """
    rater = rater or rate_text_with_ktb_judge
    # A blank variable (an unset GitHub variable is passed as "") means default.
    if max_attempts is None:
        max_attempts = int(
            get_env_variable("LLM_JUDGE_MAX_ATTEMPTS", default="").strip()
            or DEFAULT_MAX_ATTEMPTS
        )
    if max_failed_share is None:
        max_failed_share = float(
            get_env_variable("LLM_JUDGE_MAX_FAILED_SHARE", default="").strip()
            or DEFAULT_MAX_FAILED_SHARE
        )
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    if not 0 <= max_failed_share <= 1:
        raise ValueError("max_failed_share must be between 0 and 1")

    input_files = sorted(input_dir.glob(file_pattern))
    if not input_files:
        raise FileNotFoundError(
            f"No files found in {input_dir} matching pattern '{file_pattern}'"
        )

    datasets = [(path, _load_json_with_config(path)) for path in input_files]
    expected = sum(sum(1 for _ in _replies(df)) for _, df in datasets)
    if expected == 0:
        raise KtbJudgeFailuresError(
            f"No assistant replies to rate in files matching '{file_pattern}'"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    summary = KtbJudgeRunSummary()
    for path, df in tqdm(datasets, desc="Rating files with the KTB judge"):
        rated = _rate_file(
            df, rater, max_attempts, summary, max_failed=max_failed_share * expected
        )
        output_path = output_dir / f"{output_prefix}{path.name}"
        rated.write_json(output_path)
        print(f"Wrote {rated.height} KTB judge ratings to {output_path}")
        if summary.stopped_early:
            break

    print(
        f"KTB judge: {summary.total} of {expected} replies, {summary.unsafe} "
        f"unsafe, {summary.failed} failed ({summary.failed_share:.1%}, "
        f"allowed {max_failed_share:.1%})"
    )
    if summary.stopped_early:
        raise KtbJudgeFailuresError(
            f"Stopped after {summary.failed} of {summary.total} replies tried "
            f"(of {expected}) got no KTB judge verdict, "
            f"{summary.consecutive_failed} of them in a row; the failed share "
            f"of {max_failed_share:.1%} can no longer be met or the judge is "
            "down. See the ktb_error column of the rated files"
        )
    if summary.failed_share > max_failed_share:
        raise KtbJudgeFailuresError(
            f"{summary.failed} of {summary.total} replies got no KTB judge verdict "
            f"({summary.failed_share:.1%} > {max_failed_share:.1%}); see the "
            "ktb_error column of the rated files"
        )
    return summary
