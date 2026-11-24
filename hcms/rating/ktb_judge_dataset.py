"""Rate generated conversations with the Kill The Bot (KTB) judge.

Every assistant reply is rated in context: up to four earlier user/assistant
turns go to the judge as ``context_messages``. KTB grades the bot's reply, so
user turns are context only and get no rating of their own.

As in the realtime watcher, a failed rating is never written as a safe verdict.
Transport errors, truncated completions and invalid verdicts are recorded as
failures (``ktb_unsafe`` is null and ``ktb_error`` says why), counted, and the
run fails when their share exceeds ``LLM_JUDGE_MAX_FAILED_SHARE``.
"""

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
    for _ in range(max_attempts):
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


def _rate_file(
    path: Path,
    rater: KtbJudgeRater,
    max_attempts: int,
    summary: KtbJudgeRunSummary,
) -> pl.DataFrame:
    """Rate every nonblank assistant reply in one generated dataset file."""
    df = _load_json_with_config(path)
    rows = []
    for conversation_id, conversation in zip(
        df["conversation_id"].to_list(), df["conversation"].to_list()
    ):
        for index, message in enumerate(conversation or []):
            content = message.get("content") or ""
            if message.get("role") != ASSISTANT or not content.strip():
                continue
            result = _rate_reply(
                rater, content, _context_for(conversation, index), max_attempts
            )
            if result["ktb_error"] is not None:
                summary.failed += 1
            else:
                summary.rated += 1
                summary.unsafe += bool(result["ktb_unsafe"])
            rows.append(
                {
                    "conversation_id": str(conversation_id),
                    "turn": index,
                    "role": ASSISTANT,
                    "message": content,
                    **result,
                }
            )
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
    failed rows can be inspected.

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
        KtbJudgeFailuresError: If no reply was rated, or the failed share is
            above ``max_failed_share``.
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

    output_dir.mkdir(parents=True, exist_ok=True)
    summary = KtbJudgeRunSummary()
    for path in tqdm(input_files, desc="Rating files with the KTB judge"):
        rated = _rate_file(path, rater, max_attempts, summary)
        output_path = output_dir / f"{output_prefix}{path.name}"
        rated.write_json(output_path)
        print(f"Wrote {rated.height} KTB judge ratings to {output_path}")

    print(
        f"KTB judge: {summary.total} replies, {summary.unsafe} unsafe, "
        f"{summary.failed} failed ({summary.failed_share:.1%}, "
        f"allowed {max_failed_share:.1%})"
    )
    if summary.total == 0:
        raise KtbJudgeFailuresError(
            f"No assistant replies to rate in files matching '{file_pattern}'"
        )
    if summary.failed_share > max_failed_share:
        raise KtbJudgeFailuresError(
            f"{summary.failed} of {summary.total} replies got no KTB judge verdict "
            f"({summary.failed_share:.1%} > {max_failed_share:.1%}); see the "
            "ktb_error column of the rated files"
        )
    return summary
