"""Tests for rating generated datasets with the KTB judge.

All tests use synthetic conversations and a fake judge: no live calls, since
hcms loads the main checkout's .env and a live run would reach real services.
"""

# pylint: disable=missing-function-docstring,too-few-public-methods

from __future__ import annotations

import asyncio
import importlib
import json
from pathlib import Path

import openai
import polars as pl
import pytest

from hcms.rating.ktb_judge import KtbJudgeVerdict
from hcms.rating.ktb_judge_dataset import (
    KtbJudgeFailuresError,
    rate_conversations_with_ktb_judge,
)

SAFE = KtbJudgeVerdict(categories=[], rationale="Supportive reply.")
UNSAFE = KtbJudgeVerdict(
    categories=["Suicide methods or planning"], rationale="Gives a method."
)


def _conversation(*turns: tuple[str, str]) -> list[dict[str, str]]:
    return [{"role": role, "content": content} for role, content in turns]


def _write_dataset(directory: Path, name: str, conversations: dict) -> Path:
    path = directory / name
    pl.DataFrame(
        {
            "conversation_id": list(conversations),
            "conversation": list(conversations.values()),
        }
    ).write_json(path)
    return path


class _FakeJudge:
    """Records calls and answers from a script keyed by reply text."""

    def __init__(self, answers: dict[str, list] | None = None, default=SAFE):
        self.answers = {key: list(value) for key, value in (answers or {}).items()}
        self.default = default
        self.calls: list[dict] = []

    def __call__(self, content, role="user", context_messages=()):
        self.calls.append(
            {"content": content, "role": role, "context": context_messages}
        )
        queue = self.answers.get(content)
        answer = queue.pop(0) if queue else self.default
        if isinstance(answer, Exception):
            raise answer
        return answer


def _rows(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(name="dataset_dir")
def fixture_dataset_dir(tmp_path: Path) -> Path:
    _write_dataset(
        tmp_path,
        "dataset-round-2-gpt-4o.json",
        {
            "c1": _conversation(
                ("system", "You are a companion."),
                ("user", "u1"),
                ("assistant", "a1"),
                ("user", "u2"),
                ("assistant", "a2"),
                ("user", "u3"),
                ("assistant", "a3"),
                ("user", "u4"),
                ("assistant", "   "),
                ("assistant", "a5"),
            ),
            "c2": _conversation(("assistant", "opening"), ("user", "bye")),
        },
    )
    return tmp_path


def test_rates_each_assistant_reply_with_up_to_four_earlier_turns(dataset_dir):
    judge = _FakeJudge()

    summary = rate_conversations_with_ktb_judge(
        input_dir=dataset_dir, output_dir=dataset_dir, rater=judge
    )

    assert [call["content"] for call in judge.calls] == [
        "a1",
        "a2",
        "a3",
        "a5",
        "opening",
    ]
    assert {call["role"] for call in judge.calls} == {"assistant"}
    # System and blank turns are not context; at most the last four turns go.
    assert judge.calls[0]["context"] == (("user", "u1"),)
    assert judge.calls[3]["context"] == (
        ("assistant", "a2"),
        ("user", "u3"),
        ("assistant", "a3"),
        ("user", "u4"),
    )
    assert judge.calls[4]["context"] == ()
    assert (summary.rated, summary.unsafe, summary.failed) == (5, 0, 0)


def test_writes_verdicts_to_rated_ktb_judge_file(dataset_dir):
    judge = _FakeJudge({"a2": [UNSAFE]})

    rate_conversations_with_ktb_judge(
        input_dir=dataset_dir, output_dir=dataset_dir, rater=judge
    )

    rows = _rows(dataset_dir / "rated-ktb-judge-dataset-round-2-gpt-4o.json")
    by_reply = {row["message"]: row for row in rows}
    assert by_reply["a2"] == {
        "conversation_id": "c1",
        "turn": 4,
        "role": "assistant",
        "message": "a2",
        "ktb_unsafe": True,
        "ktb_categories": ["Suicide methods or planning"],
        "ktb_rationale": "Gives a method.",
        "ktb_error": None,
    }
    assert by_reply["a1"]["ktb_unsafe"] is False
    assert by_reply["a1"]["ktb_categories"] == []


@pytest.mark.parametrize(
    "error",
    [
        openai.APIConnectionError(request=None),  # type: ignore[arg-type]
        ValueError("Judge did not return a complete verdict"),
        ValueError("1 validation error for KtbJudgeVerdict"),
    ],
    ids=["transport", "truncated", "invalid"],
)
def test_failed_rating_is_never_written_as_safe(dataset_dir, error):
    judge = _FakeJudge({"a1": [error, error]})

    with pytest.raises(KtbJudgeFailuresError, match="1 of 5 replies"):
        rate_conversations_with_ktb_judge(
            input_dir=dataset_dir,
            output_dir=dataset_dir,
            rater=judge,
            max_attempts=2,
            max_failed_share=0.1,
        )

    rows = _rows(dataset_dir / "rated-ktb-judge-dataset-round-2-gpt-4o.json")
    failed = next(row for row in rows if row["message"] == "a1")
    assert failed["ktb_unsafe"] is None
    assert failed["ktb_categories"] is None
    assert failed["ktb_error"].startswith(type(error).__name__)


def test_non_verdict_result_counts_as_failure(dataset_dir):
    judge = _FakeJudge({"a1": [None]})

    summary = rate_conversations_with_ktb_judge(
        input_dir=dataset_dir,
        output_dir=dataset_dir,
        rater=judge,
        max_attempts=1,
        max_failed_share=0.5,
    )

    assert summary.failed == 1
    rows = _rows(dataset_dir / "rated-ktb-judge-dataset-round-2-gpt-4o.json")
    assert next(r for r in rows if r["message"] == "a1")["ktb_unsafe"] is None


def test_retry_recovers_a_transient_failure(dataset_dir):
    judge = _FakeJudge({"a1": [ValueError("Judge returned an empty verdict"), UNSAFE]})

    summary = rate_conversations_with_ktb_judge(
        input_dir=dataset_dir,
        output_dir=dataset_dir,
        rater=judge,
        max_attempts=2,
        max_failed_share=0,
    )

    assert (summary.rated, summary.unsafe, summary.failed) == (5, 1, 0)


def test_failures_within_threshold_pass(dataset_dir):
    judge = _FakeJudge({"a1": [ValueError("boom")]})

    summary = rate_conversations_with_ktb_judge(
        input_dir=dataset_dir,
        output_dir=dataset_dir,
        rater=judge,
        max_attempts=1,
        max_failed_share=0.2,
    )

    assert summary.failed == 1
    assert summary.failed_share == pytest.approx(0.2)


def test_threshold_and_attempts_come_from_environment(dataset_dir, monkeypatch):
    monkeypatch.setenv("LLM_JUDGE_MAX_ATTEMPTS", "1")
    monkeypatch.setenv("LLM_JUDGE_MAX_FAILED_SHARE", "0")
    judge = _FakeJudge({"a1": [ValueError("boom"), SAFE]})

    with pytest.raises(KtbJudgeFailuresError):
        rate_conversations_with_ktb_judge(
            input_dir=dataset_dir, output_dir=dataset_dir, rater=judge
        )

    assert [c["content"] for c in judge.calls].count("a1") == 1


def test_blank_environment_values_use_defaults(dataset_dir, monkeypatch):
    monkeypatch.setenv("LLM_JUDGE_MAX_ATTEMPTS", "")
    monkeypatch.setenv("LLM_JUDGE_MAX_FAILED_SHARE", " ")
    # Default: two attempts, so one transient failure still yields a verdict.
    judge = _FakeJudge({"a1": [ValueError("boom"), SAFE]})

    summary = rate_conversations_with_ktb_judge(
        input_dir=dataset_dir, output_dir=dataset_dir, rater=judge
    )

    assert summary.failed == 0


def test_no_matching_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        rate_conversations_with_ktb_judge(
            input_dir=tmp_path, output_dir=tmp_path, rater=_FakeJudge()
        )


def test_no_assistant_reply_fails(tmp_path):
    _write_dataset(
        tmp_path,
        "dataset-round-2-empty.json",
        {"c1": _conversation(("user", "hello"))},
    )

    with pytest.raises(KtbJudgeFailuresError, match="No assistant replies"):
        rate_conversations_with_ktb_judge(
            input_dir=tmp_path, output_dir=tmp_path, rater=_FakeJudge()
        )


@pytest.fixture(name="raters")
def fixture_raters(monkeypatch):
    """Replace every rater evaluate_dataset calls, recording which ran.

    evaluate_dataset builds the moderation client at import, so it is imported
    here with a dummy key; building the client makes no request.
    """
    monkeypatch.setenv("OPENAI_MODERATION_API_KEY", "test")
    evaluate_dataset = importlib.import_module("evaluate_dataset")
    called: list[str] = []
    for name in (
        "rate_conversations_with_llama_guard",
        "rate_conversations_with_openai",
        "rate_conversations_with_ktb_judge",
        "rate_conversations_with_llm_as_judge",
    ):
        monkeypatch.setattr(
            evaluate_dataset,
            name,
            lambda *_args, _name=name, **_kwargs: called.append(_name),
        )
    return evaluate_dataset, called


@pytest.mark.parametrize(
    ("ktb", "legacy", "expected"),
    [
        ("true", "false", ["rate_conversations_with_ktb_judge"]),
        ("false", "false", []),
        ("false", "true", ["rate_conversations_with_llm_as_judge"]),
        (
            "TRUE",
            "True",
            [
                "rate_conversations_with_ktb_judge",
                "rate_conversations_with_llm_as_judge",
            ],
        ),
    ],
)
def test_evaluate_dataset_judge_flags(monkeypatch, raters, ktb, legacy, expected):
    monkeypatch.setenv("LLM_AS_JUDGE_ENABLED", ktb)
    monkeypatch.setenv("LEGACY_LLM_JUDGE_ENABLED", legacy)

    evaluate_dataset, called = raters
    asyncio.run(evaluate_dataset.evaluate_dataset(2))

    assert called[:2] == [
        "rate_conversations_with_llama_guard",
        "rate_conversations_with_openai",
    ]
    assert called[2:] == expected
