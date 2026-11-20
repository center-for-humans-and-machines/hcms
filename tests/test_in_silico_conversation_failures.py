"""Tests for dataset generation surviving requests the model rejects (BAN-484)."""

# pylint: disable=missing-function-docstring,protected-access,too-few-public-methods

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from typing import Any, Callable, List

import httpx
import openai
import polars as pl
import pytest
from langfuse._client.resource_manager import LangfuseResourceManager

import generate_dataset
from hcms.constants import CONTENT, ROLE, USER
from hcms.monitoring.processors.in_silico_conversation import (
    MAX_FAILED_CONVERSATION_SHARE_ENV,
    InSilicoConversationProcessor,
    TooManyFailedConversationsError,
    check_failed_conversation_share,
    parse_max_failed_conversation_share,
)

BAD_REQUEST_MESSAGE = "could not parse JSON body"


@pytest.fixture(autouse=True)
def _no_langfuse_tracing(monkeypatch):
    """Keep the ``@observe`` spans of the processor off the network."""
    for instance in LangfuseResourceManager._instances.values():
        monkeypatch.setattr(instance, "tracing_enabled", False)


def _bad_request() -> openai.BadRequestError:
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    return openai.BadRequestError(
        BAD_REQUEST_MESSAGE, response=httpx.Response(400, request=request), body=None
    )


class _FakeCompletions:
    """Chat completions that either always reject the request or always answer."""

    def __init__(self, reject: bool) -> None:
        self.reject = reject
        self.calls = 0

    def create(self, **_kwargs: Any) -> SimpleNamespace:
        self.calls += 1
        if self.reject:
            raise _bad_request()
        return SimpleNamespace(
            usage=None,
            choices=[SimpleNamespace(message=SimpleNamespace(content="reply"))],
        )


def _processor(monkeypatch, reject: bool, max_turns: int = 2):
    completions = _FakeCompletions(reject)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    monkeypatch.setattr(
        InSilicoConversationProcessor,
        "_init_clients",
        lambda self, config: SimpleNamespace(companion=client, clinician=client),
    )
    processor = InSilicoConversationProcessor(max_turns=max_turns)
    return processor, completions


def test_rejected_request_marks_conversation_failed_without_raising(
    monkeypatch, caplog
):
    processor, completions = _processor(monkeypatch, reject=True)

    with caplog.at_level(logging.ERROR):
        asyncio.run(processor.simulate_conversation("Hello"))

    assert completions.calls == 1, "a 400 is deterministic and must not be retried"
    assert processor.session.failed is True
    assert BAD_REQUEST_MESSAGE in str(processor.session.failure_reason)
    assert processor.session.updated_at is not None
    assert processor.get_conversation_history() == [{ROLE: USER, CONTENT: "Hello"}]
    assert processor.session.session_id in caplog.text
    assert processor.model_config.model in caplog.text


def test_answered_conversation_is_not_failed(monkeypatch):
    processor, completions = _processor(monkeypatch, reject=False, max_turns=2)

    asyncio.run(processor.simulate_conversation("Hello"))

    assert processor.session.failed is False
    assert processor.session.failure_reason is None
    # Two assistant turns and one clinician turn in between.
    assert completions.calls == 3
    assert len(processor.get_conversation_history()) == 4


def _run_generate_dataset(
    monkeypatch, tmp_path, rejected: List[bool], max_failed_share: float
) -> Callable[[], pl.DataFrame]:
    """Drive ``generate_dataset`` with fake clients, one starter per entry."""
    processors: List[InSilicoConversationProcessor] = []

    def make_processor(**_kwargs: Any) -> InSilicoConversationProcessor:
        processor, _ = _processor(monkeypatch, reject=rejected[len(processors)])
        processors.append(processor)
        return processor

    async def no_cost(_trace_id):
        return 0.0

    starters = [f"starter {i}" for i in range(len(rejected))]
    monkeypatch.setattr(
        generate_dataset, "InSilicoConversationProcessor", make_processor
    )
    monkeypatch.setattr(
        generate_dataset, "get_conversation_starters", lambda _r: starters
    )
    monkeypatch.setattr(
        generate_dataset, "get_env_variable", lambda _name: "fake-model"
    )
    monkeypatch.setattr(generate_dataset, "get_cost_per_conversation", no_cost)
    monkeypatch.setattr(generate_dataset, "DATASET_DIR", tmp_path)

    asyncio.run(
        generate_dataset.generate_dataset(
            2, max_turns=1, max_failed_share=max_failed_share
        )
    )
    return lambda: pl.read_json(tmp_path / "dataset-round-2-fake-model.json")


def test_generate_dataset_continues_past_rejected_conversation(monkeypatch, tmp_path):
    read_dataset = _run_generate_dataset(
        monkeypatch, tmp_path, rejected=[True, False, False], max_failed_share=0.5
    )

    dataset = read_dataset()
    assert dataset["conversation_id"].to_list() == ["c1", "c2"]
    assert dataset["model"].to_list() == ["fake-model", "fake-model"]


def test_generate_dataset_fails_once_rejections_exceed_threshold(monkeypatch, tmp_path):
    with pytest.raises(TooManyFailedConversationsError, match="1 of 3"):
        _run_generate_dataset(
            monkeypatch, tmp_path, rejected=[True, False, False], max_failed_share=0.1
        )

    assert not list(tmp_path.iterdir()), "the run stops before writing a dataset"


def test_generate_dataset_fails_when_every_conversation_is_rejected(
    monkeypatch, tmp_path
):
    with pytest.raises(TooManyFailedConversationsError, match="All 2"):
        _run_generate_dataset(
            monkeypatch, tmp_path, rejected=[True, True], max_failed_share=1.0
        )


@pytest.mark.parametrize(
    ("failed", "total", "max_share"),
    [(0, 10, 0.0), (1, 10, 0.1), (0, 0, 0.0), (10, 10, 1.0)],
)
def test_failed_share_within_threshold_passes(failed, total, max_share):
    check_failed_conversation_share(failed, total, max_share)


@pytest.mark.parametrize(
    ("failed", "total", "max_share"), [(1, 10, 0.0), (2, 10, 0.1), (1, 3, 0.1)]
)
def test_failed_share_above_threshold_raises(failed, total, max_share):
    with pytest.raises(TooManyFailedConversationsError, match=f"{failed} of {total}"):
        check_failed_conversation_share(failed, total, max_share)


@pytest.mark.parametrize(("value", "share"), [("0.1", 0.1), ("0", 0.0), ("1", 1.0)])
def test_parse_max_failed_share(value, share):
    assert parse_max_failed_conversation_share(value) == share


@pytest.mark.parametrize("value", ["abc", "-0.1", "1.5", ""])
def test_parse_max_failed_share_rejects_invalid_values(value):
    with pytest.raises(ValueError, match=MAX_FAILED_CONVERSATION_SHARE_ENV):
        parse_max_failed_conversation_share(value)
