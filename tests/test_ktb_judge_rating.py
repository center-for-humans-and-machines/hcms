"""Contract tests for judge output, independent of model quality."""

# pylint: disable=missing-function-docstring,too-few-public-methods

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from hcms.rating import ktb_judge
from hcms.rating.ktb_judge import KtbJudgeVerdict


class _FakeLangfuse:
    """Records prompt requests and serves a fixed Langfuse prompt."""

    prompt = "Langfuse prompt"
    requests: list = []

    def get_prompt(self, name, **kwargs):
        _FakeLangfuse.requests.append((name, kwargs))
        return SimpleNamespace(prompt=self.prompt)


@pytest.fixture(name="fake_langfuse", autouse=True)
def fixture_fake_langfuse(monkeypatch):
    """Keep tests off the live Langfuse that the local .env configures."""
    _FakeLangfuse.requests = []
    monkeypatch.setattr(ktb_judge, "get_client", _FakeLangfuse)
    monkeypatch.setattr(ktb_judge, "langfuse_configured", lambda: True)
    return _FakeLangfuse


@pytest.mark.parametrize(
    "categories,unsafe",
    [([], False), (["Suicidal ideation"], True), (["Self-harm"], True)],
)
def test_categories_determine_flag(categories, unsafe):
    """A supported category cannot be suppressed by a separate severity score."""
    verdict = KtbJudgeVerdict(categories=categories, rationale="Explanation")
    assert verdict.unsafe is unsafe


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"categories": ["unknown"], "rationale": "Explanation"},
        {"categories": [], "rationale": " "},
        {"categories": [], "rationale": "Explanation", "severity": 0},
        {"categories": ["Self-harm", "Self-harm"], "rationale": "Explanation"},
    ],
)
def test_invalid_output_is_not_a_verdict(payload):
    """Incomplete or inconsistent classifications remain failures."""
    with pytest.raises(ValidationError):
        KtbJudgeVerdict.model_validate(payload)


@pytest.mark.parametrize("finish_reason,content", [("length", "{}"), ("stop", None)])
def test_incomplete_completion_raises(monkeypatch, finish_reason, content):
    """Truncation and empty output must not pass the minimum acceptance gate."""
    client = MagicMock()
    client.__enter__.return_value = client
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason=finish_reason,
                message=SimpleNamespace(content=content),
            )
        ]
    )
    monkeypatch.setattr(ktb_judge, "OpenAI", lambda **_kwargs: client)
    monkeypatch.setattr(
        ktb_judge,
        "get_env_variable",
        lambda name, default="": (
            default if name == "LLM_JUDGE_TIMEOUT_SECONDS" else "test"
        ),
    )
    with pytest.raises(ValueError):
        ktb_judge.rate_text_with_ktb_judge("I want to die")


class _RecordingOpenAI:
    """Captures the client options and the completion request."""

    client_kwargs: dict = {}
    create_kwargs: dict = {}
    response_content = (
        '{"categories": ["Suicidal ideation"], "rationale": "wish to die"}'
    )

    def __init__(self, **kwargs):
        _RecordingOpenAI.client_kwargs = kwargs
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def __enter__(self):
        return self

    def __exit__(self, *_exc_info):
        return False

    def _create(self, **kwargs):
        _RecordingOpenAI.create_kwargs = kwargs
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(content=self.response_content),
                )
            ]
        )


@pytest.fixture(name="recording_openai")
def fixture_recording_openai(monkeypatch):
    _RecordingOpenAI.client_kwargs = {}
    _RecordingOpenAI.create_kwargs = {}
    monkeypatch.setattr(ktb_judge, "OpenAI", _RecordingOpenAI)
    monkeypatch.setenv("LLM_JUDGE_API_KEY", "test-key")
    monkeypatch.setenv("LLM_JUDGE_ENDPOINT", "http://127.0.0.1:8081/v1")
    monkeypatch.setenv("LLM_JUDGE_MODEL", "judge-alias")
    monkeypatch.setenv("LLM_JUDGE_TIMEOUT_SECONDS", "45")
    return _RecordingOpenAI


def test_transcript_keeps_the_last_four_conversation_turns(recording_openai):
    """System turns and blank turns are dropped; order is kept; target is last."""
    context = (
        ("system", "instructions"),
        ("user", "turn 1"),
        ("assistant", "turn 2"),
        ("user", "   "),
        ("user", "turn 3"),
        ("assistant", "turn 4"),
        ("user", "turn 5"),
    )

    verdict = ktb_judge.rate_text_with_ktb_judge(
        "I want to die", role="user", context_messages=context
    )

    assert verdict.categories == ["Suicidal ideation"]
    messages = recording_openai.create_kwargs["messages"]
    assert messages[0] == {"role": "system", "content": ktb_judge.judge_prompt()}
    transcript = json.loads(messages[1]["content"])
    assert transcript == {
        "earlier_turns": [
            {"role": "assistant", "content": "turn 2"},
            {"role": "user", "content": "turn 3"},
            {"role": "assistant", "content": "turn 4"},
            {"role": "user", "content": "turn 5"},
        ],
        "target": {"role": "user", "content": "I want to die"},
    }


def test_request_is_deterministic_schema_constrained_and_bounded(recording_openai):
    ktb_judge.rate_text_with_ktb_judge("Ich will sterben")

    assert recording_openai.client_kwargs == {
        "api_key": "test-key",
        "base_url": "http://127.0.0.1:8081/v1",
        "timeout": 45.0,
        "max_retries": 0,
    }
    request = recording_openai.create_kwargs
    assert request["model"] == "judge-alias"
    assert request["temperature"] == 0
    assert request["max_tokens"] == 256
    assert request["response_format"]["type"] == "json_schema"
    assert request["response_format"]["json_schema"]["strict"] is True
    assert (
        request["response_format"]["json_schema"]["schema"]
        == KtbJudgeVerdict.model_json_schema()
    )


@pytest.mark.parametrize(
    ("content", "role"),
    [("", "user"), ("   ", "user"), ("text", "system"), ("text", "")],
)
def test_blank_content_or_unsupported_role_is_rejected(recording_openai, content, role):
    _ = recording_openai
    with pytest.raises(ValueError):
        ktb_judge.rate_text_with_ktb_judge(content, role=role)


def test_every_category_is_described_in_the_prompt():
    prompt = ktb_judge.bundled_judge_prompt()
    for category in ktb_judge.JudgeCategory.__args__:
        assert category in prompt


def test_prompt_comes_from_langfuse_with_the_bundled_copy_as_fallback(
    fake_langfuse,
):
    assert ktb_judge.judge_prompt() == "Langfuse prompt"
    name, options = fake_langfuse.requests[-1]
    assert name == "ktb-judge"
    assert options["label"] == ktb_judge.get_env_variable(
        "LLM_JUDGE_PROMPT_LABEL", default="production"
    )
    assert options["fallback"] == ktb_judge.bundled_judge_prompt()
    assert options["max_retries"] == 0


def test_prompt_is_bundled_without_langfuse_credentials(monkeypatch, fake_langfuse):
    monkeypatch.setattr(ktb_judge, "langfuse_configured", lambda: False)
    assert ktb_judge.judge_prompt() == ktb_judge.bundled_judge_prompt()
    assert not fake_langfuse.requests
