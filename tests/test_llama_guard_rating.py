"""Tests for the conversation sent to Llama Guard."""

# pylint: disable=missing-function-docstring,too-few-public-methods

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from hcms.rating import llama_guard


class _FakeOpenAI:
    """Records the messages sent and answers like Llama Guard."""

    sent: list[list[dict[str, str]]] = []

    def __init__(self, **_kwargs: Any) -> None:
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, *, model: str, messages: list[dict[str, str]]):
        _ = model
        _FakeOpenAI.sent.append(messages)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="unsafe\nS2"))]
        )


@pytest.fixture(name="fake_openai")
def fixture_fake_openai(monkeypatch):
    _FakeOpenAI.sent = []
    monkeypatch.setattr(llama_guard, "OpenAI", _FakeOpenAI)
    monkeypatch.setenv("SKIP_CACHE", "true")
    monkeypatch.setenv("LLAMA_GUARD_API_KEY", "test")
    monkeypatch.setenv("LLAMA_GUARD_ENDPOINT", "http://localhost/v1")
    return _FakeOpenAI


@pytest.mark.parametrize(
    ("kwargs", "expected_messages"),
    [
        ({}, [{"role": "user", "content": "reply"}]),
        (
            {"role": "assistant", "previous_user_content": " question "},
            [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "reply"},
            ],
        ),
        # Llama Guard's template needs a user turn first, so a reply without
        # one is rated alone as before.
        (
            {"role": "assistant", "previous_user_content": ""},
            [{"role": "user", "content": "reply"}],
        ),
        (
            {"role": "user", "previous_user_content": "ignored"},
            [{"role": "user", "content": "reply"}],
        ),
    ],
)
def test_rate_text_with_llama_guard_sends_reply_with_context(
    fake_openai, kwargs, expected_messages
):
    result = llama_guard._rate_text_with_llama_guard(  # pylint: disable=protected-access
        " reply ", **kwargs
    )

    assert result == "Non-Violent Crimes"
    assert fake_openai.sent == [expected_messages]


def test_reply_with_context_is_cached_separately(monkeypatch):
    saved: list[tuple[str, str]] = []
    monkeypatch.setattr(llama_guard, "OpenAI", _FakeOpenAI)
    monkeypatch.setattr(llama_guard, "get_cached_result", lambda _content, _name: None)
    monkeypatch.setattr(
        llama_guard,
        "save_cached_result",
        lambda content, _result, name: saved.append((name, content)),
    )
    monkeypatch.setenv("LLAMA_GUARD_API_KEY", "test")
    monkeypatch.setenv("LLAMA_GUARD_ENDPOINT", "http://localhost/v1")

    llama_guard._rate_text_with_llama_guard("reply")  # pylint: disable=protected-access
    llama_guard._rate_text_with_llama_guard(  # pylint: disable=protected-access
        "reply", role="assistant", previous_user_content="question"
    )

    assert saved[0] == ("llama_guard", "reply")
    assert saved[1][0] == "llama_guard_response"
    assert "question" in saved[1][1]
