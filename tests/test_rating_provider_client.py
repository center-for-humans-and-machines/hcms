"""Tests for the moderation providers' shared client settings."""

# pylint: disable=missing-function-docstring,protected-access,too-few-public-methods

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from hcms.rating import llama_guard
from hcms.rating.provider_client import (
    PROVIDER_MAX_RETRIES_ENV,
    PROVIDER_TIMEOUT_ENV,
    provider_client_options,
)


def test_provider_client_options_defaults(monkeypatch):
    monkeypatch.delenv(PROVIDER_TIMEOUT_ENV, raising=False)
    monkeypatch.delenv(PROVIDER_MAX_RETRIES_ENV, raising=False)

    assert provider_client_options() == {"timeout": 30.0, "max_retries": 2}


def test_provider_client_options_reads_env(monkeypatch):
    monkeypatch.setenv(PROVIDER_TIMEOUT_ENV, "7.5")
    monkeypatch.setenv(PROVIDER_MAX_RETRIES_ENV, "0")

    assert provider_client_options() == {"timeout": 7.5, "max_retries": 0}


@pytest.mark.parametrize(
    ("name", "value"),
    [
        (PROVIDER_TIMEOUT_ENV, "0"),
        (PROVIDER_TIMEOUT_ENV, "-1"),
        (PROVIDER_MAX_RETRIES_ENV, "-1"),
    ],
)
def test_provider_client_options_rejects_invalid_values(monkeypatch, name, value):
    monkeypatch.setenv(name, value)

    with pytest.raises(ValueError, match=name):
        provider_client_options()


class _RecordingOpenAI:
    """Records the keyword arguments each client was created with."""

    client_kwargs: list[dict[str, Any]] = []

    def __init__(self, **kwargs: Any) -> None:
        _RecordingOpenAI.client_kwargs.append(kwargs)
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **_: SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content="safe"))]
                )
            )
        )


def test_llama_guard_client_is_bounded_by_timeout_and_retries(monkeypatch):
    _RecordingOpenAI.client_kwargs = []
    monkeypatch.setattr(llama_guard, "OpenAI", _RecordingOpenAI)
    monkeypatch.setenv("SKIP_CACHE", "true")
    monkeypatch.setenv("LLAMA_GUARD_API_KEY", "test")
    monkeypatch.setenv("LLAMA_GUARD_ENDPOINT", "http://localhost/v1")
    monkeypatch.setenv(PROVIDER_TIMEOUT_ENV, "12")
    monkeypatch.setenv(PROVIDER_MAX_RETRIES_ENV, "1")

    assert llama_guard._rate_text_with_llama_guard("hello") == ""
    assert len(_RecordingOpenAI.client_kwargs) == 1
    assert _RecordingOpenAI.client_kwargs[0]["timeout"] == 12.0
    assert _RecordingOpenAI.client_kwargs[0]["max_retries"] == 1
