"""Tests for reading a conversation's cost from Langfuse (BAN-1460)."""

# pylint: disable=missing-function-docstring,too-few-public-methods

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest

import generate_dataset


def _page(costs: List[Optional[float]], cursor: Optional[str]) -> SimpleNamespace:
    return SimpleNamespace(
        data=[SimpleNamespace(total_cost=cost) for cost in costs],
        meta=SimpleNamespace(cursor=cursor),
    )


class _FakeObservations:
    """Serves observation pages keyed by cursor and records each request."""

    def __init__(self, pages: Dict[Optional[str], SimpleNamespace]) -> None:
        self.pages = pages
        self.requests: List[Dict[str, Any]] = []

    async def get_many(self, **kwargs: Any) -> SimpleNamespace:
        self.requests.append(kwargs)
        return self.pages[kwargs.get("cursor")]


def _use_observations(monkeypatch, observations: Any) -> None:
    client = SimpleNamespace(async_api=SimpleNamespace(observations=observations))
    monkeypatch.setattr(generate_dataset, "get_client", lambda: client)


def test_cost_sums_observations_across_pages(monkeypatch):
    observations = _FakeObservations(
        {
            None: _page([0.25, None], cursor="next"),
            "next": _page([0.5], cursor=None),
        }
    )
    _use_observations(monkeypatch, observations)

    cost = asyncio.run(generate_dataset.get_cost_per_conversation("trace-1"))

    assert cost == pytest.approx(0.75)
    assert [request["cursor"] for request in observations.requests] == [None, "next"]
    assert all(request["trace_id"] == "trace-1" for request in observations.requests)
    assert all("usage" in request["fields"] for request in observations.requests)


def test_cost_is_zero_without_a_trace_id(monkeypatch):
    observations = _FakeObservations({})
    _use_observations(monkeypatch, observations)

    assert asyncio.run(generate_dataset.get_cost_per_conversation(None)) == 0.0
    assert not observations.requests


class _FailingObservations:
    """Observations endpoint that is always unavailable."""

    async def get_many(self, **_: Any) -> SimpleNamespace:
        raise RuntimeError("Langfuse unavailable")


def test_cost_is_zero_when_langfuse_fails(monkeypatch):
    _use_observations(monkeypatch, _FailingObservations())

    assert asyncio.run(generate_dataset.get_cost_per_conversation("trace-1")) == 0.0
