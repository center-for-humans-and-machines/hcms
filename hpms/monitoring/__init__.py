"""Entrypoint for the monitoring package."""

from typing import TYPE_CHECKING, Any

from hpms.monitoring.api.client import create_client
from hpms.monitoring.api.config import BatchConfig
from hpms.monitoring.processors import (
    BatchProcessor,
    RegTestProcessor,
)
from hpms.monitoring.processors.base import BaseProcessor
from hpms.monitoring.processors.rate_conversation import (
    RateConversationsProcessor,
    RateMessagesProcessor,
)

if TYPE_CHECKING:
    from hpms.monitoring.realtime_watcher import RealtimeConversationWatcher

__all__ = [
    "create_client",
    "BatchConfig",
    "BatchProcessor",
    "BaseProcessor",
    "RegTestProcessor",
    "RateConversationsProcessor",
    "RateMessagesProcessor",
    "RealtimeConversationWatcher",
]


def __getattr__(name: str) -> Any:
    """Load watcher lazily to avoid import-time moderation credential requirements."""
    if name == "RealtimeConversationWatcher":
        # pylint: disable=import-outside-toplevel
        from hpms.monitoring.realtime_watcher import RealtimeConversationWatcher

        return RealtimeConversationWatcher
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
