"""Entrypoint for the monitoring package."""

from typing import TYPE_CHECKING, Any

from hcms.monitoring.api.client import create_client
from hcms.monitoring.api.config import BatchConfig
from hcms.monitoring.processors import BatchProcessor, RegTestProcessor
from hcms.monitoring.processors.base import BaseProcessor
from hcms.monitoring.processors.rate_conversation import (
    RateConversationsProcessor,
    RateMessagesProcessor,
)

if TYPE_CHECKING:
    from hcms.monitoring.realtime_watcher import RealtimeConversationWatcher

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
        from hcms.monitoring.realtime_watcher import RealtimeConversationWatcher

        return RealtimeConversationWatcher
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
