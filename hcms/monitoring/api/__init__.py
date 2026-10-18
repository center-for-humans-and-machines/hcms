"""Entrypoint for the API used in monitoring."""

from hcms.monitoring.api.client import create_client
from hcms.monitoring.api.config import BatchConfig

__all__ = ["create_client", "BatchConfig"]
