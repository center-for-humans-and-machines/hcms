"""Shared HTTP client settings for the moderation providers."""

from typing import Any

from hcms.utils import get_env_variable

PROVIDER_TIMEOUT_ENV = "HCMS_PROVIDER_TIMEOUT_SECONDS"
PROVIDER_MAX_RETRIES_ENV = "HCMS_PROVIDER_MAX_RETRIES"
DEFAULT_PROVIDER_TIMEOUT_SECONDS = 30.0
DEFAULT_PROVIDER_MAX_RETRIES = 2


def provider_client_options() -> dict[str, Any]:
    """Timeout and retry settings for the OpenAI-compatible provider clients.

    The SDK defaults (600 s per attempt, 2 retries) let one hung request stall
    the realtime watcher for half an hour. These bound every rating call so a
    slow provider costs one message its rating, not the whole queue its latency.

    Returns:
        dict[str, Any]: `timeout` (seconds per attempt) and `max_retries`
        keyword arguments for `openai.OpenAI`.

    Raises:
        ValueError: If the configured timeout is not positive or the retry
            count is negative.
    """
    timeout_seconds = float(
        get_env_variable(
            PROVIDER_TIMEOUT_ENV, default=str(DEFAULT_PROVIDER_TIMEOUT_SECONDS)
        )
    )
    max_retries = int(
        get_env_variable(
            PROVIDER_MAX_RETRIES_ENV, default=str(DEFAULT_PROVIDER_MAX_RETRIES)
        )
    )
    if timeout_seconds <= 0:
        raise ValueError(
            f"{PROVIDER_TIMEOUT_ENV} must be positive, got {timeout_seconds}"
        )
    if max_retries < 0:
        raise ValueError(f"{PROVIDER_MAX_RETRIES_ENV} must be >= 0, got {max_retries}")
    return {"timeout": timeout_seconds, "max_retries": max_retries}
