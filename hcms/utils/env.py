"""Shared environment-related utilities.

Load and validate environment variables.
"""

import os
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


def get_env_variable(var_name: str, default: Optional[str] = None) -> str:
    """Get the environment variable or return default value if not defined.

    Args:
        var_name: The name of the environment variable.
        default: The default value to return if the environment variable is not set.
                If None, raises an EnvironmentError when the variable is not found.

    Returns:
        str: The value of the environment variable or the default value.

    Raises:
        EnvironmentError: If the environment variable is not set and no default is provided.
    """
    # Ensure environment variables are loaded
    load_dotenv(override=True)

    try:
        return os.environ[var_name]
    except KeyError as e:
        if default is not None:
            return default
        raise EnvironmentError(f"Set the environment variable {var_name}") from e
