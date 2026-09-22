"""API client factory module.

This module provides a factory function to create OpenAI or Azure client based on the endpoint.
"""

from typing import Union

from hcms.utils import get_env_variable
from openai import AzureOpenAI, OpenAI


def create_client() -> Union[OpenAI, AzureOpenAI]:
    """Create OpenAI or Azure client based on endpoint.

    Returns:
        Union[OpenAI, AzureOpenAI]: API client
    """
    endpoint = get_env_variable("MODEL_ENDPOINT")

    if "azure" in endpoint.lower():
        return AzureOpenAI(
            api_key=get_env_variable("MODEL_API_KEY"),
            api_version=get_env_variable("MODEL_API_VERSION"),
            azure_endpoint=endpoint,
        )

    return OpenAI(api_key=get_env_variable("MODEL_API_KEY"))
