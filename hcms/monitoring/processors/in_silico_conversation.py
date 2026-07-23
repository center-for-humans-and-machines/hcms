"""Class to simulate conversation between clinician and companion using agents."""

import os
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import backoff
import openai
import pytz
from dotenv import load_dotenv
from hcms.constants import ASSISTANT, CONTENT, ROLE, SYSTEM, USER
from hcms.loading import COMPANION_SYSTEM_PROMPT_PATH, PSYCHIATRIST_SYSTEM_PROMPT_PATH
from hcms.loading.models import Role
from hcms.monitoring.api import BatchConfig
from hcms.utils import UTF_8, clean_text, get_env_variable
from langfuse import get_client, observe
from openai import AzureOpenAI, OpenAI
from pydantic import BaseModel, ConfigDict, Field
from tqdm import tqdm

load_dotenv()

os.environ["AZURE_OPENAI_API_KEY"] = get_env_variable("MODEL_API_KEY")
os.environ["AZURE_OPENAI_ENDPOINT"] = get_env_variable("MODEL_ENDPOINT")

# Initialize Langfuse client
langfuse = get_client()


# pylint: disable=too-few-public-methods
class ModelConfig(BaseModel):
    """Configuration for model settings."""

    model: str
    api_version: Optional[str] = None
    temperature: float = Field(ge=0.0, le=2.0)
    provider: str = ""


# pylint: disable-next=too-few-public-methods
class SessionInfo(BaseModel):
    """Session tracking information."""

    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    trace_id: Optional[str] = None


# pylint: disable-next=too-few-public-methods
class SystemPrompts(BaseModel):
    """System prompts for both agents."""

    companion: str
    clinician: str


# pylint: disable-next=too-few-public-methods
class ClientConfig(BaseModel):
    """OpenAI clients for both agents."""

    companion: AzureOpenAI | OpenAI
    clinician: AzureOpenAI | OpenAI

    model_config = ConfigDict(arbitrary_types_allowed=True)


# pylint: disable-next=too-few-public-methods
class InSilicoConversationProcessor:
    """Simulates streaming multi-turn conversations between two agents."""

    def __init__(
        self,
        config: BatchConfig = BatchConfig(
            system_prompt=clean_text(
                Path(COMPANION_SYSTEM_PROMPT_PATH).read_text(encoding=UTF_8)
            ),
            # Dummy value to bypass validation
            input_file="data",
        ),
        tags: Optional[List[str]] = None,
        max_turns: int = 10,
    ):
        """Initialize the conversation processor."""
        self.max_turns = max_turns
        self.print_conversation: bool = False

        # Group model-related attributes
        self.model_config = ModelConfig(
            model=config.chat_completions_model, temperature=config.temperature
        )

        # Group session-related attributes
        self.session = SessionInfo(tags=tags or [])

        # Group client-related attributes
        self.clients = self._init_clients(config)

        # Load and store system prompts
        self.prompts = SystemPrompts(
            companion=clean_text(
                Path(COMPANION_SYSTEM_PROMPT_PATH).read_text(encoding=UTF_8)
            ),
            clinician=clean_text(
                Path(PSYCHIATRIST_SYSTEM_PROMPT_PATH).read_text(encoding=UTF_8)
            ),
        )

        self.conversation_history: List[Dict] = []

    def _init_clients(self, config: BatchConfig) -> ClientConfig:
        """Initialize OpenAI clients based on configuration.

        Args:
            config (BatchConfig): Configuration object containing model endpoint and API keys.

        Returns:
            ClientConfig: A configuration object containing initialized OpenAI clients for both
            clinician and companion roles.
        """
        if config.is_endpoint_azure:
            self.model_config.api_version = get_env_variable("MODEL_API_VERSION")
            # AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT are set in the environment
            self.model_config.provider = "Azure OpenAI"

            client_kwargs = {
                "api_key": get_env_variable("MODEL_API_KEY"),
                "api_version": get_env_variable("MODEL_API_VERSION"),
                "azure_endpoint": get_env_variable("MODEL_ENDPOINT"),
            }
            return ClientConfig(
                companion=AzureOpenAI(**client_kwargs),
                clinician=AzureOpenAI(**client_kwargs),
            )

        if config.is_endpoint_together:
            self.model_config.provider = "Together AI"
            client_kwargs = {
                "api_key": get_env_variable("MODEL_API_KEY"),
                "base_url": get_env_variable("MODEL_ENDPOINT"),
            }
            return ClientConfig(
                companion=OpenAI(**client_kwargs),
                clinician=OpenAI(**client_kwargs),
            )

        if config.is_endpoint_openai:
            self.model_config.provider = "OpenAI"
            client_kwargs = {
                "api_key": get_env_variable("MODEL_API_KEY"),
            }

            # If using Google OpenAI, set the base URL
            if config.is_endpoint_google_openai:
                self.model_config.provider = "Google"
                client_kwargs["base_url"] = get_env_variable("MODEL_ENDPOINT")

            return ClientConfig(
                companion=OpenAI(**client_kwargs),
                clinician=OpenAI(**client_kwargs),
            )

        raise ValueError(
            "Invalid model endpoint. Please set the environment variable "
            "'MODEL_ENDPOINT' to either 'azure', 'together', or 'openai'."
        )

    def _get_timestamp(self) -> str:
        """Get the current timestamp in ISO format."""
        return datetime.now(pytz.utc).isoformat()

    def get_conversation_history(self) -> List[Dict]:
        """Get the conversation history as a string.
        Returns:
            List[Dict]: A list of dictionaries containing the conversation history.
        """
        if not self.conversation_history:
            return []
        return self.conversation_history[1:]

    def get_conversation_duration_in_seconds(self) -> int:
        """Get the conversation duration in seconds.

        Returns:
            int: The duration of the conversation in seconds.
        """
        if not self.session.created_at or not self.session.updated_at:
            return 0
        created_at: datetime = datetime.fromisoformat(self.session.created_at)
        updated_at: datetime = datetime.fromisoformat(self.session.updated_at)
        duration: timedelta = updated_at - created_at
        return int(duration.total_seconds())

    def _get_conversation_role(self, message_role: str, perspective_role: str) -> str:
        """
        Determine the correct role for a message based on the conversation perspective.

        This method handles role-swapping to ensure the LLM responds from the correct perspective.
        When we want the LLM to respond as a USER, we need to swap the roles in the conversation
        history so that USER messages appear as ASSISTANT messages (and vice versa) from the
        LLM's perspective.

        Args:
            message_role (str): The original role of the message
                (USER or ASSISTANT)
            perspective_role (str): The role perspective we want the LLM to adopt
                (USER or ASSISTANT)

        Returns:
            str: The role that should be used in the conversation sent to the LLM

        Examples:
            - If perspective_role is USER and message_role is USER → returns ASSISTANT
            - If perspective_role is USER and message_role is ASSISTANT → returns USER
            - If perspective_role is ASSISTANT and message_role is USER → returns USER
            - If perspective_role is ASSISTANT and message_role is ASSISTANT → returns ASSISTANT
        """
        if perspective_role == USER:
            # Swap roles when responding from USER perspective
            return ASSISTANT if message_role == USER else USER
        # Keep original roles when responding from ASSISTANT perspective
        return message_role

    @backoff.on_exception(
        backoff.expo,
        (openai.RateLimitError, openai.InternalServerError),
        max_time=60,
        max_tries=6,
        on_backoff=lambda details: print(
            f"Backing off {details['wait']:.1f}s after {details['tries']} tries for chat completion"
        ),
    )
    @observe(as_type="generation")
    def get_response(self, client, messages, system_prompt, role) -> Optional[str]:
        """
        Generate a response from the language model while maintaining role consistency.

        Args:
            client: The OpenAI or AzureOpenAI client instance used to generate the response.
            messages (List[Dict]): The conversation history, where each message is a dictionary
                containing at least the keys ROLE and CONTENT. The first message is expected to be
                the original system prompt.
            system_prompt (str): The system prompt to prepend to the conversation for context.
            role (str): The role (USER or ASSISTANT) from whose perspective the response should be
                generated.

        Returns:
            str: The generated response content from the LLM.

        Raises:
            ValueError: If the response from the language model is empty.
        """
        # Create a new conversation array with the system prompt
        conversation = [{ROLE: SYSTEM, CONTENT: system_prompt}]

        # Add all messages except the original system message
        for msg in messages[1:]:
            # Determine the correct role for the conversation perspective
            msg_role = self._get_conversation_role(msg[ROLE], role)
            conversation.append({ROLE: msg_role, CONTENT: msg[CONTENT]})

        # Update Langfuse generation with input details
        langfuse.update_current_generation(
            input=conversation,
            model=self.model_config.model,
            metadata={
                "temperature": self.model_config.temperature,
                "provider": self.model_config.provider,
                "role_perspective": role,
                "session_id": self.session.session_id,
            },
        )

        response = client.chat.completions.create(
            model=self.model_config.model,
            messages=conversation,
            temperature=self.model_config.temperature,
        )

        # Update Langfuse generation with usage details
        if response.usage:
            langfuse.update_current_generation(
                usage_details={
                    "input": response.usage.prompt_tokens,
                    "output": response.usage.completion_tokens,
                    "total": response.usage.total_tokens,
                },
                output=response.choices[0].message.content,
            )

        message: Optional[str] = response.choices[0].message.content

        if not message:
            raise ValueError(
                "Received empty response from the language model.", response
            )

        return message

    def _print_conversation_message(self, role: str, content: str) -> None:
        """Print a conversation message with role formatting."""
        if self.print_conversation:
            print(f"\033[1m{role.capitalize()}:\033[0;0m {content}\n")

    def _get_user_response(self, user_prompt: str, conversation: List) -> List:
        """Get the initial user response for the conversation.

        Args:
            user_prompt (str): The prompt for the user to generate a response.
            conversation (List): The current conversation history.

        Returns:
            List: The updated conversation history with the user's response appended.
        """
        user_response = self.get_response(
            self.clients.clinician, conversation, user_prompt, USER
        )
        self._print_conversation_message(
            role=Role.CLINICIAN.value, content=user_response
        )
        conversation.append({ROLE: USER, CONTENT: user_response})
        return conversation

    def _get_assistant_response(
        self, assistant_prompt: str, conversation: List
    ) -> List:
        """Get the assistant's response for the conversation.

        Args:
            assistant_prompt (str): The prompt for the assistant to generate a response.
            conversation (List): The current conversation history.

        Returns:
            List: The updated conversation history with the assistant's response appended.
        """
        assistant_response = self.get_response(
            self.clients.companion, conversation, assistant_prompt, ASSISTANT
        )
        self._print_conversation_message(
            role=Role.COMPANION.value, content=assistant_response
        )
        conversation.append({ROLE: ASSISTANT, CONTENT: assistant_response})
        return conversation

    @observe()
    async def simulate_conversation(self, initial_user_message: str) -> None:
        """Simulate conversation between the agents.

        Args:
            initial_user_message (str): The initial message from the user to start the conversation.
        """
        if self.session.tags:
            # Add tag to the trace
            langfuse.update_current_trace(tags=self.session.tags)

        # Set the trace ID for the session
        # The trace ID is used to link the conversation to a specific trace in Langfuse
        self.session.trace_id = langfuse.get_current_trace_id()

        user_prompt = self.prompts.clinician
        assistant_prompt = self.prompts.companion

        if self.print_conversation:
            self._print_conversation_message(
                role=Role.CLINICIAN.value, content=initial_user_message
            )

        self.session.created_at = self._get_timestamp()

        # Initial turn
        conversation = [
            {ROLE: SYSTEM, CONTENT: user_prompt},
            {
                ROLE: USER,
                CONTENT: initial_user_message,
            },
        ]

        # Chat loop with progress bar
        with tqdm(
            total=self.max_turns, desc="Conversation progress", unit="turn", leave=False
        ) as pbar:
            for turn in range(self.max_turns):
                try:
                    # Assistant turn
                    conversation = self._get_assistant_response(
                        assistant_prompt=assistant_prompt, conversation=conversation
                    )
                except ValueError as e:
                    print(
                        f"\nConversation ended early due to assistant response error: {e}"
                    )
                    self.session.updated_at = self._get_timestamp()
                    break

                # Check if this is the last turn - if so, don't get clinician response
                if turn == self.max_turns - 1:
                    self.session.updated_at = self._get_timestamp()
                    pbar.update(1)  # Update progress bar for the final turn
                    break

                try:
                    # User turn
                    conversation = self._get_user_response(
                        user_prompt=user_prompt, conversation=conversation
                    )
                except ValueError as e:
                    print(f"\nConversation ended early due to user response error: {e}")
                    self.session.updated_at = self._get_timestamp()
                    break

                self.session.updated_at = self._get_timestamp()
                pbar.update(1)  # Update progress bar after each complete turn

        # Store the conversation history
        self.conversation_history = conversation
