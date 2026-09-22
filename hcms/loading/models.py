"""Data schemas for Pydantic validation."""

from datetime import datetime
from enum import Enum
from typing import Annotated, List, Optional, Union

from pydantic import BaseModel, Field, field_validator


class Role(str, Enum):
    """Enum for conversation roles."""

    CLINICIAN = "clinician"
    COMPANION = "companion"


class ConversationType(str, Enum):
    """Enum for conversation types."""

    OPEN_ENDED = "open-ended"
    STANDARDIZED_SAFETY = "standardized-safety-evaluation"


# pylint: disable=too-few-public-methods
class RegTestPrompt(BaseModel):
    """Schema for regression test prompts."""

    Question: str = Field(..., min_length=1)
    Title: str = Field(..., min_length=1)
    Situation: str = Field(..., min_length=1)
    Ideal_Response: str = Field(..., min_length=1)


class ConversationMessage(BaseModel):
    """Schema for a single conversation message."""

    role: Role = Field(...)
    content: str = Field(..., min_length=1)
    turn: int = Field(..., ge=0)


class ConversationDataset(BaseModel):
    """Schema for conversation dataset entries."""

    conversation_id: str = Field(..., min_length=1)
    conversation: List[ConversationMessage] = Field(..., min_length=1)
    conversation_type: ConversationType = Field(...)
    model: str = Field(..., min_length=1)
    model_provider: str = Field(..., min_length=1)
    api_version: Optional[str] = None
    temperature: float = Field(..., ge=0.0, le=2.0)
    companion_system_prompt: str = Field(..., min_length=1)
    clinician_system_prompt: str = Field(..., min_length=1)
    created_at: datetime
    updated_at: datetime
    conversation_duration_s: float = Field(..., ge=0.0)


class OriginalRole(str, Enum):
    """Enum for original conversation roles."""

    USER = "user"
    ASSISTANT = "assistant"


# def parse_string_list(v):
#         # If the input is a string, try to parse it as JSON
#         if isinstance(v, str):
#             validated = ast.literal_eval(v)
#             try:
#                 [int(i) for i in validated]
#                 return v
#             except (ValueError, TypeError):
#                 raise ValueError("Invalid list format")
#         return v


class MessageDataset(BaseModel):
    """Schema for conversation dataset entries."""

    conversation_id: str = Field(..., min_length=1)
    turn: int = Field(..., ge=0)
    role: OriginalRole = Field(...)
    message: str = Field(..., min_length=1)
    OpenAI_Moderation: Annotated[
        List[Union[int, str]], Field(..., min_length=21, max_length=21)
    ]

    @field_validator("OpenAI_Moderation")
    @classmethod
    def validate_moderation_values(cls, v):
        """Validate OpenAI_Moderation: list of any strings or integers."""
        if not v:
            raise ValueError("OpenAI_Moderation cannot be empty")

        # Allow any combination of integers and strings
        if all(isinstance(item, (int, str)) for item in v):
            return v

        raise ValueError("OpenAI_Moderation must contain only integers or strings")
