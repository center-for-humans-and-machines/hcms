"""MongoDB document schemas for HPMS conversation data."""

from datetime import datetime
from typing import Literal

from bson import ObjectId
from pydantic import BaseModel, ConfigDict, Field, field_validator


# pylint: disable=too-few-public-methods
class _NonBlankRequiredFieldMixin:
    """Reject whitespace-only values for required identity fields."""

    @field_validator(
        "reviewer_id",
        "reviewer_username",
        "reviewer_by_username",
        "conversation_id",
        "project_id",
        mode="before",
        check_fields=False,
    )
    @classmethod
    def reject_blank_required_field(cls, value: object) -> object:
        """Ensure required string fields contain a non-whitespace value."""
        if isinstance(value, str) and not value.strip():
            raise ValueError("Value must contain at least 1 non-whitespace character")
        return value


class _NonBlankOptionalFieldMixin:
    """Reject whitespace-only values for optional identity fields."""

    @field_validator("flagged_by", mode="before", check_fields=False)
    @classmethod
    def reject_blank_optional_field(cls, value: object) -> object:
        """Ensure optional identity strings are either missing or non-blank."""
        if value is None:
            return None
        if value == "":
            return value
        if isinstance(value, str) and not value.strip():
            raise ValueError(
                "Value must be empty or contain at least 1 non-whitespace character"
            )
        return value


class UserFlagReviewDocument(_NonBlankRequiredFieldMixin, BaseModel):
    """Review of a participant flag by a human reviewer."""

    model_config = ConfigDict(extra="forbid")

    reviewer_id: str = Field(..., min_length=1)
    reviewer_username: str = Field(..., min_length=1)
    approved: bool
    comment: str
    reviewed_at: datetime


class UserFlagDocument(BaseModel):
    """Participant-created flag on a single message."""

    model_config = ConfigDict(extra="forbid")

    category: str
    category_other: str
    reviews: list[UserFlagReviewDocument]


class ReviewerFlagDocument(_NonBlankRequiredFieldMixin, BaseModel):
    """Reviewer-created flag on a single message."""

    model_config = ConfigDict(extra="forbid")

    reviewer_id: str = Field(..., min_length=1)
    reviewer_by_username: str = Field(..., min_length=1)
    created_at: datetime
    categories: list[str]
    category_other: str | None = None
    comment: str


class DuplicateFlagDocument(_NonBlankRequiredFieldMixin, BaseModel):
    """Duplicate flag metadata for a single message."""

    model_config = ConfigDict(extra="forbid")

    reviewer_id: str = Field(..., min_length=1)
    reviewer_username: str = Field(..., min_length=1)
    flagged_at: datetime


class MessageDocument(_NonBlankOptionalFieldMixin, BaseModel):
    """Message document embedded in a conversation."""

    model_config = ConfigDict(extra="forbid")

    content: str = Field(..., min_length=1)
    role: str = Field(..., min_length=1)
    timestamp: datetime
    type: str = Field(..., min_length=1)
    flagged: bool
    flagged_at: datetime | None
    flagged_by: str | None
    flag_category: str | None
    flag_other_reason: str | None
    user_flag: UserFlagDocument | None
    reviewer_flags: list[ReviewerFlagDocument]
    duplicate_flags: list[DuplicateFlagDocument]

    @field_validator("user_flag", mode="before")
    @classmethod
    def normalize_empty_user_flag(cls, value: object) -> object:
        """Treat explicit empty user flag payloads as an absent flag."""
        if value is None:
            return None
        if value == {}:
            return None
        return value


class OpenedByDocument(_NonBlankRequiredFieldMixin, BaseModel):
    """Reviewer open-state entry."""

    model_config = ConfigDict(extra="forbid")

    reviewer_id: str = Field(..., min_length=1)
    opened_at: datetime


class ReviewedMessageDocument(_NonBlankRequiredFieldMixin, BaseModel):
    """Reviewer reviewed-state entry for a specific message."""

    model_config = ConfigDict(extra="forbid")

    reviewer_id: str = Field(..., min_length=1)
    message_index: int = Field(..., ge=0)
    reviewed_at: datetime


class AssignedMessageDocument(_NonBlankRequiredFieldMixin, BaseModel):
    """Assignment metadata for a reviewer and message."""

    model_config = ConfigDict(extra="forbid")

    reviewer_id: str = Field(..., min_length=1)
    message_index: int = Field(..., ge=0)
    reason: Literal["random_sample", "participant_flag", "expert_escalation"]
    assigned_at: datetime


class NaturalnessRatingDocument(_NonBlankRequiredFieldMixin, BaseModel):
    """Human evaluation rating for conversational naturalness."""

    model_config = ConfigDict(extra="forbid")

    reviewer_id: str = Field(..., min_length=1)
    rated_at: datetime
    coherence: int
    topic_progression: int


class RealismRatingDocument(_NonBlankRequiredFieldMixin, BaseModel):
    """Human evaluation rating for conversational realism."""

    model_config = ConfigDict(extra="forbid")

    reviewer_id: str = Field(..., min_length=1)
    rated_at: datetime
    rating: int


class ConversationDocument(_NonBlankRequiredFieldMixin, BaseModel):
    """MongoDB conversation document matching the dashboard schema."""

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        arbitrary_types_allowed=True,
    )

    id: ObjectId = Field(..., alias="_id")
    conversation_id: str = Field(..., min_length=1)
    participant_id: str | None = Field(...)
    model: str = Field(..., min_length=1)
    experiment_id: str = Field(..., min_length=1)
    project_id: str = Field(..., min_length=1)
    created_at: datetime
    custom_system_message_id: str | None
    multi_rounds: bool
    messages: list[MessageDocument]
    opened_by: list[OpenedByDocument]
    assigned_messages: list[AssignedMessageDocument]
    reviewed_messages: list[ReviewedMessageDocument]
    naturalness_ratings: list[NaturalnessRatingDocument] = Field(default_factory=list)
    realism_ratings: list[RealismRatingDocument] = Field(default_factory=list)

    @field_validator("participant_id", mode="before")
    @classmethod
    def normalize_blank_participant_id(cls, value: object) -> object:
        """Treat blank participant IDs as absent instead of invalid."""
        if isinstance(value, str) and not value.strip():
            return None
        return value
