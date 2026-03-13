"""MongoDB document schemas for HPMS conversation data."""

from datetime import datetime
from typing import Any, List, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


# pylint: disable=too-few-public-methods
class _BlankOptionalIdentityFieldMixin:
    """Normalize blank optional identity fields to ``None``."""

    @field_validator(
        "reviewer_username",
        "created_by",
        "reviewer_by_username",
        "flagged_by",
        mode="before",
        check_fields=False,
    )
    @classmethod
    def blank_optional_identity_field_to_none(cls, value: Any) -> Any:
        """Treat blank legacy identity strings as missing values."""
        if value is None:
            return None
        if isinstance(value, str) and not value.strip():
            return None
        return value


class _NonBlankRequiredIdentityFieldMixin:
    """Reject whitespace-only values for required identity fields."""

    @field_validator(
        "reviewer_id",
        "participant_id",
        "experiment_id",
        "conversation_id",
        "project_id",
        mode="before",
        check_fields=False,
    )
    @classmethod
    def reject_blank_required_identity_field(cls, value: Any) -> Any:
        """Ensure required identity fields contain a non-whitespace value."""
        if isinstance(value, str) and not value.strip():
            raise ValueError("Value must contain at least 1 non-whitespace character")
        return value


class UserFlagReviewDocument(
    _BlankOptionalIdentityFieldMixin, _NonBlankRequiredIdentityFieldMixin, BaseModel
):
    """Review of a participant flag by a human reviewer."""

    model_config = ConfigDict(extra="forbid")

    reviewer_id: str = Field(..., min_length=1)
    reviewer_username: str | None = Field(default=None, min_length=1)
    approved: bool
    comment: str = ""
    reviewed_at: datetime


class UserFlagDocument(_BlankOptionalIdentityFieldMixin, BaseModel):
    """Participant-created flag on a single message."""

    model_config = ConfigDict(extra="forbid")

    category: str = ""
    category_other: str = ""
    created_at: datetime | None = None
    created_by: str | None = Field(default=None, min_length=1)
    reviews: List[UserFlagReviewDocument] = Field(default_factory=list)


class ReviewerFlagDocument(
    _BlankOptionalIdentityFieldMixin, _NonBlankRequiredIdentityFieldMixin, BaseModel
):
    """Reviewer-created flag on a single message."""

    model_config = ConfigDict(extra="forbid")

    reviewer_id: str = Field(..., min_length=1)
    reviewer_by_username: str | None = Field(default=None, min_length=1)
    created_at: datetime
    categories: List[str] = Field(default_factory=list)
    category_other: str = ""
    comment: str = ""


class DuplicateFlagDocument(
    _BlankOptionalIdentityFieldMixin, _NonBlankRequiredIdentityFieldMixin, BaseModel
):
    """Duplicate flag metadata for a single message."""

    model_config = ConfigDict(extra="forbid")

    reviewer_id: str = Field(..., min_length=1)
    reviewer_username: str | None = Field(default=None, min_length=1)
    flagged_at: datetime


class MessageDocument(_BlankOptionalIdentityFieldMixin, BaseModel):
    """Message document embedded in a conversation."""

    model_config = ConfigDict(extra="forbid")

    content: str = Field(..., min_length=1)
    role: str = Field(..., min_length=1)
    timestamp: datetime
    type: str = Field(..., min_length=1)
    flagged: bool | None = None
    flagged_at: datetime | None = None
    flagged_by: str | None = Field(default=None, min_length=1)
    flag_category: str | None = None
    flag_other_reason: str | None = None
    user_flag: UserFlagDocument | None = None
    reviewer_flags: List[ReviewerFlagDocument] = Field(default_factory=list)
    duplicate_flags: List[DuplicateFlagDocument] = Field(default_factory=list)


class OpenedByDocument(_NonBlankRequiredIdentityFieldMixin, BaseModel):
    """Reviewer open-state entry."""

    model_config = ConfigDict(extra="forbid")

    reviewer_id: str = Field(..., min_length=1)
    opened_at: datetime


class ReviewedByDocument(_NonBlankRequiredIdentityFieldMixin, BaseModel):
    """Reviewer reviewed-state entry."""

    model_config = ConfigDict(extra="forbid")

    reviewer_id: str = Field(..., min_length=1)
    reviewed_at: datetime


class ConversationAssignmentDocument(_NonBlankRequiredIdentityFieldMixin, BaseModel):
    """Assignment metadata for a reviewer and conversation."""

    model_config = ConfigDict(extra="forbid")

    reviewer_id: str = Field(..., min_length=1)
    reason: str = Field(..., min_length=1)
    assigned_at: datetime


class MessageAssignmentDocument(_NonBlankRequiredIdentityFieldMixin, BaseModel):
    """Assignment metadata for a reviewer and a specific message index."""

    model_config = ConfigDict(extra="forbid")

    reviewer_id: str = Field(..., min_length=1)
    message_index: int
    reason: str = Field(..., min_length=1)
    assigned_at: datetime


class ReviewedMessageDocument(_NonBlankRequiredIdentityFieldMixin, BaseModel):
    """Reviewed-state entry for a specific message index."""

    model_config = ConfigDict(extra="forbid")

    reviewer_id: str = Field(..., min_length=1)
    message_index: int
    reviewed_at: datetime


class NaturalnessRatingDocument(_NonBlankRequiredIdentityFieldMixin, BaseModel):
    """Naturalness rating metadata for a conversation."""

    model_config = ConfigDict(extra="forbid")

    reviewer_id: str = Field(..., min_length=1)
    coherence: int = Field(..., ge=1, le=5)
    topic_progression: int = Field(..., ge=1, le=5)
    rated_at: datetime


class RealismRatingDocument(_NonBlankRequiredIdentityFieldMixin, BaseModel):
    """Realism rating metadata for a conversation."""

    model_config = ConfigDict(extra="forbid")

    reviewer_id: str = Field(..., min_length=1)
    rating: Literal[0, 5, 10]
    rated_at: datetime


class ConversationDocument(_NonBlankRequiredIdentityFieldMixin, BaseModel):
    """MongoDB conversation document matching the dashboard schema."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    id: Any = Field(..., alias="_id")
    participant_id: str = Field(..., min_length=1)
    model: str = Field(..., min_length=1)
    experiment_id: str = Field(..., min_length=1)
    conversation_id: str = Field(..., min_length=1)
    project_id: str = Field(..., min_length=1)
    created_at: datetime
    custom_system_message_id: str | None = None
    multi_rounds: bool = True
    messages: List[MessageDocument] = Field(...)
    opened_by: List[OpenedByDocument] = Field(default_factory=list)
    reviewed_by: List[ReviewedByDocument] = Field(default_factory=list)
    assigned_to: List[ConversationAssignmentDocument] = Field(default_factory=list)
    assigned_messages: List[MessageAssignmentDocument] = Field(default_factory=list)
    reviewed_messages: List[ReviewedMessageDocument] = Field(default_factory=list)
    naturalness_ratings: List[NaturalnessRatingDocument] = Field(default_factory=list)
    realism_ratings: List[RealismRatingDocument] = Field(default_factory=list)
