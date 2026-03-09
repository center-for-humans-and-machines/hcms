"""MongoDB document schemas for HPMS conversation data."""

from datetime import datetime
from typing import Any, List, Literal

from pydantic import BaseModel, ConfigDict, Field


class UserFlagReviewDocument(BaseModel):
    """Review of a participant flag by a human reviewer."""

    model_config = ConfigDict(extra="forbid")

    reviewer_id: str = Field(..., min_length=1)
    reviewer_username: str | None = Field(default=None, min_length=1)
    approved: bool
    comment: str = ""
    reviewed_at: datetime


class UserFlagDocument(BaseModel):
    """Participant-created flag on a single message."""

    model_config = ConfigDict(extra="forbid")

    category: str = ""
    category_other: str = ""
    created_at: datetime | None = None
    created_by: str | None = Field(default=None, min_length=1)
    reviews: List[UserFlagReviewDocument] = Field(default_factory=list)


class ReviewerFlagDocument(BaseModel):
    """Reviewer-created flag on a single message."""

    model_config = ConfigDict(extra="forbid")

    reviewer_id: str = Field(..., min_length=1)
    reviewer_by_username: str | None = Field(default=None, min_length=1)
    created_at: datetime
    categories: List[str] = Field(default_factory=list)
    category_other: str = ""
    comment: str = ""


class DuplicateFlagDocument(BaseModel):
    """Duplicate flag metadata for a single message."""

    model_config = ConfigDict(extra="forbid")

    reviewer_id: str = Field(..., min_length=1)
    reviewer_username: str | None = Field(default=None, min_length=1)
    flagged_at: datetime


class MessageDocument(BaseModel):
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


class OpenedByDocument(BaseModel):
    """Reviewer open-state entry."""

    model_config = ConfigDict(extra="forbid")

    reviewer_id: str = Field(..., min_length=1)
    opened_at: datetime


class ReviewedByDocument(BaseModel):
    """Reviewer reviewed-state entry."""

    model_config = ConfigDict(extra="forbid")

    reviewer_id: str = Field(..., min_length=1)
    reviewed_at: datetime


class ConversationAssignmentDocument(BaseModel):
    """Assignment metadata for a reviewer and conversation."""

    model_config = ConfigDict(extra="forbid")

    reviewer_id: str = Field(..., min_length=1)
    reason: str = Field(..., min_length=1)
    assigned_at: datetime


class NaturalnessRatingDocument(BaseModel):
    """Naturalness rating metadata for a conversation."""

    model_config = ConfigDict(extra="forbid")

    reviewer_id: str = Field(..., min_length=1)
    coherence: int = Field(..., ge=1, le=5)
    topic_progression: int = Field(..., ge=1, le=5)
    rated_at: datetime


class RealismRatingDocument(BaseModel):
    """Realism rating metadata for a conversation."""

    model_config = ConfigDict(extra="forbid")

    reviewer_id: str = Field(..., min_length=1)
    rating: Literal[0, 5, 10]
    rated_at: datetime


class ConversationDocument(BaseModel):
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
    naturalness_ratings: List[NaturalnessRatingDocument] = Field(default_factory=list)
    realism_ratings: List[RealismRatingDocument] = Field(default_factory=list)
