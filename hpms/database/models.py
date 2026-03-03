"""MongoDB document schemas for HPMS conversation data."""

from datetime import datetime
from typing import Any, List

from pydantic import BaseModel, ConfigDict, Field


class UserFlagReviewDocument(BaseModel):
    """Review of a participant flag by a human reviewer."""

    model_config = ConfigDict(extra="forbid")

    reviewer_id: str = Field(..., min_length=1)
    approved: bool
    comment: str = ""
    reviewed_at: datetime


class UserFlagDocument(BaseModel):
    """Participant-created flag on a single message."""

    model_config = ConfigDict(extra="forbid")

    category: str = ""
    category_other: str = ""
    reviews: List[UserFlagReviewDocument] = Field(default_factory=list)


class ReviewerFlagDocument(BaseModel):
    """Reviewer-created flag on a single message."""

    model_config = ConfigDict(extra="forbid")

    reviewer_id: str = Field(..., min_length=1)
    created_at: datetime
    categories: List[str] = Field(default_factory=list)
    category_other: str = ""


class MessageDocument(BaseModel):
    """Message document embedded in a conversation."""

    model_config = ConfigDict(extra="forbid")

    content: str = Field(..., min_length=1)
    role: str = Field(..., min_length=1)
    timestamp: datetime
    type: str = Field(..., min_length=1)
    user_flag: UserFlagDocument = Field(default_factory=UserFlagDocument)
    reviewer_flags: List[ReviewerFlagDocument] = Field(default_factory=list)


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


class ConversationDocument(BaseModel):
    """MongoDB conversation document matching the dashboard schema."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    id: Any = Field(..., alias="_id")
    participant_id: str = Field(..., min_length=1)
    model: str = Field(..., min_length=1)
    experiment_id: str = Field(..., min_length=1)
    created_at: datetime
    messages: List[MessageDocument] = Field(...)
    opened_by: List[OpenedByDocument] = Field(default_factory=list)
    reviewed_by: List[ReviewedByDocument] = Field(default_factory=list)
    assigned_to: List[ConversationAssignmentDocument] = Field(default_factory=list)
