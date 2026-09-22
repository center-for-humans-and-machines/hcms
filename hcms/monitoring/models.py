"""Data schemas for Pydantic validation."""

from pydantic import (
    BaseModel,
    Field,
)


# pylint: disable-next=too-few-public-methods
class BotRatingResponse(BaseModel):
    """Schema for Batch API response with custom rating."""

    # OpenAI does not allow setting a range of valid values like ge=0, le=10
    # 'minimum' is not permitted.
    # https://community.openai.com/t/new-function-calling-with-strict-has-a-problem-with-minimum-integer-type/903258
    rating: int = Field(..., description="Numeric rating value. Range: 0 to 10")
    explanation: str = Field(..., description="Explanation of the rating")


class ConversationTurn(BaseModel):
    """Represents a single turn in a conversation."""

    role: str
    content: str
