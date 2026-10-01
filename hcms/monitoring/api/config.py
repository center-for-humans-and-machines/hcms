"""Configuration for batch processing with OpenAI API."""

from pathlib import Path
from typing import Callable, Optional, Type

import polars as pl
from hcms.constants import DATA_DIR
from hcms.utils import UTF_8, FileValidatorMixin, get_env_variable
from pydantic import BaseModel, Field, field_validator


class BatchConfig(FileValidatorMixin):
    """Configuration for batch processing with OpenAI API."""

    # API Settings
    system_prompt: str = Field(..., min_length=1)
    """System prompt for the model"""

    batch_model: str = Field(
        default_factory=lambda: get_env_variable("BATCH_MODEL_NAME")
    )
    """Model name for batch processing"""

    chat_completions_model: str = Field(
        default_factory=lambda: get_env_variable("CHAT_COMPLETIONS_MODEL_NAME")
    )
    """Model for chat completions"""

    endpoint: str = Field(default_factory=lambda: get_env_variable("MODEL_ENDPOINT"))
    """OpenAI/Azure endpoint URL"""

    response_format_model: Optional[Type[BaseModel]] = Field(
        None, description="Model for response format"
    )

    temperature: float = Field(default=1.0, ge=0.0, le=1.0)
    """Temperature for sampling to control randomness"""

    # Batch Settings
    VALIDATE_FIELDS = {"input_file"}
    input_file: Path
    """Path to input file for batch processing"""

    data_loader: Callable[[Path], pl.DataFrame] = pl.read_json
    """Function to load data from input file"""

    output_dir: Path = Field(default_factory=lambda: DATA_DIR / "batch")
    """Directory for batch output files"""

    encoding: str = UTF_8
    """File encoding for read/write operations"""

    @property
    def is_endpoint_azure(self) -> bool:
        """Check if the endpoint is an Azure endpoint."""
        return "azure" in self.endpoint

    @property
    def is_endpoint_together(self) -> bool:
        """Check if the endpoint is an together.ai endpoint."""
        return "together" in self.endpoint

    @property
    def is_endpoint_openai(self) -> bool:
        """Check if the endpoint is an OpenAI endpoint."""
        return not self.is_endpoint_azure and "openai" in self.endpoint

    @property
    def is_endpoint_google_openai(self) -> bool:
        """Check if the endpoint is an OpenAI-compatible Google endpoint."""
        return self.is_endpoint_openai and "google" in self.endpoint

    @field_validator("output_dir")
    @classmethod
    def validate_output_dir(cls, v: Path) -> Path:
        """Create output directory if it doesn't exist."""
        v.mkdir(exist_ok=True, parents=True)
        return v
