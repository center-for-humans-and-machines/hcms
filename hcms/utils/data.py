"""Data utilities.

This module contains utilities for data handling and validation.
"""

from functools import wraps
from pathlib import Path
from typing import ClassVar, List, Set, Type, TypeVar

import polars as pl
from pydantic import BaseModel, ValidationError, field_validator

T = TypeVar("T", bound=BaseModel)


def validate_schema(schema: Type[T]):
    """Validate DataFrame against Pydantic schema.

    Args:
        schema: Pydantic model class for row validation
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs) -> pl.DataFrame:
            df = func(*args, **kwargs)
            validated_rows = []
            try:
                # Validate and return clean data
                validated_rows = [schema(**row).model_dump() for row in df.to_dicts()]
            except ValidationError as exc:
                print("Pydantic validation errors:\n", repr(exc.errors()))
            return pl.DataFrame(validated_rows)

        return wrapper

    return decorator

# pylint: disable-next=too-few-public-methods
class FileValidatorMixin(BaseModel):
    """Mixin for file validation.

    This mixin provides a method to validate that a file exists.
    """

    VALIDATE_FIELDS: ClassVar[Set[str]] = set()

    @field_validator("*")
    @classmethod
    def validate_file_exists(cls, v: Path, info) -> Path:
        """Validate that file exists for fields in VALIDATE_FIELDS."""
        if info.field_name in cls.VALIDATE_FIELDS:
            if not v.exists():
                raise ValueError(f"File {v} does not exist")
        return v


# pylint: disable-next=too-few-public-methods
class DataConfig(FileValidatorMixin):
    """Configuration for data loading.

    Check if a file exists at the given path.
    """

    VALIDATE_FIELDS = {"file_path"}
    file_path: Path


def load_json_file(file_path: Path) -> pl.DataFrame:
    """Load JSON file into DataFrame.

    Args:
        file_path: Path to JSON file.

    Returns:
        pl.DataFrame: DataFrame with JSON data.
    """
    return pl.read_json(file_path, infer_schema_length=int(1e10))


def convert_date_columns(df: pl.DataFrame, date_cols: List[str]) -> pl.DataFrame:
    """Convert date columns to datetime objects.

    Args:
        df: DataFrame to convert date columns.
        date_cols: List of date columns to convert.

    Returns:
        pl.DataFrame: DataFrame with converted date columns.
    """
    for col in date_cols:
        df = df.with_columns(
            pl.col(col).str.strptime(pl.Datetime, strict=False).alias(col)
        )
    return df
