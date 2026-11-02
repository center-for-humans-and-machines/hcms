"""Test that regression test prompts load through the validated loader."""

import polars as pl

from hcms.loading import DATASET_2_CONVERSATION_STARTERS, load_regression_test_prompts
from hcms.loading.models import RegTestPrompt
from hcms.monitoring.constants import RegTestColumns


def test_regression_test_prompts_expose_schema_columns():
    """Every CSV row validates and the frame has exactly the schema's fields."""
    df = load_regression_test_prompts()

    assert df.columns == list(RegTestPrompt.model_fields)
    assert df.height == pl.read_csv(DATASET_2_CONVERSATION_STARTERS).height
    assert df.height > 0


def test_regression_processor_input_column_is_loaded():
    """The column RegTestProcessor reads exists in the validated frame."""
    df = load_regression_test_prompts()

    assert RegTestColumns.QUESTION in df.columns
    assert df.get_column(RegTestColumns.QUESTION).null_count() == 0
