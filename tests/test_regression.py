"""Regression test for LLM responses."""

from hcms.monitoring import RegTestProcessor


def test_regression():
    """Test the regression of the LLM responses."""

    processor = RegTestProcessor()

    results = processor.process_batch()

    assert processor.validate_results(  # noqa
        results
    ), "Regression test failed: Results validation failed"

    assert results is not None, "Regression test failed: No results returned"
    assert len(results) > 0, "Regression test failed: No results returned"
