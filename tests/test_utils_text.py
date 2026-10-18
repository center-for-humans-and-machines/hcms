"""Unit tests for hcms.utils.text module."""

from hcms.utils import clean_text


class TestCleanText:
    """Test cases for the clean_text function."""

    def test_clean_text_removes_extra_whitespace(self):
        """Test that clean_text removes extra whitespace."""
        input_text = "  Hello   world  \n\n  "
        expected = "Hello   world"
        result = clean_text(input_text)
        assert result == expected

    def test_clean_text_handles_empty_string(self):
        """Test that clean_text handles empty strings."""
        input_text = ""
        expected = ""
        result = clean_text(input_text)
        assert result == expected

    def test_clean_text_handles_whitespace_only(self):
        """Test that clean_text handles whitespace-only strings."""
        input_text = "   \n\t  "
        expected = ""
        result = clean_text(input_text)
        assert result == expected

    def test_clean_text_preserves_single_spaces(self):
        """Test that clean_text preserves single spaces between words."""
        input_text = "Hello world"
        expected = "Hello world"
        result = clean_text(input_text)
        assert result == expected

    def test_clean_text_handles_tabs_and_newlines(self):
        """Test that clean_text handles tabs and newlines properly."""
        input_text = "Hello\tworld\ntest"
        expected = "Hello\tworld test"
        result = clean_text(input_text)
        assert result == expected
