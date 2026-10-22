"""Test that path constants point to existing files."""

import inspect
from pathlib import Path

from hcms.loading import constants


def test_all_path_constants_exist():
    """Test that all constants ending in '_PATH' point to existing files."""
    # Get all attributes from the constants module
    all_attrs = inspect.getmembers(constants)

    # Filter for variables ending in '_PATH'
    path_vars = [
        (name, value)
        for name, value in all_attrs
        if name.endswith("_PATH") and hasattr(value, "exists")
    ]

    # Test each path exists
    for name, path in path_vars:
        assert path.exists(), f"File does not exist: {name} = {path}"


def test_found_expected_number_of_paths():
    """Test that we found the expected number of _PATH constants."""
    all_attrs = inspect.getmembers(constants)
    path_vars = [
        name
        for name, value in all_attrs
        if name.endswith("_PATH") and hasattr(value, "exists")
    ]

    # Adjust this number based on how many _PATH constants you expect
    message = f"Expected at least 4 _PATH constants, found: {path_vars}"
    assert len(path_vars) >= 4, message


def test_path_constants_are_path_objects():
    """Test that all _PATH constants are Path objects."""

    all_attrs = inspect.getmembers(constants)
    path_vars = [(name, value) for name, value in all_attrs if name.endswith("_PATH")]

    for name, value in path_vars:
        message = f"{name} should be a Path object, got {type(value)}"
        assert isinstance(value, Path), message
