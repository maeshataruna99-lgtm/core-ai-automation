"""Shared test setup.

Why: The default database path is relative, so running each test from its own temporary directory
gives it its own empty database. No test can see another test's data or touch a real file.

Used by: pytest, which loads this file automatically and applies the fixture to every test.
"""

import pytest


@pytest.fixture(autouse=True)
def run_in_temporary_directory(tmp_path, monkeypatch):
    """Run each test from a new temporary directory.

    Why: See the module docstring.

    Used by: pytest, for every test.

    Args:
        tmp_path: Pytest's temporary directory.
        monkeypatch: Pytest's tool for changes that are undone after the test.
    """
    monkeypatch.chdir(tmp_path)
