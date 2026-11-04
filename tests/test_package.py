"""Smoke test for the project setup.

Why: There is no application code yet, and pytest exits with an error when it collects
no tests. This test gives CI something real to run and proves the install works.

Used by: `pytest`, locally and in `.github/workflows/ci.yml`.
"""

from importlib.metadata import version

import ai_ops


def test_package_is_installed_from_src():
    """Check that `ai_ops` is installed and importable.

    Why: `version()` reads the metadata that pip writes during install, so it raises
    `PackageNotFoundError` if `pip install -e .` was skipped or the name in
    `pyproject.toml` is wrong. The import alone would not catch a missing install
    if `src/` happened to be on the path.

    Used by: pytest, which collects functions named `test_*`.
    """
    assert version("ai-ops")
    assert ai_ops.__doc__
