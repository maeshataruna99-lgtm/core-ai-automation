"""Tests that keep the demo and the documentation true.

Why: Documentation goes out of date without anyone noticing, and for a portfolio a wrong claim is
worse than a missing one. These tests run the demo for real, and compare the docs with the code.

Used by: `pytest`, locally and in `.github/workflows/ci.yml`.
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from ai_ops.api.app import create_app
from ai_ops.config import ENV_PREFIX, Settings

REPO = Path(__file__).parent.parent
DOCUMENTS = [
    "README.md",
    "docs/API.md",
    "docs/ARCHITECTURE.md",
    "docs/CONFIGURATION.md",
    "docs/EVALUATION.md",
    "docs/OOP_DESIGN_NOTES.md",
    "docs/OVERVIEW_ID.md",
    "docs/PORTFOLIO_POSITIONING.md",
]
# A path that a document names as a real file, such as `src/ai_ops/config.py`.
FILE_REFERENCE = re.compile(r"\b(?:docs|src|tests|data|scripts)/[\w./-]+\.\w+")


def read(relative_path: str) -> str:
    """Read a text file from the repository.

    Why: Tests run from a temporary directory, so paths must start from the repository root.

    Used by: the tests in this module.

    Args:
        relative_path: The path from the repository root.

    Returns:
        The text.
    """
    return (REPO / relative_path).read_text(encoding="utf-8")


def test_demo_runs_the_whole_sequence_and_passes_its_checks():
    """Run the demo as a person would.

    Why: The roadmap requires a demo showing retrieval with sources, a tool lookup, and an action
    held for approval. The script checks each step itself, so this fails if any step stops working.

    Used by: pytest.
    """
    env = {key: value for key, value in os.environ.items() if not key.startswith("AI_OPS_")}

    result = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "demo.py")],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    output = result.stdout
    assert "refund-policy.md#0" in output
    assert "I could not find this in the knowledge base." in output
    assert "lookup_record" in output
    assert "pending_approval" in output
    assert "Demo finished" in output
    assert "Traceback" not in output


def test_committed_openapi_file_matches_the_code():
    """Compare the exported API contract with the running app.

    Why: A reviewer reads this file without running anything, so it must match the real API.
    Regenerate it with the command in `docs/API.md` after changing an endpoint.

    Used by: pytest.
    """
    expected = create_app(Settings(llm_provider="mock")).openapi()

    assert json.loads(read("docs/openapi.json")) == expected


def test_api_guide_covers_every_endpoint():
    """Check that each endpoint appears in the written guide.

    Why: A new endpoint with no example would be easy to forget.

    Used by: pytest.
    """
    guide = read("docs/API.md")
    paths = create_app(Settings(llm_provider="mock")).openapi()["paths"]

    for path in paths:
        assert path.split("{")[0] in guide, path


def test_configuration_guide_covers_every_setting():
    """Check that each setting is documented, and that no removed setting still is.

    Why: A setting nobody knows about cannot be used, and a documented one that no longer exists
    wastes a reader's time.

    Used by: pytest.
    """
    guide = read("docs/CONFIGURATION.md")
    expected = {f"{ENV_PREFIX}{name.upper()}" for name in Settings.model_fields}

    assert set(re.findall(rf"{ENV_PREFIX}[A-Z_]+", guide)) == expected


@pytest.mark.parametrize("document", DOCUMENTS)
def test_files_named_in_the_documents_exist(document):
    """Check every file path that a document mentions.

    Why: The portfolio's evidence table points at files and tests. A claim whose evidence has moved
    would look like an unsupported claim.

    Used by: pytest.

    Args:
        document: The document to check.
    """
    missing = sorted(
        reference
        for reference in set(FILE_REFERENCE.findall(read(document)))
        # The database file is created at startup, so it is not in the repository.
        if not reference.endswith(".db") and not (REPO / reference).exists()
    )

    assert missing == []


def test_every_portfolio_claim_names_evidence():
    """Check that the evidence table is complete.

    Why: The roadmap requires every resume claim to be supported by the repository. Each row must
    name a file that exists.

    Used by: pytest.
    """
    rows = re.findall(r"^\| \d+ \|(.+)$", read("docs/PORTFOLIO_POSITIONING.md"), re.MULTILINE)

    assert len(rows) >= 5
    for row in rows:
        references = FILE_REFERENCE.findall(row)
        assert references, row
        assert all((REPO / reference).exists() for reference in references), row


def test_readme_images_exist():
    """Check the images in the README.

    Why: The README is the first thing a reviewer sees, and a broken image is obvious there.

    Used by: pytest.
    """
    images = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", read("README.md"))

    assert images
    assert all((REPO / image).exists() for image in images), images
