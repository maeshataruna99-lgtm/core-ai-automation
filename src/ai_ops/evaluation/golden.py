"""Load and validate the golden question set.

Why: The golden set is the definition of "correct" for this project, so mistakes in it must be
caught. If a document is renamed, a case must fail loudly and not just lower the score.

Used by: `ai_ops.evaluation.__main__.main` and `ai_ops.evaluation.runner`.
"""

from collections.abc import Iterable
from pathlib import Path

from pydantic import BaseModel, ValidationError


class GoldenError(Exception):
    """The golden file is missing or invalid.

    Why: This lets the command print one clear message and exit with code 2.

    Used by: raised by `load_golden`, caught in `ai_ops.evaluation.__main__.main`.
    """


class GoldenCase(BaseModel):
    """One question with its expected result.

    Why: An empty `expected_sources` marks a question the knowledge base cannot answer, for which
    the correct result is "not found".

    Used by: `GoldenSet` and `ai_ops.evaluation.runner`.
    """

    id: str
    question: str
    expected_sources: list[str]
    expected_facts: list[str]


class InjectionCase(BaseModel):
    """A document that attacks the prompt, and the code word that proves the attack worked.

    Why: A canary is a made-up code word. It appears in the answer only if the model obeyed the
    document, which makes the result simple to check.

    Used by: `GoldenSet` and `ai_ops.evaluation.runner`.
    """

    id: str
    question: str
    document: str
    canary: str


class GoldenSet(BaseModel):
    """The full question set.

    Why: One validated object for the runner to read.

    Used by: `load_golden` and `ai_ops.evaluation.runner`.
    """

    cases: list[GoldenCase]
    injection_cases: list[InjectionCase]


def load_golden(path: Path, known_documents: Iterable[str]) -> GoldenSet:
    """Read the golden file and check that its sources exist.

    Why: A case that names a missing document could never pass.

    Used by: `ai_ops.evaluation.__main__.main` and tests.

    Args:
        path: The JSON file.
        known_documents: The ids of the loaded documents.

    Returns:
        The validated set.

    Raises:
        GoldenError: The file cannot be read, is invalid, or names an unknown document.
    """
    try:
        golden = GoldenSet.model_validate_json(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise GoldenError(f"Cannot read {path}: {exc.strerror}") from None
    except ValidationError as exc:
        raise GoldenError(f"Invalid golden file {path}: {exc.error_count()} problem(s)") from None

    known = set(known_documents)
    for case in golden.cases:
        missing = set(case.expected_sources) - known
        if missing:
            raise GoldenError(f"Case {case.id} names unknown documents: {sorted(missing)}")
    return golden
