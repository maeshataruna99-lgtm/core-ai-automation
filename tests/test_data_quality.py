"""Checks on the test data itself: the sample documents, orders, golden set, and thresholds.

Why: Every score and every demo depends on this data, so a mistake in it looks like a bug in the
system, or hides one. The project also promises that all data is synthetic, and that needs checking,
because real personal data is easy to paste in by accident.

Used by: `pytest`, locally and in `.github/workflows/ci.yml`.
"""

import json
import re
from pathlib import Path

import pytest

from ai_ops.evaluation.golden import load_golden
from ai_ops.integrations.records import sample_records
from ai_ops.llm.prompt_loader import load_prompt
from ai_ops.retrieval.chunking import chunk_text
from ai_ops.retrieval.documents import load_documents

REPO = Path(__file__).parent.parent
KNOWLEDGE = REPO / "data" / "knowledge"
DOCUMENTS = load_documents(KNOWLEDGE)
TEXT_BY_ID = {document.id: document.text for document in DOCUMENTS}
GOLDEN = load_golden(REPO / "data" / "evaluation" / "golden.json", set(TEXT_BY_ID))
THRESHOLDS = json.loads((REPO / "data" / "evaluation" / "thresholds.json").read_text("utf-8"))
ALL_DATA_FILES = sorted(
    path
    for folder in ("data", "tests/fixtures")
    for path in (REPO / folder).rglob("*")
    # A person's own uploads are ignored by Git, so they are not part of the project's data.
    if path.is_file() and path.suffix in {".md", ".txt", ".json"} and "uploads" not in path.parts
)
SCORE_NAMES = {
    "hit_rate",
    "precision",
    "recall",
    "context_fact_recall",
    "not_found_accuracy",
    "answer_fact_recall",
    "groundedness",
    "citation_rate",
    "injection_resistance",
}
PERSONAL_DATA = {
    "email address": r"[\w.+-]+@[\w-]+\.[a-z]{2,}",
    "phone number": r"(?<![\w-])(?:\+62|08)[\d\s-]{8,}\d",
    "card number": r"\b(?:\d[ -]?){13,16}\b",
    "16-digit id number": r"\b\d{16}\b",
}


@pytest.mark.parametrize("path", ALL_DATA_FILES, ids=lambda path: path.name)
@pytest.mark.parametrize("kind", PERSONAL_DATA)
def test_data_files_hold_no_personal_data(path, kind):
    """Check every data file for common kinds of personal data.

    Why: The project rules forbid real customer data, and these patterns catch the usual slips. This
    cannot prove that data is synthetic, for example it cannot recognise a real name.

    Used by: pytest.

    Args:
        path: The file.
        kind: The kind of data to look for.
    """
    matches = re.findall(PERSONAL_DATA[kind], path.read_text(encoding="utf-8"))

    assert matches == []


@pytest.mark.parametrize("document", DOCUMENTS, ids=lambda document: document.id)
def test_each_knowledge_document_says_it_is_synthetic(document):
    """Check the label on each sample document.

    Why: Anyone who finds these files should see at once that the policies are made up.

    Used by: pytest.

    Args:
        document: The document.
    """
    assert "synthetic sample data" in document.text


@pytest.mark.parametrize("document", DOCUMENTS, ids=lambda document: document.id)
def test_each_knowledge_document_gives_usable_chunks(document):
    """Check that a document survives chunking at the default size.

    Why: An empty or huge document would silently add nothing, or swamp the prompt.

    Used by: pytest.

    Args:
        document: The document.
    """
    chunks = chunk_text(document.text, size=800, overlap=100)

    assert chunks
    assert all(0 < len(chunk) <= 800 for chunk in chunks)


def test_each_knowledge_document_is_used_by_the_golden_set():
    """Check that no document goes untested.

    Why: A document that no question refers to tells the evaluation nothing about it.

    Used by: pytest.
    """
    used = {source for case in GOLDEN.cases for source in case.expected_sources}

    assert used == set(TEXT_BY_ID)


def test_golden_ids_are_unique():
    """Check for repeated ids.

    Why: Reports use the id to name a case, so a repeat would make a result ambiguous.

    Used by: pytest.
    """
    ids = [case.id for case in GOLDEN.cases] + [case.id for case in GOLDEN.injection_cases]

    assert len(ids) == len(set(ids))


def test_golden_questions_are_unique():
    """Check for repeated questions.

    Why: A repeat counts twice, and moves a score for no reason.

    Used by: pytest.
    """
    questions = [case.question.casefold() for case in GOLDEN.cases]

    assert len(questions) == len(set(questions))


@pytest.mark.parametrize("case", GOLDEN.cases, ids=lambda case: case.id)
def test_golden_case_is_well_formed(case):
    """Check the shape of each case.

    Why: An answerable case with no fact could never lower a score, and an unanswerable case with a
    fact makes no sense. Both would be mistakes that nobody would notice.

    Used by: pytest.

    Args:
        case: The case.
    """
    assert case.question.strip().endswith("?")
    assert 0 < len(case.question) <= 1000, "must pass the API's own limit"
    assert bool(case.expected_sources) == bool(case.expected_facts)
    assert all(fact.strip() for fact in case.expected_facts)


@pytest.mark.parametrize(
    "case",
    [case for case in GOLDEN.cases if case.expected_sources],
    ids=lambda case: case.id,
)
def test_each_expected_fact_is_in_its_expected_source(case):
    """Check that each case can actually be passed.

    Why: A fact that its document does not contain could never be found, so it would be a mistake in
    the data that looks like a retrieval failure.

    Used by: pytest.

    Args:
        case: The case.
    """
    text = " ".join(TEXT_BY_ID[source] for source in case.expected_sources).casefold()

    for fact in case.expected_facts:
        assert fact.casefold() in text


@pytest.mark.parametrize("case", GOLDEN.cases, ids=lambda case: case.id)
def test_question_does_not_give_away_its_fact(case):
    """Check that a question does not contain its own answer.

    Why: The answer metric looks for the fact in the answer. A model that just repeated the question
    would otherwise pass. `refund-late` asks about "30 days" and expects "store credit", which is
    fine, since only the expected fact matters.

    Used by: pytest.

    Args:
        case: The case.
    """
    for fact in case.expected_facts:
        assert fact.casefold() not in case.question.casefold()


def test_golden_set_covers_both_kinds_of_question():
    """Check the balance of the set.

    Why: With no unanswerable questions, a system that always answers would score perfectly.

    Used by: pytest.
    """
    answerable = [case for case in GOLDEN.cases if case.expected_sources]
    unanswerable = [case for case in GOLDEN.cases if not case.expected_sources]

    assert len(answerable) >= 20
    assert len(unanswerable) >= 5


@pytest.mark.parametrize("case", GOLDEN.injection_cases, ids=lambda case: case.id)
def test_injection_case_can_detect_an_attack(case):
    """Check each attack document.

    Why: The canary must be in the attack, or obeying could not show up, and it must be absent from
    the question and the real documents, or an honest answer could set it off.

    Used by: pytest.

    Args:
        case: The attack.
    """
    assert case.canary in case.document
    assert case.canary not in case.question
    assert not any(case.canary in text for text in TEXT_BY_ID.values())


def test_thresholds_are_valid_scores():
    """Check the thresholds file.

    Why: A minimum above 1 can never pass, a negative one can never fail, and a misspelled name
    would remove the check. An empty section means no gate, so the mock's must not be empty.

    Used by: pytest.
    """
    assert THRESHOLDS["mock"]
    for provider, minimums in THRESHOLDS.items():
        assert set(minimums) <= SCORE_NAMES, provider
        assert all(0 <= value <= 1 for value in minimums.values()), provider


def test_each_provider_has_a_thresholds_section():
    """Check that the file covers each configured provider.

    Why: A provider with no section would run with no regression gate, without anyone noticing.

    Used by: pytest.
    """
    from ai_ops.config import Settings

    annotation = Settings.model_fields["llm_provider"].annotation

    assert set(annotation.__args__) == set(THRESHOLDS)


def test_sample_orders_are_valid_and_synthetic():
    """Check the sample orders.

    Why: Tests and the demo rely on these ids and statuses, and customers must be plainly made up.

    Used by: pytest.
    """
    records = sample_records()

    assert len({record.id for record in records}) == len(records)
    for record in records:
        assert re.fullmatch(r"ORD-[0-9]+", record.id)
        assert record.status in ("processing", "shipped", "delivered", "cancelled")
        assert record.customer.startswith("Sample Customer")


@pytest.mark.parametrize(
    ("name", "placeholders"),
    [
        ("grounded_answer_system", set()),
        ("grounded_answer_user", {"context", "question"}),
        ("agent_system", set()),
        ("ticket_classify", {"ticket"}),
        ("ticket_draft_system", set()),
        ("ticket_draft_user", {"record", "context", "ticket"}),
        ("record_extract_system", set()),
        ("record_extract_user", {"text"}),
    ],
)
def test_prompts_have_the_placeholders_their_callers_fill(name, placeholders):
    """Check each prompt file against what the code passes.

    Why: A misspelled placeholder fails only when that path runs, and with a real model that costs
    money. A system prompt must have none, so untrusted text can never reach the instructions.

    Used by: pytest.

    Args:
        name: The prompt.
        placeholders: The placeholders it must use.
    """
    text = load_prompt(name, version=1).text

    assert set(re.findall(r"\$(\w+)", text)) == placeholders


@pytest.mark.parametrize(
    "name",
    ["grounded_answer_user", "ticket_classify", "ticket_draft_user", "record_extract_user"],
)
def test_user_prompts_end_with_the_untrusted_text(name):
    """Check where the untrusted text sits.

    Why: The mock provider echoes the last line, and ending with the data also keeps instructions
    from following text that an attacker controls.

    Used by: pytest.

    Args:
        name: The prompt.
    """
    last_line = load_prompt(name, version=1).text.strip().splitlines()[-1]

    assert re.fullmatch(r"\$\w+", last_line)
