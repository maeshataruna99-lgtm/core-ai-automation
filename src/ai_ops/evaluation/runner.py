"""Run the golden set through the real RAG service and build a report.

Why: This uses the same `RagService` as `POST /ask`, so it measures what users get. Scores are at
document level, because the expected sources name documents and chunk ids change with chunk size.

Used by: `ai_ops.evaluation.__main__.main` and `tests/test_evaluation.py`.
"""

import hashlib
import subprocess
from datetime import UTC, datetime
from typing import Any

from ai_ops.config import Settings
from ai_ops.evaluation.golden import GoldenCase, GoldenSet, InjectionCase
from ai_ops.evaluation.metrics import (
    citation_found,
    fact_recall,
    groundedness,
    mean,
    percentile,
    precision,
    recall,
)
from ai_ops.llm.factory import build_provider
from ai_ops.llm.provider import LLMProvider
from ai_ops.retrieval.documents import Document
from ai_ops.retrieval.vector_store import InMemoryVectorStore
from ai_ops.services.rag import PROMPT_VERSION, RagAnswer, RagService


def _build_service(settings: Settings, provider: LLMProvider) -> RagService:
    """Create a RAG service with an empty store.

    Why: Each run and each injection case needs its own store, so nothing leaks between them.

    Used by: `run_evaluation` and `_measure_injection`.

    Args:
        settings: The settings.
        provider: The model provider.

    Returns:
        The service.
    """
    return RagService(
        provider=provider,
        store=InMemoryVectorStore(),
        chunk_size=settings.rag_chunk_size,
        chunk_overlap=settings.rag_chunk_overlap,
        top_k=settings.rag_top_k,
        min_score=settings.rag_min_score,
    )


def _score_case(case: GoldenCase, result: RagAnswer, scores_answers: bool) -> dict[str, Any]:
    """Score one question.

    Why: Keeping each case's detail in the report shows which question got worse.

    Used by: `run_evaluation`.

    Args:
        case: The question and its expected result.
        result: What the service returned.
        scores_answers: Whether the answer came from a real model.

    Returns:
        The scores for this case. Answer scores are None if they were not measured.
    """
    documents = [source.chunk.document_id for source in result.sources]
    source_ids = [source.chunk.source_id for source in result.sources]
    context = "\n".join(source.chunk.text for source in result.sources)
    scored = {
        "id": case.id,
        "answerable": bool(case.expected_sources),
        "found": result.found,
        "retrieved": source_ids,
        "top_score": round(result.sources[0].score, 3) if result.sources else None,
        "hit": bool(set(documents) & set(case.expected_sources)),
        "precision": precision(documents, case.expected_sources),
        "recall": recall(documents, case.expected_sources),
        "context_fact_recall": fact_recall(case.expected_facts, context),
        "missing_facts": [
            fact for fact in case.expected_facts if fact.casefold() not in context.casefold()
        ],
        "latency_ms": result.latency_ms,
        "input_tokens": result.usage.input_tokens,
        "output_tokens": result.usage.output_tokens,
        "answer_fact_recall": None,
        "groundedness": None,
        "citation": None,
    }
    if scores_answers and result.found:
        scored["answer_fact_recall"] = fact_recall(case.expected_facts, result.answer)
        scored["groundedness"] = groundedness(result.answer, context)
        scored["citation"] = citation_found(result.answer, source_ids)
    return scored


async def _measure_injection(
    settings: Settings, provider: LLMProvider, cases: list[InjectionCase]
) -> float | None:
    """Return the share of attacks that the model resisted.

    Why: Each attack document is the only document in its store, and a threshold of -1 keeps it from
    being filtered out, so the model always reads it. Otherwise a score of 1.0 could just mean that
    the attack was never retrieved.

    Used by: `run_evaluation`.

    Args:
        settings: The settings.
        provider: The model provider.
        cases: The attacks.

    Returns:
        A value from 0 to 1, or None if there are no cases.
    """
    resisted = []
    for case in cases:
        service = _build_service(settings.model_copy(update={"rag_min_score": -1.0}), provider)
        await service.ingest([Document(id=f"{case.id}.md", text=case.document)])
        result = await service.ask(case.question)
        resisted.append(float(case.canary.casefold() not in result.answer.casefold()))
    return mean(resisted)


def _git_state() -> tuple[str | None, bool | None]:
    """Read the current commit and whether the working tree has changes.

    Why: This ties a stored number to the exact code. A report from a changed tree is marked, so it
    is not mistaken for a result of the commit alone.

    Used by: `run_evaluation`.

    Returns:
        The commit hash and the changed flag, both None outside a Git repository.
    """
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None, None
    return commit, bool(status.strip())


def _fingerprint(texts: list[str]) -> str:
    """Return a hash that identifies a set of texts.

    Why: If the documents or questions change, scores are no longer comparable, and this shows it.

    Used by: `run_evaluation`.

    Args:
        texts: The texts.

    Returns:
        A SHA-256 hex digest.
    """
    return hashlib.sha256("\n\0".join(texts).encode()).hexdigest()


async def run_evaluation(
    settings: Settings,
    golden: GoldenSet,
    documents: list[Document],
    label: str,
    provider: LLMProvider | None = None,
    input_price_per_million: float | None = None,
    output_price_per_million: float | None = None,
) -> dict[str, Any]:
    """Run every case and build the report.

    Why: The mock only echoes the question, so scoring its answers would produce meaningless numbers
    that look real. Those metrics stay None and are reported as "not measured". Token counts from
    the mock are word counts, so they are only useful for comparing configurations.

    The report has an explicit list of settings, not a dump of `Settings`, so a new secret field can
    never end up in a file that goes into Git.

    Used by: `ai_ops.evaluation.__main__.main` and tests.

    Args:
        settings: The configuration under test.
        golden: The questions.
        documents: The knowledge base.
        label: A short name for this run.
        provider: Replaces the configured provider. Tests pass a fake.
        input_price_per_million: Price of one million input tokens, if known.
        output_price_per_million: Price of one million output tokens, if known.

    Returns:
        A JSON-ready report with `metadata`, `metrics`, and `cases`.

    Raises:
        LLMError: A model call failed.

    Side effects:
        Calls the model provider and runs `git` to read the current commit.
    """
    owns_provider = provider is None
    active = provider or build_provider(settings)
    scores_answers = settings.llm_provider != "mock"
    try:
        service = _build_service(settings, active)
        await service.ingest(documents)
        cases = [
            _score_case(case, await service.ask(case.question), scores_answers)
            for case in golden.cases
        ]
        injection = (
            await _measure_injection(settings, active, golden.injection_cases)
            if scores_answers
            else None
        )
    finally:
        if owns_provider:
            await active.aclose()

    answerable = [case for case in cases if case["answerable"]]
    unanswerable = [case for case in cases if not case["answerable"]]
    answered = [case for case in answerable if case["answer_fact_recall"] is not None]
    latencies = [case["latency_ms"] for case in cases]
    input_tokens = sum(case["input_tokens"] for case in cases)
    output_tokens = sum(case["output_tokens"] for case in cases)
    has_prices = input_price_per_million is not None and output_price_per_million is not None

    metrics = {
        "hit_rate": mean([float(case["hit"]) for case in answerable]),
        "precision": mean([case["precision"] for case in answerable]),
        "recall": mean([case["recall"] for case in answerable]),
        "context_fact_recall": mean([case["context_fact_recall"] for case in answerable]),
        "not_found_accuracy": mean([float(not case["found"]) for case in unanswerable]),
        "answer_fact_recall": mean([case["answer_fact_recall"] for case in answered]),
        "groundedness": mean([case["groundedness"] for case in answered]),
        "citation_rate": mean([float(case["citation"]) for case in answered]),
        "injection_resistance": injection,
        "latency_p50_ms": percentile(latencies, 50),
        "latency_p95_ms": percentile(latencies, 95),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "estimated_cost": (
            (input_tokens * input_price_per_million + output_tokens * output_price_per_million)
            / 1_000_000
            if has_prices
            else None
        ),
    }
    commit, changed = _git_state()
    metadata = {
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "label": label,
        "git_commit": commit,
        "git_tree_changed": changed,
        "provider": settings.llm_provider,
        "chat_model": settings.llm_chat_model,
        "embedding_model": settings.llm_embedding_model,
        "chunk_size": settings.rag_chunk_size,
        "chunk_overlap": settings.rag_chunk_overlap,
        "top_k": settings.rag_top_k,
        "min_score": settings.rag_min_score,
        "prompt_version": PROMPT_VERSION,
        "document_count": len(documents),
        "case_count": len(cases),
        "golden_sha256": _fingerprint([golden.model_dump_json()]),
        "knowledge_sha256": _fingerprint([f"{d.id}\n{d.text}" for d in documents]),
    }
    return {"metadata": metadata, "metrics": metrics, "cases": cases}
