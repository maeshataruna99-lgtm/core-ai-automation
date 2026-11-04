"""Metric formulas.

Why: These are pure functions, so each one can be checked against a value worked out by hand. That
matters because a wrong metric gives false confidence.

Used by: `ai_ops.evaluation.runner` and `tests/test_evaluation.py`.
"""

import math
import re
from collections.abc import Iterable, Mapping, Sequence

# Short words are mostly grammar, so they appear in any text and say nothing about the facts.
_MIN_CONTENT_WORD_LENGTH = 4


def precision(retrieved: Sequence[str], relevant: Iterable[str]) -> float:
    """Return the share of retrieved items that are relevant.

    Why: Low precision means the model is given irrelevant text, which costs tokens and can mislead
    it.

    Used by: `ai_ops.evaluation.runner`.

    Args:
        retrieved: What the system found.
        relevant: What it should have found.

    Returns:
        A value from 0 to 1. It is 0 if nothing was retrieved.
    """
    if not retrieved:
        return 0.0
    return len(set(retrieved) & set(relevant)) / len(set(retrieved))


def recall(retrieved: Sequence[str], relevant: Iterable[str]) -> float:
    """Return the share of relevant items that were retrieved.

    Why: Low recall means the model never sees the text it needs, so it cannot answer well.

    Used by: `ai_ops.evaluation.runner`.

    Args:
        retrieved: What the system found.
        relevant: What it should have found.

    Returns:
        A value from 0 to 1. It is 0 if nothing is relevant.
    """
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    return len(set(retrieved) & relevant_set) / len(relevant_set)


def fact_recall(facts: Sequence[str], text: str) -> float:
    """Return the share of expected facts that appear in a text.

    Why: Exact phrases are simple and repeatable, and need no second model to judge the answer. The
    trade-off is that a correct answer in other words counts as a miss, so scores can be too low.

    Used by: `ai_ops.evaluation.runner`, for both the retrieved context and the model's answer.

    Args:
        facts: The expected phrases.
        text: The text to search, ignoring case.

    Returns:
        A value from 0 to 1. It is 1 if no facts are expected.
    """
    if not facts:
        return 1.0
    folded = text.casefold()
    return sum(fact.casefold() in folded for fact in facts) / len(facts)


def _content_words(text: str) -> set[str]:
    """Return the distinct longer words in a text.

    Why: Comparing sets means a repeated word does not count twice.

    Used by: `groundedness`.

    Args:
        text: The text.

    Returns:
        The lower-case words with at least `_MIN_CONTENT_WORD_LENGTH` characters.
    """
    words = re.findall(r"\w+", text.casefold())
    return {word for word in words if len(word) >= _MIN_CONTENT_WORD_LENGTH}


def groundedness(answer: str, context: str) -> float:
    """Return the share of an answer's content words that appear in the context.

    Why: Words from outside the context may be invented, so a low score is a warning. This is a
    rough check: it cannot see a false statement made from words that are in the context.

    Used by: `ai_ops.evaluation.runner`.

    Args:
        answer: The model's answer.
        context: The retrieved text.

    Returns:
        A value from 0 to 1. It is 1 if the answer has no content words.
    """
    answer_words = _content_words(answer)
    if not answer_words:
        return 1.0
    return len(answer_words & _content_words(context)) / len(answer_words)


def citation_found(answer: str, source_ids: Iterable[str]) -> bool:
    """Check whether an answer names a retrieved source.

    Why: The prompt asks the model to cite sources, and this checks that it does. It does not check
    that the citation supports the claim.

    Used by: `ai_ops.evaluation.runner`.

    Args:
        answer: The model's answer.
        source_ids: The ids of the retrieved chunks.

    Returns:
        True if any id appears in the answer.
    """
    return any(source_id in answer for source_id in source_ids)


def mean(values: Sequence[float]) -> float | None:
    """Return the average.

    Why: None means "not measured", which must stay different from a score of 0.

    Used by: `ai_ops.evaluation.runner`.

    Args:
        values: The values.

    Returns:
        The average, or None if there are no values.
    """
    return sum(values) / len(values) if values else None


def percentile(values: Sequence[float], percent: float) -> float | None:
    """Return a percentile using the nearest-rank method.

    Why: An average hides slow requests. p95 shows what the slowest requests feel like. Nearest rank
    always returns a value that was really observed.

    Used by: `ai_ops.evaluation.runner`, for latency.

    Args:
        values: The values.
        percent: The percentile, from 0 to 100.

    Returns:
        The percentile, or None if there are no values.
    """
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(math.ceil(percent / 100 * len(ordered)) - 1, 0)]


def check_thresholds(
    metrics: Mapping[str, float | None], thresholds: Mapping[str, float]
) -> list[str]:
    """Compare metrics with their minimums.

    Why: This is the regression gate. A metric that was not measured passes, or the mock could never
    run. An unknown name fails, because a typo would otherwise silently remove a check.

    Used by: `ai_ops.evaluation.__main__.main`.

    Args:
        metrics: The measured values. None means not measured.
        thresholds: The minimum for each metric.

    Returns:
        A description of each failure. An empty list means the run passed.
    """
    failures = []
    for name, minimum in thresholds.items():
        if name not in metrics:
            failures.append(f"{name}: unknown metric")
        elif metrics[name] is not None and metrics[name] < minimum:
            failures.append(f"{name}: {metrics[name]:.3f} is below the minimum of {minimum:.3f}")
    return failures
