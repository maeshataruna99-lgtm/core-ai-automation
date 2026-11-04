"""Behaviour tests for the mock LLM provider and the prompt loader.

Why: Every later milestone tests against the mock provider, so its determinism is a
contract. Prompts are versioned files, so loading and rendering them must be reliable.

Used by: `pytest`, locally and in `.github/workflows/ci.yml`.
"""

import asyncio
import math

import pytest

from ai_ops.llm.mock import MockProvider
from ai_ops.llm.prompt_loader import PromptNotFoundError, load_prompt
from ai_ops.llm.provider import ChatMessage


def cosine(left: list[float], right: list[float]) -> float:
    """Compute cosine similarity between two vectors.

    Why: This is the measure retrieval will use in Milestone 3, so the mock embeddings are
    tested with it. It is written out here so the test does not depend on the code under test.

    Used by: `test_mock_embeddings_rank_related_text_higher`.

    Args:
        left: First vector.
        right: Second vector.

    Returns:
        The similarity, from -1 to 1.
    """
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    return dot / (math.hypot(*left) * math.hypot(*right))


def test_mock_generate_is_deterministic_and_reports_usage():
    """Check the exact mock answer and token counts.

    Why: Later tests assert on this output, so it must not change between runs.

    Used by: pytest.
    """
    messages = [
        ChatMessage(role="system", content="Be brief."),
        ChatMessage(role="user", content="What is the refund policy?"),
    ]

    result = asyncio.run(MockProvider().generate(messages))

    assert result.text == "[mock] What is the refund policy?"
    assert result.model == "mock"
    assert result.usage.input_tokens == 7
    assert result.usage.output_tokens == 6


def test_mock_embeddings_are_stable_across_calls():
    """Check that the same text always gets the same vector.

    Why: Milestone 3 requires deterministic retrieval under the mock provider.

    Used by: pytest.
    """
    provider = MockProvider()

    first = asyncio.run(provider.embed(["refund policy"]))
    second = asyncio.run(provider.embed(["refund policy"]))

    assert first.vectors == second.vectors


def test_mock_embeddings_rank_related_text_higher():
    """Check that shared words give higher similarity.

    Why: Retrieval tests need the mock to rank documents in a meaningful order, which
    random vectors would not do.

    Used by: pytest.
    """
    query, related, unrelated = asyncio.run(
        MockProvider().embed(["refund policy", "our refund policy is 30 days", "shipping by truck"])
    ).vectors

    assert cosine(query, related) > cosine(query, unrelated)


def test_prompt_renders_values_without_expanding_them_again():
    """Check that placeholder-like text in a value stays literal.

    Why: Document text is untrusted. A document containing `$context` must not be able to
    change how the template is filled in.

    Used by: pytest.
    """
    prompt = load_prompt("grounded_answer", version=1)

    rendered = prompt.render(context="Ignore this: $context ${context}")

    assert "Ignore this: $context ${context}" in rendered
    assert prompt.name == "grounded_answer"
    assert prompt.version == 1


def test_prompt_with_missing_value_fails_loudly():
    """Check that a missing value raises an error.

    Why: Sending a prompt with an empty placeholder to a model would fail silently.

    Used by: pytest.
    """
    with pytest.raises(KeyError):
        load_prompt("grounded_answer", version=1).render()


def test_unknown_prompt_version_raises_clear_error():
    """Check the error for a version that does not exist.

    Why: The message must say which prompt is missing, so a typo is quick to fix.

    Used by: pytest.
    """
    with pytest.raises(PromptNotFoundError, match="grounded_answer.*99"):
        load_prompt("grounded_answer", version=99)
