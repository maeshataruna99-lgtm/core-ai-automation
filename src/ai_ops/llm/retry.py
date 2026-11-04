"""Retry wrapper that adds bounded retries to any LLM provider.

Why: A wrapper with the same interface as the provider (the decorator pattern) keeps retry
logic out of each provider, and it can be tested with a small fake.

Used by: `ai_ops.llm.factory.build_provider` wraps the real provider with it.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import TypeVar

from ai_ops.llm.provider import (
    ChatMessage,
    EmbeddingResult,
    GenerationResult,
    LLMError,
    LLMProvider,
)
from ai_ops.logging import LOGGER_NAME

logger = logging.getLogger(f"{LOGGER_NAME}.llm")

T = TypeVar("T")

BASE_DELAY_SECONDS = 0.5


class RetryingProvider:
    """Implementation of `LLMProvider` that retries temporary failures of another provider.

    Why: Temporary failures such as rate limits are normal with remote APIs. The worst-case
    wait is about `(max_retries + 1) * timeout` plus the delays, so keep `max_retries` small.

    Used by: `ai_ops.llm.factory.build_provider` and `tests/test_llm_openai_and_retry.py`.
    """

    def __init__(
        self,
        inner: LLMProvider,
        max_retries: int,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        """Store the wrapped provider and the retry limit.

        Why: `sleep` is a parameter so tests can run instantly and check the delays.

        Used by: `ai_ops.llm.factory.build_provider` and tests.

        Args:
            inner: The provider to call.
            max_retries: Retries allowed after the first attempt.
            sleep: Function used to wait between attempts.
        """
        self._inner = inner
        self._max_retries = max_retries
        self._sleep = sleep

    async def _with_retries(self, call: Callable[[], Awaitable[T]]) -> T:
        """Run a call, retrying while the error is retryable and attempts remain.

        Why: The delay doubles each time, which gives an overloaded provider room to recover.

        shortcut: no jitter and `Retry-After` is ignored, add both when many clients share a
        rate limit.

        Used by: `generate` and `embed`.

        Args:
            call: The provider call to run.

        Returns:
            The result of the first successful attempt.

        Raises:
            LLMError: The error is not retryable, or the attempts ran out.

        Side effects:
            Waits between attempts and logs a warning for each retry.
        """
        for attempt in range(self._max_retries):
            try:
                return await call()
            except LLMError as exc:
                if not exc.retryable:
                    raise
                delay = BASE_DELAY_SECONDS * 2**attempt
                logger.warning(
                    "llm call failed, retrying",
                    extra={"attempt": attempt + 1, "delay_s": delay, "error": str(exc)},
                )
                await self._sleep(delay)
        return await call()

    async def generate(self, messages: list[ChatMessage]) -> GenerationResult:
        """Call the wrapped provider's `generate` with retries.

        Why: Callers use the same interface with or without retries.

        Used by: services, through the `LLMProvider` interface.

        Args:
            messages: The conversation so far.

        Returns:
            The wrapped provider's result.

        Raises:
            LLMError: The call failed after any retries.
        """
        return await self._with_retries(lambda: self._inner.generate(messages))

    async def embed(self, texts: list[str]) -> EmbeddingResult:
        """Call the wrapped provider's `embed` with retries.

        Why: Callers use the same interface with or without retries.

        Used by: retrieval code, through the `LLMProvider` interface.

        Args:
            texts: The texts to embed.

        Returns:
            The wrapped provider's result.

        Raises:
            LLMError: The call failed after any retries.
        """
        return await self._with_retries(lambda: self._inner.embed(texts))

    async def aclose(self) -> None:
        """Close the wrapped provider.

        Why: The wrapper owns the provider it wraps, so closing passes through.

        Used by: the lifespan function in `ai_ops.api.app`.
        """
        await self._inner.aclose()
