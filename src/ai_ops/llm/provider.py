"""The interface and data types shared by all LLM providers.

Why: A `Protocol` describes what a provider must do without forcing inheritance. Any class
with these methods works, including small fakes in tests. This is Python's version of
structural typing, similar to how TypeScript interfaces work.

Used by: `ai_ops.llm.mock`, `ai_ops.llm.openai_compatible`, `ai_ops.llm.retry`,
`ai_ops.llm.factory`, and `ai_ops.api.errors`.
"""

from dataclasses import dataclass
from typing import Literal, Protocol


@dataclass(frozen=True)
class ChatMessage:
    """One message in a conversation with a model.

    Why: Keeping the prompt split by role lets the system instructions stay separate from
    user input, which is part of the prompt injection defence.

    Used by: callers of `LLMProvider.generate`.
    """

    role: Literal["system", "user", "assistant"]
    content: str


@dataclass(frozen=True)
class Usage:
    """Token counts for one call.

    Why: Tokens are what providers charge for. Milestone 6 turns these counts into cost.

    Used by: `GenerationResult` and `EmbeddingResult`.
    """

    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class GenerationResult:
    """The result of a text generation call.

    Why: A typed result hides the vendor's response format from the rest of the code.

    Used by: `LLMProvider.generate`.
    """

    text: str
    model: str
    usage: Usage


@dataclass(frozen=True)
class EmbeddingResult:
    """The result of an embedding call.

    Why: Same reason as `GenerationResult`. `vectors` is in the same order as the input.

    Used by: `LLMProvider.embed`.
    """

    vectors: list[list[float]]
    model: str
    usage: Usage


class LLMError(Exception):
    """Base class for failures of an LLM call.

    Why: Callers can handle every provider failure in one place. Messages must be safe to
    log: never put an API key, a prompt, or a raw provider response in them.

    Used by: `ai_ops.llm.retry` reads `retryable`. `ai_ops.api.errors` maps it to HTTP.
    """

    retryable = False


class LLMTimeoutError(LLMError):
    """The provider did not answer in time.

    Why: A timeout needs a different HTTP status (504) from a failure (502). It is retryable
    because the next attempt may succeed.

    Used by: raised in `ai_ops.llm.openai_compatible`, mapped in `ai_ops.api.errors`.
    """

    retryable = True


class LLMProviderError(LLMError):
    """The provider failed or returned an unusable response.

    Why: Some failures are temporary (a 503), others will never succeed (a 401), so the
    raiser decides whether a retry makes sense.

    Used by: raised in `ai_ops.llm.openai_compatible`, mapped in `ai_ops.api.errors`.
    """

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        """Store the message and whether a retry could help.

        Why: This overrides the class default, because retrying depends on the failure.

        Used by: `ai_ops.llm.openai_compatible`.

        Args:
            message: Safe description of the failure.
            retryable: Whether trying again could succeed.
        """
        super().__init__(message)
        self.retryable = retryable


class LLMProvider(Protocol):
    """What the application needs from any model provider.

    Why: Services depend on this interface, not on a vendor, so tests can use the mock and
    production can use a real API with the same service code.

    Used by: `ai_ops.llm.factory.build_provider` returns one. `create_app` stores it.
    """

    async def generate(self, messages: list[ChatMessage]) -> GenerationResult:
        """Generate a reply to a conversation.

        Why: This is the core operation for answers, agents, and drafting.

        Used by: services, from Milestone 3 onwards.

        Args:
            messages: The conversation so far.

        Returns:
            The generated text and its token usage.

        Raises:
            LLMError: The call failed.
        """
        ...

    async def embed(self, texts: list[str]) -> EmbeddingResult:
        """Convert texts to vectors for similarity search.

        Why: Retrieval compares these vectors to find relevant document chunks.

        Used by: the retrieval code, from Milestone 3 onwards.

        Args:
            texts: The texts to embed.

        Returns:
            One vector per input text, in the same order.

        Raises:
            LLMError: The call failed.
        """
        ...

    async def aclose(self) -> None:
        """Release resources such as open HTTP connections.

        Why: Connections stay open for reuse, so something must close them at shutdown.

        Used by: the lifespan function in `ai_ops.api.app`.
        """
        ...
