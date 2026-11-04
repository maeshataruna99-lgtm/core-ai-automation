"""The interface and data types shared by all LLM providers.

Why: A `Protocol` describes what a provider must do without forcing inheritance. Any class
with these methods works, including small fakes in tests. This is Python's version of
structural typing, similar to how TypeScript interfaces work.

Used by: `ai_ops.llm.mock`, `ai_ops.llm.openai_compatible`, `ai_ops.llm.retry`,
`ai_ops.llm.factory`, and `ai_ops.api.errors`.
"""

from dataclasses import dataclass
from typing import Any, Literal, Protocol


@dataclass(frozen=True)
class ToolSchema:
    """Describes a tool that a model may ask to call.

    Why: The model only sees this description. It never runs anything itself: it can only ask,
    and our code decides what happens.

    Used by: `ai_ops.agents.registry.ToolRegistry` builds these. Providers send them.
    """

    name: str
    description: str
    parameters: dict[str, Any]


@dataclass(frozen=True)
class ToolCall:
    """A model's request to call a tool.

    Why: Vendors send arguments in their own format. This gives the rest of the code one type.
    Arguments come from the model, so treat them as untrusted input.

    Used by: `GenerationResult`, `ChatMessage`, and `ai_ops.agents.loop.Agent`.
    """

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ChatMessage:
    """One message in a conversation with a model.

    Why: Keeping the prompt split by role lets the system instructions stay separate from
    user input, which is part of the prompt injection defence. `tool_calls` and `tool_call_id`
    let the model see which tool results answer which of its own requests.

    Used by: callers of `LLMProvider.generate`.
    """

    role: Literal["system", "user", "assistant", "tool"]
    content: str
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None


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
    tool_calls: tuple[ToolCall, ...] = ()


@dataclass(frozen=True)
class EmbeddingResult:
    """The result of an embedding call.

    Why: Same reason as `GenerationResult`. `vectors` is in the same order as the input.

    Used by: `LLMProvider.embed`.
    """

    vectors: list[list[float]]
    model: str
    usage: Usage


@dataclass
class ModelSelection:
    """The model to use, which can change while the app runs.

    Why: The provider reads this for each request, so a change on the Settings page applies to the
    next
    request, with no need to rebuild the provider.

    Used by: `ai_ops.api.app.create_app`, `ai_ops.llm.openai_compatible`, and
    `ai_ops.services.llm_settings`.
    """

    current: str | None


@dataclass(frozen=True)
class ConnectionCheck:
    """The result of checking that a provider can be reached.

    Why: A failed check is the answer to the question, so it is data and not an exception.

    Used by: returned by `LLMProvider.check_connection`.
    """

    ok: bool
    detail: str
    models: tuple[str, ...] = ()


def describe_http_failure(status: int | None) -> str:
    """Say what a failure means, and what to do about it.

    Why: "HTTP 402" does not help a person. These follow DeepSeek's documented error codes, and
    they are
    fixed texts, so a response body that echoes a key can never reach the page.

    Used by: `ai_ops.llm.openai_compatible` and `ai_ops.services.llm_settings`.

    Args:
        status: The HTTP status, or None if there was no response.

    Returns:
        A message for a person.
    """
    if status is None:
        return "Could not reach the provider or it did not answer in time. Check your network."
    if status == 401:
        return "The API key was rejected. Check the key in your .env file, then restart the API."
    if status == 402:
        return "The account has no balance left. Top up with the provider, then try again."
    if status == 429:
        return "The rate limit was reached. Wait a moment, then try again."
    if status >= 500:
        return f"The provider has a problem on its side (HTTP {status}). Try again later."
    return f"The provider refused the request (HTTP {status})."


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

    def __init__(self, message: str, *, retryable: bool = False, status: int | None = None) -> None:
        """Store the message, whether a retry could help, and the HTTP status.

        Why: This overrides the class default, because retrying depends on the failure. The status
        lets
        the Settings page say what to do.

        Used by: `ai_ops.llm.openai_compatible`.

        Args:
            message: Safe description of the failure.
            retryable: Whether trying again could succeed.
            status: The HTTP status, if there was a response.
        """
        super().__init__(message)
        self.retryable = retryable
        self.status = status


class LLMProvider(Protocol):
    """What the application needs from any model provider.

    Why: Services depend on this interface, not on a vendor, so tests can use the mock and
    production can use a real API with the same service code.

    Used by: `ai_ops.llm.factory.build_provider` returns one. `create_app` stores it.
    """

    async def generate(
        self, messages: list[ChatMessage], tools: list[ToolSchema] | None = None
    ) -> GenerationResult:
        """Generate a reply to a conversation.

        Why: This is the core operation for answers, agents, and drafting.

        Used by: `ai_ops.services`, and `ai_ops.agents.loop.Agent` with tools.

        Args:
            messages: The conversation so far.
            tools: Tools the model may ask to call.

        Returns:
            The generated text, any tool calls the model asked for, and the token usage.

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

    async def check_connection(self) -> ConnectionCheck:
        """Check that the provider can be reached with the configured key.

        Why: The Settings page needs a real answer, and this must not cost tokens, since it runs
        each
        time that page opens.

        Used by: `ai_ops.services.llm_settings`.

        Returns:
            The result. Failures are in the result, and they are not raised.
        """
        ...

    async def aclose(self) -> None:
        """Release resources such as open HTTP connections.

        Why: Connections stay open for reuse, so something must close them at shutdown.

        Used by: the lifespan function in `ai_ops.api.app`.
        """
        ...
