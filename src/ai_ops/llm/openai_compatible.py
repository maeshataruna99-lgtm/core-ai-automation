"""LLM provider for any API that follows the OpenAI chat and embeddings format.

Why: Many vendors and local servers accept this format, so one class covers them through
configuration. Plain HTTP is used instead of a vendor SDK, which keeps the dependency list
short and shows what actually goes over the wire.

Used by: `ai_ops.llm.factory.build_provider` when `AI_OPS_LLM_PROVIDER=openai_compatible`.
"""

from typing import Any

import httpx2

from ai_ops.llm.provider import (
    ChatMessage,
    EmbeddingResult,
    GenerationResult,
    LLMProviderError,
    LLMTimeoutError,
    Usage,
)

# 408 and 429 mean "try later". Server errors are usually temporary.
_RETRYABLE_STATUSES = {408, 429}


class OpenAICompatibleProvider:
    """Implementation of `LLMProvider` over HTTP.

    Why: One `AsyncClient` is kept for the life of the app, because reusing its connections
    is much faster than opening a new one per call. It must be closed with `aclose`.

    Used by: `ai_ops.llm.factory.build_provider` and `tests/test_llm_openai_and_retry.py`.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        chat_model: str,
        embedding_model: str,
        timeout_seconds: float,
        transport: httpx2.AsyncBaseTransport | None = None,
    ) -> None:
        """Create the HTTP client.

        Why: The key goes in a default header, so no other code handles it after this point.

        Used by: `ai_ops.llm.factory.build_provider` and tests.

        Args:
            base_url: API root, such as `https://api.example.com/v1`.
            api_key: Secret key, sent as a bearer token.
            chat_model: Model name for `generate`.
            embedding_model: Model name for `embed`.
            timeout_seconds: Limit for each network step of one attempt.
            transport: Replaces the network. Tests pass an in-memory transport.
        """
        self._chat_model = chat_model
        self._embedding_model = embedding_model
        self._client = httpx2.AsyncClient(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout_seconds,
            transport=transport,
        )

    async def _post(self, path: str, payload: dict[str, Any]) -> Any:
        """Send one request and translate every failure to an `LLMError`.

        Why: This is the only place that knows about HTTP errors, so callers only deal with
        our own exception types. The response body is never copied into an error message,
        because real APIs can echo part of a rejected key in it.

        Used by: `generate` and `embed`.

        Args:
            path: Endpoint path, relative to the base URL.
            payload: The JSON body.

        Returns:
            The parsed JSON response.

        Raises:
            LLMTimeoutError: The provider did not answer in time.
            LLMProviderError: A connection failure, an error status, or invalid JSON.

        Side effects:
            Sends a network request.
        """
        try:
            response = await self._client.post(path, json=payload)
        except httpx2.TimeoutException as exc:
            raise LLMTimeoutError(f"Timeout calling {path}") from exc
        except httpx2.TransportError as exc:
            message = f"Connection failure calling {path}: {type(exc).__name__}"
            raise LLMProviderError(message, retryable=True) from exc

        status = response.status_code
        if status >= 400:
            retryable = status in _RETRYABLE_STATUSES or status >= 500
            raise LLMProviderError(f"HTTP {status} from {path}", retryable=retryable)
        try:
            return response.json()
        except ValueError:
            raise LLMProviderError(f"Invalid JSON from {path}") from None

    async def generate(self, messages: list[ChatMessage]) -> GenerationResult:
        """Call the chat completions endpoint.

        Why: This converts between the vendor format and our types, so the rest of the code
        never sees the vendor format.

        Used by: services, through the `LLMProvider` interface.

        Args:
            messages: The conversation so far.

        Returns:
            The first choice's text and the token usage.

        Raises:
            LLMTimeoutError: The provider did not answer in time.
            LLMProviderError: The call failed or the response has an unexpected shape.
        """
        data = await self._post(
            "/chat/completions",
            {
                "model": self._chat_model,
                "messages": [{"role": m.role, "content": m.content} for m in messages],
            },
        )
        try:
            return GenerationResult(
                text=data["choices"][0]["message"]["content"],
                model=data["model"],
                usage=Usage(
                    input_tokens=data["usage"]["prompt_tokens"],
                    output_tokens=data["usage"]["completion_tokens"],
                ),
            )
        except (KeyError, IndexError, TypeError):
            raise LLMProviderError("Unexpected response from /chat/completions") from None

    async def embed(self, texts: list[str]) -> EmbeddingResult:
        """Call the embeddings endpoint.

        Why: Results are sorted by `index`, because the API does not promise their order and
        a wrong order would attach vectors to the wrong chunks.

        Used by: retrieval code, through the `LLMProvider` interface.

        Args:
            texts: The texts to embed.

        Returns:
            One vector per text, in input order.

        Raises:
            LLMTimeoutError: The provider did not answer in time.
            LLMProviderError: The call failed or the response has an unexpected shape.
        """
        data = await self._post(
            "/embeddings",
            {"model": self._embedding_model, "input": texts},
        )
        try:
            items = sorted(data["data"], key=lambda item: item["index"])
            return EmbeddingResult(
                vectors=[item["embedding"] for item in items],
                model=data["model"],
                usage=Usage(input_tokens=data["usage"]["prompt_tokens"], output_tokens=0),
            )
        except (KeyError, TypeError):
            raise LLMProviderError("Unexpected response from /embeddings") from None

    async def aclose(self) -> None:
        """Close the HTTP client and its open connections.

        Why: Open connections would otherwise leak until the process exits.

        Used by: the lifespan function in `ai_ops.api.app`, through `RetryingProvider`.
        """
        await self._client.aclose()
