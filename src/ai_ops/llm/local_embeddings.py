"""A provider that sends chat to a remote model, and computes embeddings locally.

Why: The DeepSeek API documentation lists no embeddings endpoint, but retrieval needs embeddings.
This pairs its chat with the mock provider's word-hash embeddings. They run offline at no cost, and
only match exact words, so retrieval scores stay the same as with the mock provider.

Used by: `ai_ops.llm.factory.build_provider` when `AI_OPS_LLM_PROVIDER=deepseek`.
"""

from ai_ops.llm.mock import MockProvider
from ai_ops.llm.provider import (
    ChatMessage,
    ConnectionCheck,
    EmbeddingResult,
    GenerationResult,
    LLMProvider,
    ToolSchema,
)


class LocalEmbeddingProvider:
    """Implementation of `LLMProvider` that splits the two jobs.

    Why: Services ask one provider for both, so they need no change. A document's text does not
    leave this computer for embedding, only the chunks that a question retrieves.

    Used by: `ai_ops.llm.factory.build_provider`.
    """

    def __init__(self, chat: LLMProvider) -> None:
        """Store the provider that handles chat.

        Why: Composition keeps each provider unaware of the other.

        Used by: `ai_ops.llm.factory.build_provider`.

        Args:
            chat: The provider for `generate`.
        """
        self._chat = chat
        self._embedder = MockProvider()

    async def generate(
        self, messages: list[ChatMessage], tools: list[ToolSchema] | None = None
    ) -> GenerationResult:
        """Generate a reply with the remote model.

        Why: See the module docstring.

        Used by: services and the agent, through the `LLMProvider` interface.

        Args:
            messages: The conversation so far.
            tools: Tools the model may ask to call.

        Returns:
            The remote model's result.

        Raises:
            LLMError: The call failed.
        """
        return await self._chat.generate(messages, tools)

    async def embed(self, texts: list[str]) -> EmbeddingResult:
        """Embed texts locally, with no network call.

        Why: See the module docstring.

        Used by: retrieval code, through the `LLMProvider` interface.

        Args:
            texts: The texts to embed.

        Returns:
            One vector per text.
        """
        return await self._embedder.embed(texts)

    async def check_connection(self) -> ConnectionCheck:
        """Check the chat provider.

        Why: Embeddings run locally, so chat is the only part that can fail to connect.

        Used by: `ai_ops.services.llm_settings`, through the `LLMProvider` interface.

        Returns:
            The chat provider's result.
        """
        return await self._chat.check_connection()

    async def aclose(self) -> None:
        """Close the chat provider.

        Why: Only the chat provider holds open connections.

        Used by: the lifespan function in `ai_ops.api.app`, through `RetryingProvider`.
        """
        await self._chat.aclose()
