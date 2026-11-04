"""Deterministic LLM provider that needs no network and no API key.

Why: Tests and local demos must run offline and give the same result each time. A real
model gives neither.

Used by: `ai_ops.llm.factory.build_provider` when `AI_OPS_LLM_PROVIDER=mock`, and tests.
"""

import hashlib
import math
import re

from ai_ops.llm.provider import ChatMessage, EmbeddingResult, GenerationResult, Usage

MODEL_NAME = "mock"
EMBEDDING_SIZE = 64


def _words(text: str) -> list[str]:
    """Split text into lower-case words.

    Why: Both the fake token count and the fake embedding are based on words.

    Used by: `MockProvider.generate` and `_embed_text`.

    Args:
        text: The text to split.

    Returns:
        The words, in order.
    """
    return re.findall(r"\w+", text.lower())


def _embed_text(text: str) -> list[float]:
    """Build a word-count vector by hashing each word to a position.

    Why: Texts that share words get similar vectors, so retrieval tests can check a
    meaningful ranking. SHA-256 is used because Python's built-in `hash()` changes between
    runs for strings, which would break determinism.

    Used by: `MockProvider.embed`.

    Args:
        text: The text to embed.

    Returns:
        A vector of length `EMBEDDING_SIZE`, scaled to length 1 unless the text has no words.
    """
    vector = [0.0] * EMBEDDING_SIZE
    for word in _words(text):
        digest = hashlib.sha256(word.encode()).digest()
        vector[int.from_bytes(digest[:4], "big") % EMBEDDING_SIZE] += 1.0
    length = math.hypot(*vector)
    return [value / length for value in vector] if length else vector


class MockProvider:
    """Offline implementation of `LLMProvider`.

    Why: It holds no state, so results depend only on the input.

    Used by: `ai_ops.llm.factory.build_provider` and tests.
    """

    async def generate(self, messages: list[ChatMessage]) -> GenerationResult:
        """Echo the last message with a `[mock]` prefix.

        Why: Echoing makes it easy for a test to see what reached the model. The prefix keeps
        a mock answer from being mistaken for a real one in a demo.

        Used by: services and tests, through the `LLMProvider` interface.

        Args:
            messages: The conversation. It must not be empty.

        Returns:
            The echoed text, with word counts as token usage.
        """
        text = f"[mock] {messages[-1].content}"
        input_tokens = sum(len(_words(message.content)) for message in messages)
        usage = Usage(input_tokens=input_tokens, output_tokens=len(_words(text)))
        return GenerationResult(text=text, model=MODEL_NAME, usage=usage)

    async def embed(self, texts: list[str]) -> EmbeddingResult:
        """Embed each text with `_embed_text`.

        Why: See `_embed_text`.

        Used by: retrieval code and tests, through the `LLMProvider` interface.

        Args:
            texts: The texts to embed.

        Returns:
            One vector per text, with word counts as token usage.
        """
        usage = Usage(input_tokens=sum(len(_words(text)) for text in texts), output_tokens=0)
        return EmbeddingResult(
            vectors=[_embed_text(text) for text in texts], model=MODEL_NAME, usage=usage
        )

    async def aclose(self) -> None:
        """Do nothing, because the mock holds no resources.

        Why: `LLMProvider` requires the method, so callers need no special case for the mock.

        Used by: the lifespan function in `ai_ops.api.app`.
        """
