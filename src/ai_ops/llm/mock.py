"""Deterministic LLM provider that needs no network and no API key.

Why: Tests and local demos must run offline and give the same result each time. A real
model gives neither.

Used by: `ai_ops.llm.factory.build_provider` when `AI_OPS_LLM_PROVIDER=mock`, and tests.
"""

import hashlib
import math
import re

from ai_ops.llm.provider import (
    ChatMessage,
    ConnectionCheck,
    EmbeddingResult,
    GenerationResult,
    ToolCall,
    ToolSchema,
    Usage,
)

MODEL_NAME = "mock"
# Large enough that two different words rarely land on the same position.
EMBEDDING_SIZE = 256
# These words appear everywhere, so sharing them says nothing about the meaning.
STOPWORDS = frozenset(
    "a about all am an and any are as at be by can could did do does for from had has have how i "
    "if in is it me my of on or our that the their them they this to us was we were what when "
    "where which who why will with would you your".split()
)


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
    runs for strings, which would break determinism. This only matches exact words, not
    meaning: "refund" does not match "refunds", which a real embedding model would handle.

    Used by: `MockProvider.embed`.

    Args:
        text: The text to embed.

    Returns:
        A vector of length `EMBEDDING_SIZE`, scaled to length 1. It is all zeros if the text
        has only stopwords.
    """
    vector = [0.0] * EMBEDDING_SIZE
    for word in _words(text):
        if word in STOPWORDS:
            continue
        digest = hashlib.sha256(word.encode()).digest()
        vector[int.from_bytes(digest[:4], "big") % EMBEDDING_SIZE] += 1.0
    length = math.hypot(*vector)
    return [value / length for value in vector] if length else vector


def _choose_tool_call(text: str, tool_names: set[str]) -> ToolCall | None:
    """Pick a tool with simple keyword rules.

    Why: The offline demo needs something that behaves like a model calling tools. These rules
    are only for that demo: choosing tools well is what a real model is for.

    Used by: `MockProvider.generate`.

    Args:
        text: The user's message.
        tool_names: The tools on offer.

    Returns:
        A tool call, or None if no offered tool fits.
    """
    record = re.search(r"ORD-\d+", text, re.IGNORECASE)
    status = re.search(r"\b(processing|shipped|delivered|cancelled)\b", text, re.IGNORECASE)
    wants_change = re.search(r"\b(update|change|set|mark)\b", text, re.IGNORECASE)
    if record and status and wants_change and "update_record" in tool_names:
        arguments = {"record_id": record.group().upper(), "status": status.group().lower()}
        return ToolCall(id="mock-1", name="update_record", arguments=arguments)
    if record and "lookup_record" in tool_names:
        arguments = {"record_id": record.group().upper()}
        return ToolCall(id="mock-1", name="lookup_record", arguments=arguments)
    if "search_knowledge" in tool_names:
        return ToolCall(id="mock-1", name="search_knowledge", arguments={"query": text})
    return None


class MockProvider:
    """Offline implementation of `LLMProvider`.

    Why: It holds no state, so results depend only on the input.

    Used by: `ai_ops.llm.factory.build_provider` and tests.
    """

    async def generate(
        self, messages: list[ChatMessage], tools: list[ToolSchema] | None = None
    ) -> GenerationResult:
        """Echo the last line of the last message, or choose a tool when tools are offered.

        Why: Echoing makes it easy for a test to see what reached the model. The prefix keeps
        a mock answer from being mistaken for a real one in a demo. Only the last line is
        used because the RAG prompt ends with the question, so the context is not repeated.
        The mock does not answer anything: real answers need a real provider.

        Used by: services, the agent, and tests, through the `LLMProvider` interface.

        Args:
            messages: The conversation. It must not be empty.
            tools: Tools the mock may choose from. It only asks for a tool when the last
                message came from the user, so a tool result ends the agent's loop.

        Returns:
            The echoed text or a tool call, with word counts as token usage.
        """
        last = messages[-1]
        input_tokens = sum(len(_words(message.content)) for message in messages)
        if tools and last.role == "user":
            call = _choose_tool_call(last.content, {tool.name for tool in tools})
            if call is not None:
                return GenerationResult(
                    text="",
                    model=MODEL_NAME,
                    usage=Usage(input_tokens=input_tokens, output_tokens=0),
                    tool_calls=(call,),
                )
        text = f"[mock] {last.content.splitlines()[-1]}"
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

    async def check_connection(self) -> ConnectionCheck:
        """Report that there is nothing to connect to.

        Why: The result must not look like a tested connection to a real service.

        Used by: `ai_ops.services.llm_settings`, through the `LLMProvider` interface.

        Returns:
            A successful result that says it is the mock.
        """
        return ConnectionCheck(
            ok=True, detail="Mock provider: it runs offline, with no connection."
        )

    async def aclose(self) -> None:
        """Do nothing, because the mock holds no resources.

        Why: `LLMProvider` requires the method, so callers need no special case for the mock.

        Used by: the lifespan function in `ai_ops.api.app`.
        """
