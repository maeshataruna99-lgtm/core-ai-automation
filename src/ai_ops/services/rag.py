"""Retrieval-Augmented Generation: answer questions from the knowledge base.

Why: A model alone can invent facts. RAG finds relevant text first and asks the model to
answer only from it, so answers can be checked against their sources.

Used by: `ai_ops.api.app` creates the service. `ai_ops.api.routes.ask` calls it.
"""

import html
import time
from collections.abc import Iterable
from dataclasses import dataclass

from ai_ops.llm.prompt_loader import load_prompt
from ai_ops.llm.provider import ChatMessage, LLMProvider, Usage
from ai_ops.retrieval.chunking import chunk_text
from ai_ops.retrieval.documents import Document
from ai_ops.retrieval.vector_store import Chunk, ScoredChunk, VectorStore

NOT_FOUND_ANSWER = "I could not find this in the knowledge base."
PROMPT_VERSION = 1
# Real embedding APIs limit how many inputs one request may contain.
EMBEDDING_BATCH_SIZE = 64


@dataclass(frozen=True)
class RagAnswer:
    """The result of answering a question.

    Why: `found` lets clients tell "no answer" apart from an answer, without reading the text.

    Used by: returned by `RagService.ask`, read by `ai_ops.api.routes.ask`.
    """

    answer: str
    found: bool
    sources: list[ScoredChunk]
    usage: Usage
    latency_ms: float


def _format_context(results: Iterable[ScoredChunk]) -> str:
    """Format retrieved chunks for the prompt.

    Why: Each chunk is labelled with its id, so the model can cite it. HTML escaping stops a
    document from closing the `<context>` tag and placing its text outside the data section.

    Used by: `RagService.ask`.

    Args:
        results: The retrieved chunks.

    Returns:
        The context text.
    """
    return "\n\n".join(
        f"[{html.escape(result.chunk.source_id)}]\n{html.escape(result.chunk.text, quote=False)}"
        for result in results
    )


class RagService:
    """Ingest documents and answer questions from them.

    Why: It depends only on the `LLMProvider` and `VectorStore` interfaces, so either one can
    be replaced without changing this class.

    Used by: `ai_ops.api.app` and `tests/test_rag.py`.
    """

    def __init__(
        self,
        provider: LLMProvider,
        store: VectorStore,
        chunk_size: int,
        chunk_overlap: int,
        top_k: int,
        min_score: float,
    ) -> None:
        """Store the collaborators and load the prompts.

        Why: Loading prompts here makes a missing prompt file fail at startup, not on the
        first question.

        Used by: `ai_ops.api.app.create_app` and tests.

        Args:
            provider: The model provider.
            store: The vector store.
            chunk_size: Maximum characters per chunk.
            chunk_overlap: Characters repeated between chunks.
            top_k: Maximum chunks used per question.
            min_score: Lowest similarity counted as relevant. It depends on the embedding
                model, so it needs checking when the model changes.

        Raises:
            PromptNotFoundError: A prompt file is missing.
        """
        self._provider = provider
        self._store = store
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._top_k = top_k
        self._min_score = min_score
        self._system_prompt = load_prompt("grounded_answer_system", PROMPT_VERSION)
        self._user_prompt = load_prompt("grounded_answer_user", PROMPT_VERSION)
        self.document_count = 0

    async def ingest(self, documents: list[Document]) -> None:
        """Chunk and embed documents, then add them to the store.

        Why: This runs once at startup. With a real provider, each restart pays to embed all
        documents again, which is acceptable for a small knowledge base.

        Used by: the lifespan function in `ai_ops.api.app`, and tests.

        Args:
            documents: The documents to add.

        Raises:
            LLMError: An embedding call failed.

        Side effects:
            Calls the provider and changes the store.
        """
        chunks = [
            Chunk(document_id=document.id, position=position, text=text)
            for document in documents
            for position, text in enumerate(
                chunk_text(document.text, self._chunk_size, self._chunk_overlap)
            )
        ]
        for start in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
            batch = chunks[start : start + EMBEDDING_BATCH_SIZE]
            result = await self._provider.embed([chunk.text for chunk in batch])
            self._store.add(batch, result.vectors)
        self.document_count += len(documents)

    async def ask(self, question: str) -> RagAnswer:
        """Answer a question from the knowledge base.

        Why: When nothing relevant is found, the model is not called at all, so an invented
        answer is impossible on that path and the call costs less. When the model is called,
        rules go in the system message and document text only in the user message, so
        documents do not get the authority of instructions.

        Used by: `ai_ops.api.routes.ask`.

        Args:
            question: The question.

        Returns:
            The answer, its sources, and the token usage and latency of this call.

        Raises:
            LLMError: A model call failed.
        """
        started_at = time.perf_counter()
        embedded = await self._provider.embed([question])
        results = [
            result
            for result in self._store.search(embedded.vectors[0], self._top_k)
            if result.score >= self._min_score
        ]

        def elapsed_ms() -> float:
            """Return the time since the call started."""
            return round((time.perf_counter() - started_at) * 1000, 1)

        if not results:
            return RagAnswer(
                answer=NOT_FOUND_ANSWER,
                found=False,
                sources=[],
                usage=embedded.usage,
                latency_ms=elapsed_ms(),
            )

        generated = await self._provider.generate(
            [
                ChatMessage(role="system", content=self._system_prompt.text),
                ChatMessage(
                    role="user",
                    content=self._user_prompt.render(
                        context=_format_context(results), question=question
                    ),
                ),
            ]
        )
        return RagAnswer(
            answer=generated.text,
            found=True,
            sources=results,
            usage=Usage(
                input_tokens=embedded.usage.input_tokens + generated.usage.input_tokens,
                output_tokens=generated.usage.output_tokens,
            ),
            latency_ms=elapsed_ms(),
        )
