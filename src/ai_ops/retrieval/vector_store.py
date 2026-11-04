"""Vector store interface and its in-memory implementation.

Why: Learning the concept comes before adding infrastructure. The `VectorStore` interface
means a database such as pgvector can replace the in-memory version later without changing
the RAG service.

Used by: `ai_ops.services.rag.RagService` and `ai_ops.api.app.create_app`.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Chunk:
    """A piece of a document, with where it came from.

    Why: The metadata is what makes citations possible.

    Used by: `VectorStore`, `ai_ops.services.rag`, and `ai_ops.api.routes`.
    """

    document_id: str
    position: int
    text: str

    @property
    def source_id(self) -> str:
        """Return the id used in citations, such as `policies/refunds.md#2`.

        Why: One string identifies a chunk in the prompt, the response, and the evaluation set.

        Used by: `ai_ops.services.rag`, `ai_ops.api.routes`, and the sort in `search`.

        Returns:
            The document id and position.
        """
        return f"{self.document_id}#{self.position}"


@dataclass(frozen=True)
class ScoredChunk:
    """A search result.

    Why: Keeping the score with the chunk lets the service apply a relevance threshold.

    Used by: returned by `VectorStore.search`.
    """

    chunk: Chunk
    score: float


class VectorStore(Protocol):
    """What the RAG service needs from any vector store.

    Why: The service depends on this interface, not on a specific database.

    Used by: `ai_ops.services.rag.RagService`.
    """

    def add(self, chunks: Sequence[Chunk], vectors: Sequence[list[float]]) -> None:
        """Store chunks with their embedding vectors.

        Why: Ingestion writes through this method.

        Used by: `RagService.ingest`.

        Args:
            chunks: The chunks.
            vectors: One vector per chunk, in the same order.

        Raises:
            ValueError: The two sequences have different lengths.
        """
        ...

    def search(self, vector: list[float], top_k: int) -> list[ScoredChunk]:
        """Find the most similar chunks.

        Why: This is the "retrieval" in Retrieval-Augmented Generation.

        Used by: `RagService.ask`.

        Args:
            vector: The query embedding.
            top_k: Maximum number of results.

        Returns:
            The best matches, highest score first.
        """
        ...


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Compute how closely two vectors point in the same direction.

    Why: Cosine ignores vector length, so a long chunk does not score higher just for having
    more words. The standard library is fast enough for a small in-memory collection.

    Used by: `InMemoryVectorStore.search`.

    Args:
        left: First vector.
        right: Second vector, of the same length.

    Returns:
        A score from -1 to 1, or 0 if either vector has no direction.

    Raises:
        ValueError: The vectors have different lengths, which happens when a store holds
            embeddings from a different model.
    """
    length = math.hypot(*left) * math.hypot(*right)
    if length == 0:
        return 0.0
    return math.sumprod(left, right) / length


class InMemoryVectorStore:
    """Implementation of `VectorStore` that compares the query with every chunk.

    Why: Exact search over a Python list is simple and enough for a few thousand chunks.

    shortcut: data is lost on restart and search time grows with each chunk, use pgvector or
    Qdrant when the collection outgrows startup ingestion.

    Used by: `ai_ops.api.app.create_app` and tests.
    """

    def __init__(self) -> None:
        """Start with an empty store.

        Why: Chunks and vectors are held in one list, so they cannot get out of step.

        Used by: `ai_ops.api.app.create_app` and tests.
        """
        self._items: list[tuple[Chunk, list[float]]] = []

    def add(self, chunks: Sequence[Chunk], vectors: Sequence[list[float]]) -> None:
        """Store chunks with their vectors.

        Why: See `VectorStore.add`.

        Used by: `RagService.ingest`.

        Args:
            chunks: The chunks.
            vectors: One vector per chunk, in the same order.

        Raises:
            ValueError: The two sequences have different lengths.
        """
        self._items.extend(zip(chunks, vectors, strict=True))

    def search(self, vector: list[float], top_k: int) -> list[ScoredChunk]:
        """Score every chunk and return the best ones.

        Why: Sorting by source id after the score makes ties deterministic, which the roadmap
        requires for retrieval under the mock provider.

        Used by: `RagService.ask`.

        Args:
            vector: The query embedding.
            top_k: Maximum number of results.

        Returns:
            The best matches, highest score first.
        """
        scored = [
            ScoredChunk(chunk=chunk, score=_cosine_similarity(vector, chunk_vector))
            for chunk, chunk_vector in self._items
        ]
        scored.sort(key=lambda result: (-result.score, result.chunk.source_id))
        return scored[:top_k]
