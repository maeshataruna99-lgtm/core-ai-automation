# Python Learning Path: From Web Developer to AI Engineer

This path assumes prior programming experience (Node.js and Vue.js). Focus on Python idioms, typing, and AI-specific concepts rather than beginner theory.

## Stage 1 — Python project basics
Learn:
- Virtual environments, `pyproject.toml`, packages and imports.
- Type hints, `dataclass`, `Protocol`.
- Exceptions and context managers.
Practice:
- Create the package skeleton and run ruff and pytest.

## Stage 2 — HTTP APIs with FastAPI
Learn:
- Routing, Pydantic models, dependency injection.
- Async endpoints and error handlers.
- Settings from environment variables.
Practice:
- Build `/health` and one validated endpoint with tests.

## Stage 3 — Calling LLM APIs
Learn:
- HTTP clients, timeouts, retries, backoff.
- Chat message format, tokens, temperature.
- Prompt templates and prompt versioning.
Practice:
- Write a provider interface, a mock provider, and an OpenAI-compatible provider.

## Stage 4 — Embeddings and retrieval
Learn:
- What embeddings are; cosine similarity.
- Chunking and overlap trade-offs.
- Vector stores and metadata filtering.
Practice:
- Build an in-memory vector store; later swap in pgvector, Qdrant, or Chroma.
- Cite sources in every answer.

## Stage 5 — Agents and tool calling
Learn:
- Tool schemas and structured outputs.
- Agent loops, step limits, and stopping rules.
- Human-in-the-loop approval.
- Prompt injection and why retrieved text is data.
Practice:
- Build a tool registry and a guarded write tool.

## Stage 6 — Data integration
Learn:
- SQL, parameterised queries, repository pattern.
- Migrations.
- Adapters for third-party APIs; Redis caching basics.
Practice:
- Add an orders or tickets repository and an adapter stub.

## Stage 7 — Evaluation and testing
Learn:
- Golden sets, hit rate, precision and recall at k.
- Groundedness checks.
- Latency, token, and cost measurement.
- Regression tests for prompts and models.
Practice:
- Write the evaluation runner and record results with date and configuration.

## Stage 8 — Optional frontend and delivery
Learn:
- Vue 3 composition API, API client, loading and error states.
- Packaging and basic containerisation (only if needed).
Practice:
- Build a chat view that shows sources and approvals.

## Learning principle

Every new concept should solve a concrete problem in the assistant. Avoid adding frameworks or features merely to make the project look impressive.

## Code documentation rule

The code is also study material, so every script and function must be understandable when reread later. Each one documents:

- **Why:** the problem it solves and why it is built this way.
- **Used by:** the callers, or what triggers it. Update this when callers change.
- **What it does:** its purpose, plus inputs, outputs, errors, and side effects when they exist.

Write for a middle developer: assume programming experience, explain Python idioms and AI concepts that are not obvious, and do not restate each line. This applies to source, tests, scripts, and CI workflows.

```python
def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Split text into overlapping chunks for embedding.

    Why: Embedding models have an input limit. Overlap keeps a sentence that
    crosses a boundary retrievable from at least one chunk.

    Used by: `RagService.ingest` during document ingestion.

    Args:
        text: Document text.
        size: Maximum characters per chunk.
        overlap: Characters shared between neighbouring chunks.

    Returns:
        The chunks, in document order.
    """
```
