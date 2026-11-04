# OOP Design Notes

This document records actual design decisions as implementation progresses.

## Principles

- Python supports OOP directly; use classes and interfaces where they clarify a real boundary.
- Use `Protocol` or ABC for replaceable components (LLM provider, vector store, repository, tool).
- Use dataclasses or Pydantic models for data; do not add behaviour-free classes with logic.
- Prefer composition and dependency injection over inheritance.
- Keep pure logic (chunking, scoring, prompt building) in functions.
- Prefer simple designs over patterns added for appearance.

## Implementation record

| Concept | Language | File / symbol | Status | Rationale |
|---|---|---|---|---|
| Interface (provider) | Python | `src/ai_ops/llm/provider.py`, `LLMProvider` | Implemented | Swap mock and real LLM without changing services |
| Decorator (retries) | Python | `src/ai_ops/llm/retry.py`, `RetryingProvider` | Implemented | Adds retries to any provider, keeps them out of each implementation. It owns the wrapped provider and closes it |
| Factory (provider selection) | Python | `src/ai_ops/llm/factory.py`, `build_provider` | Implemented | One place names concrete providers. The app owns the result and closes it at shutdown |
| Interface (vector store) | Python | `src/ai_ops/retrieval/vector_store.py`, `VectorStore`, `InMemoryVectorStore` | Implemented | Move from in-memory to a vector database |
| Repository | Python | `src/ai_ops/integrations/records.py`, `RecordRepository`, `InMemoryRecordRepository` | Interface implemented, SQL planned | Keep SQL out of services and tools |
| Adapter | Python | Fill after implementation | Planned | Isolate the third-party service |
| Tool registry | Python | `src/ai_ops/agents/registry.py`, `ToolRegistry`, `Tool` | Implemented | Typed tools for the agent. One gate validates, logs, and sends writes to approval |
| State machine (approval) | Python | `src/ai_ops/services/approval.py`, `ApprovalService.decide` | Implemented | Explicit pending, executing, executed, rejected, and failed states. The workflow reads its status from here |
| Step checkpoints (workflow) | Python | `src/ai_ops/services/workflow.py`, `TicketWorkflow.run`, `_Run` | Implemented | Each step saves its result, so a retry only repeats what failed |
| Idempotency key | Python | `src/ai_ops/integrations/outbound.py`, `OutboundService.send` | Implemented | A retry after a lost response cannot send a message twice |
| Dependency injection | Python | `src/ai_ops/api/app.py`, `create_app`; `src/ai_ops/api/routes.py`, `get_rag_service`; `src/ai_ops/services/rag.py`, `RagService.__init__` | Implemented | Services receive their provider and store. The app owns them, and routes get the service through `Depends` |
| Application factory | Python | `src/ai_ops/api/app.py`, `create_app` | Implemented | Settings are passed in, so importing has no side effects and each test builds its own app |
| ASGI middleware | Python | `src/ai_ops/api/middleware.py`, `RequestContextMiddleware` | Implemented | One start and end point for the request id, request log, and last-resort 500 |

Update this table only with real source paths and symbols after implementation. Do not claim planned patterns are implemented.
