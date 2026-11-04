# Architecture

## 1. High-level components

```text
+-----------------------------+
| Client                      |
| - curl / API docs           |
| - Optional Vue dashboard    |
+--------------+--------------+
               |  HTTP/JSON  127.0.0.1:8000
               v
+-----------------------------+
| API layer (FastAPI)         |
| - Request validation        |
| - Auth placeholder          |
| - Error mapping             |
+--------------+--------------+
               |
               v
+-----------------------------+
| Application services        |
| - Chat / agent service      |
| - RAG service               |
| - Workflow service          |
| - Approval service          |
+------+--------------+-------+
       |              |
       v              v
+-------------+  +-----------------------------+
| LLM layer   |  | Integrations                |
| - Provider  |  | - Repositories (SQL)        |
|   interface |  | - Vector store interface    |
| - Mock      |  | - Third-party adapter       |
| - OpenAI-   |  | - Optional Redis cache      |
|   compatible|  +-----------------------------+
+-------------+
               |
               v
+-----------------------------+
| Logging / evaluation        |
+-----------------------------+
```

## 2. Design rules

- Application services depend on interfaces, not on a specific model vendor or database.
- The API layer holds no business logic.
- Retrieved documents and user messages are data. They never change system instructions or tool permissions.
- Tools that change data pass through the approval service.
- Secrets come from environment variables via one settings module.
- Prompt templates are files; code loads them by name and version.
- Keep the vector store, the SQL repositories, and the LLM provider independently replaceable and testable.

## 3. Suggested Python module boundaries

```text
src/ai_ops/
  api/            FastAPI app, routers, request/response schemas
  services/       rag, workflow, approval
  llm/            provider interface, mock provider, openai-compatible provider, prompts/
  retrieval/      chunking, embeddings, vector store interface and implementations
  agents/         tool registry, tools, agent loop
  integrations/   repositories (SQL), third-party adapter, cache
  evaluation/     golden set, metrics, runner
  config.py       settings loaded from environment
  logging.py      structured logging
tests/            unit, integration, evaluation
data/             sample documents and golden set
docs/
```

Do not create a new top-level package without a reason and an update to this document.

## 4. Data flow: question answering

1. `POST /ask` validates the body and length.
2. RAG service embeds the question through the LLM layer.
3. Vector store returns top-k chunks with scores.
4. Prompt template combines context and question.
5. LLM provider returns the answer.
6. Service returns the answer, sources, latency, and token counts.

## 5. Data flow: agent with a guarded tool

1. `POST /chat` receives a message.
2. Agent loop asks the LLM which tool to call, bounded by max steps.
3. Read-only tools run immediately.
4. A write tool creates a pending approval and returns its id.
5. `POST /approvals/{id}` approves or rejects; only then does the write run.
6. Every step is logged with the request id.

## 6. API contract (MVP draft)

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness and loaded document count |
| POST | `/ask` | Grounded Q&A with sources |
| POST | `/chat` | Agent chat with tool use |
| POST | `/workflows/ticket` | Run the ticket workflow |
| POST | `/approvals/{id}` | Approve or reject a pending action |
| GET | `/records/{id}` | Read a business record |

Rules:
- Every response model is a Pydantic class and is documented in the generated OpenAPI.
- Validation errors return 422; upstream LLM failures return 502 with a safe message; timeouts return 504.
- Keep the contract versioned. Breaking changes need a new version path.

## 7. Concurrency model

- FastAPI with async endpoints. LLM calls use async HTTP with timeouts.
- SQLite in MVP: one connection per request or a small pool.
- Avoid background workers until a concrete need appears (the workflow can run inline first).

## 8. Safety and network defaults

- Bind to `127.0.0.1` by default.
- Do not log prompts or documents that may contain personal data unless a debug flag is on.
- Limit request body size and agent steps.
- Do not expose the API to a network without a separate security review.
- Label all outputs as AI-generated suggestions in the UI.

## 9. Optional frontend

A Vue 3 dashboard consumes the same HTTP API. It is not part of the Python core. See `docs/OPTIONAL_VUE_FRONTEND.md`.
