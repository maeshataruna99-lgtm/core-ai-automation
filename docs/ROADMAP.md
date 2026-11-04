# Technical Roadmap

## Milestone 0 — Repository and development environment

**Deliverables**
- Git repository and README.
- Python project skeleton (`pyproject.toml`, `src/ai_ops/`, `tests/`).
- Virtual environment instructions for Windows and Linux.
- Ruff and pytest configuration; warning and formatting policy.
- CI skeleton.

**Acceptance criteria**
- [x] A clean checkout installs and runs tests on Windows.
- [x] A clean checkout installs and runs tests on Linux.
- [x] Setup instructions are reproducible.
- [x] CI reports lint and test status.

## Milestone 1 — Python API foundation

**Learning topics**
- Type hints, dataclasses, Pydantic models.
- FastAPI routing, dependency injection, error handling.
- Settings from environment variables.
- Structured logging.

**Features**
- `GET /health`.
- Settings module and logging module.
- Request id middleware.
- Consistent error responses.

**Acceptance criteria**
- [x] API starts and stops cleanly.
- [x] Invalid input returns 422 with a clear message.
- [x] Logs contain timestamp, severity, request id.
- [x] Missing configuration fails fast with a clear error.
- [x] Tests cover health, validation, and error mapping.

## Milestone 2 — LLM integration layer

**Learning topics**
- Provider interface (Protocol or ABC).
- HTTP clients, timeouts, retries, backoff.
- Prompt templates and versioning.
- Token and cost accounting.

**Features**
- Provider interface for generation and embeddings.
- Deterministic mock provider.
- OpenAI-compatible provider selected by configuration.
- Prompt template loader.

**Acceptance criteria**
- [x] Application runs fully with the mock provider and no network.
- [x] Real provider errors map to 502, timeouts to 504.
- [x] Retries are bounded and tested with a fake failing provider.
- [x] API keys never appear in logs or error messages.
- [ ] Provider can be swapped without changing service code.

## Milestone 3 — Retrieval-Augmented Generation

**Learning topics**
- Chunking strategies and overlap.
- Embeddings and similarity metrics.
- Vector stores and metadata.
- Grounded prompting and citation.

**Features**
- Document ingestion for `.txt` and `.md`.
- Chunker with size and overlap settings.
- Vector store interface with an in-memory implementation.
- `POST /ask` with sources.
- Optional second implementation (pgvector, Qdrant, or Chroma) after tests pass.

**Acceptance criteria**
- [ ] Answers include source ids.
- [ ] A question with no relevant context returns a "not found" answer, not an invented one.
- [ ] Retrieval is deterministic under the mock embedding provider.
- [ ] Tests cover chunk boundaries, empty store, and ranking order.
- [ ] Prompt injection text inside a document does not change system behaviour (tested).

## Milestone 4 — Agent and workflow automation

**Learning topics**
- Tool calling and schemas.
- Agent loops and stopping conditions.
- Human-in-the-loop approval.
- Workflow state and retries.

**Features**
- Tool registry with typed tools.
- Agent loop with max steps and timeout.
- Approval service for write actions.
- Ticket workflow: classify, look up, draft, approve, send via stub.

**Acceptance criteria**
- [ ] Agent stops at the step limit with a clear message.
- [ ] Write tools never run without approval.
- [ ] Rejected approvals leave data unchanged.
- [ ] Each tool call is logged with inputs and outcome.
- [ ] Workflow steps can be retried without duplicate side effects.

## Milestone 5 — Data and system integrations

**Learning topics**
- Repository pattern and SQL.
- Migrations.
- Adapters for third-party services.
- Caching.

**Features**
- SQLite repositories for orders or tickets.
- Third-party adapter with a stub.
- Optional Redis cache for embeddings or answers.
- `GET /records/{id}`.

**Acceptance criteria**
- [ ] Repositories are tested against a temporary database.
- [ ] SQL uses parameters; no string-built queries.
- [ ] The application still works when the cache is unavailable.
- [ ] Adapter failures are surfaced, not hidden.

## Milestone 6 — Evaluation, testing, and optimisation

**Learning topics**
- ML evaluation basics: precision, recall, hit rate.
- Groundedness checks.
- Latency and cost measurement.
- Regression testing for prompts and models.

**Features**
- Golden question set with expected sources and facts.
- Evaluation runner and report.
- Latency and token metrics per request.
- Documented optimisation steps with before and after numbers.

**Acceptance criteria**
- [ ] Evaluation runs with one command and prints a report.
- [ ] A regression threshold fails the run when scores drop.
- [ ] Reported numbers come from actual runs and are stored with the date and configuration.
- [ ] At least one optimisation (chunk size, top-k, caching) is measured.

## Milestone 7 — Portfolio polish

**Deliverables**
- English README and Indonesian project overview.
- Architecture diagram.
- API documentation (OpenAPI) and examples.
- Evaluation report.
- Screenshots and a short demo video.
- Setup instructions for Windows and Linux.
- Known limitations.
- Resume bullet points that match completed work.

**Acceptance criteria**
- [ ] A fresh clone can be set up by following the documentation.
- [ ] The demo shows a grounded answer with sources, a tool lookup, and an approval-gated action.
- [ ] No unsupported claims about production use, accuracy, or employer experience.
- [ ] Every resume claim is supported by repository evidence.

## Suggested order

Do not start the frontend before Milestone 4 is stable. First make the LLM layer reliable, then retrieval, then agents, then integrations, then measure.

## Milestone 7A — UI/UX specification and wireframe

**Deliverables**
- `docs/UI_UX_DESIGN.md`.
- Chat screen, sources panel, and approval screen definitions.
- Interaction states for loading, error, and pending approval.

**Acceptance criteria**
- [ ] The wireframe is reviewed before any frontend code.
- [ ] Loading, error, empty, no-answer, and pending-approval states are specified.
- [ ] The assistant remains fully usable through the API without the UI.

## Milestone 7B — Optional Vue frontend

**Status:** Optional enhancement, not an MVP blocker.

**Deliverables**
- Vue 3 chat and approval screens using the existing API.
- Mock API mode for UI development.

**Acceptance criteria**
- [ ] The UI shows sources next to each answer.
- [ ] Pending, approved, rejected, and failed states are visible.
- [ ] The UI cannot trigger a write action without an approval step.
- [ ] The API and tests work when the frontend is absent.
