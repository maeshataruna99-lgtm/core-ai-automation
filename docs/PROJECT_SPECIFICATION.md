# Project Specification

## 1. Product vision

Create a small, understandable AI operations assistant that shows how an AI engineer designs, integrates, evaluates, and optimises AI-powered applications for business needs. The project is a learning portfolio; it does not clone any vendor product.

## 2. Users

- Developer/learner building AI engineering skills.
- Portfolio reviewer assessing software engineering and applied AI.
- Operator persona: a staff member who asks questions, looks up records, and approves AI-drafted actions.

## 3. MVP use case

A small company has an internal knowledge base (policies, product notes) and an order or ticket database. The operator chats with the assistant. The assistant answers from documents with citations, looks up records with tools, and runs one workflow that drafts a response and waits for human approval.

## 4. Traceability to the job posting

| Posting item | Where it is covered |
|---|---|
| Design, develop, and implement AI-powered applications | Whole project, FR-01 to FR-08 |
| Build and integrate AI models, LLMs, and APIs | FR-01 LLM provider layer, FR-08 HTTP API |
| Develop AI agents, chatbots, and workflow automation | FR-04 agent, FR-05 workflow |
| Generative AI, RAG, prompt engineering | FR-02 RAG, FR-03 prompt templates |
| Translate business requirements into technical solutions | Use case in section 3 and acceptance criteria |
| Evaluate, test, and optimise for accuracy, performance, scalability, reliability | FR-07 evaluation and diagnostics |
| Integrate with databases, internal systems, third-party services | FR-06 integrations |
| Stay up to date with AI technologies | `docs/ROADMAP.md` review step per milestone |
| Python, API-based applications, clean maintainable code | Technology choices in `README.md`, non-functional requirements in section 6 |
| Vector databases, embeddings, agents (plus) | FR-02, FR-04 |
| ML fundamentals, data processing, model evaluation | FR-07 |

## 5. Functional requirements

### FR-01 LLM provider layer
- One interface for text generation and embeddings.
- A deterministic mock provider for tests and offline use.
- One real provider (OpenAI-compatible HTTP API) selected by configuration.
- Timeouts, bounded retries with backoff, and clear errors.
- API keys only from environment variables; never logged.

### FR-02 Retrieval-Augmented Generation
- Ingest `.txt` and `.md` documents into chunks with metadata (document id, position).
- Create embeddings and store them in a vector store behind an interface.
- Retrieve top-k chunks by similarity.
- Answer only from retrieved context and cite sources.
- Say so when no relevant context exists.

### FR-03 Prompt engineering
- Prompt templates live in versioned files, not inline strings.
- System prompt instructs grounded answers and refusal when unsure.
- Retrieved text and user input are treated as data, never as instructions.

### FR-04 Agent and chatbot
- Tool registry with typed inputs and outputs.
- Tools: knowledge search, record lookup, record update (guarded).
- Agent loop with a maximum step count, timeout, and tool-call log.
- Tools that change data require explicit approval before they run.

### FR-05 Workflow automation
- One end-to-end workflow: classify incoming ticket, look up related record, draft reply, request approval, send through a stubbed outbound service.
- Each step is logged and can be retried or stopped.

### FR-06 Integrations
- SQL database access through a repository layer (SQLite first).
- One third-party-style service behind an adapter with a stub for tests.
- Optional Redis cache for repeated embedding or answer calls.

### FR-07 Evaluation and diagnostics
- A golden set of questions with expected source documents and expected key facts.
- Metrics: retrieval hit rate, answer groundedness check, latency, tokens, and estimated cost.
- Regression tests that fail when a prompt, model, or chunking change lowers scores beyond a defined threshold.
- Structured logs with request id, timestamp, severity, tool calls, and latency.

### FR-08 HTTP API
- Endpoints: `GET /health`, `POST /ask`, `POST /chat`, `POST /workflows/ticket`, `POST /approvals/{id}`, `GET /records/{id}`.
- Pydantic request and response models; input length limits.
- Default bind address: `127.0.0.1`.

### FR-09 Cross-platform
- Runs and tests on Windows and Linux.
- No OS-specific code in the core logic.

## 6. Non-functional requirements

- Typed, readable Python with docstrings.
- No secrets in the repository or logs.
- External input validated at the API boundary.
- Tests run without network access or paid keys.
- Documentation states limitations.
- No real customer data in the MVP.

## 7. Out of scope

- Production deployment, autoscaling, multi-tenant access.
- Fine-tuning or training new models.
- User authentication and role management (note as future work).
- Computer vision and speech.
- Claims of accuracy beyond what the evaluation set measures.

## 8. Optional frontend

A Vue 3 dashboard may be added after the API contract is stable. See `docs/OPTIONAL_VUE_FRONTEND.md`. It must not become the only way to use the assistant.

## 9. UI/UX specification

See `docs/UI_UX_DESIGN.md`.
