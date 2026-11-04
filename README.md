# AI Operations Assistant

Educational AI engineering project and portfolio piece for demonstrating AI-powered applications in Python: LLM integration, Retrieval-Augmented Generation (RAG), AI agents and chatbots, workflow automation, and integration with databases and third-party services.

**Target role:** AI Engineer  
**Platforms:** Windows + Linux  
**Implementation order:** Python API foundation → LLM layer → RAG → agent and workflow → data integrations → evaluation → optional Vue dashboard

> This is a learning project. It is not a production system. Do not feed it real customer data, and do not claim production deployment, accuracy figures, or company experience that the repository cannot show.

## Project goals

- Design and build an AI-powered application that solves a concrete business need (see `docs/PROJECT_SPECIFICATION.md`).
- Integrate LLMs and APIs behind a provider interface so models can be swapped.
- Implement RAG with embeddings and a vector store, with source citations.
- Build an agent/chatbot that calls tools to automate a business workflow.
- Integrate with a SQL database and one third-party-style service.
- Evaluate and optimise for accuracy, latency, cost, and reliability, with tests that prove it.
- Produce an honest portfolio: tests, docs, and demo material.

## Languages and technology choices

| Area | Choice | Why |
|---|---|---|
| Core language | Python 3.12 | The posting requires Python proficiency and API-based applications |
| API framework | FastAPI + Pydantic | Typed request/response contracts, automatic docs |
| Data | SQL (SQLite first, PostgreSQL later) | The posting requires integration with databases |
| Vector search | In-memory store first, then pgvector, Qdrant, or Chroma | Learn the concept before adding infrastructure |
| LLM access | Provider interface, with a mock provider and an OpenAI-compatible provider | Tests run without a paid key |
| Cache / queue (optional) | Redis | Listed in the posting skills; add only when a concrete need appears |
| Frontend (optional) | Vue 3 + Node.js | Already in the author's skill set; the core works without it |
| Tests | pytest | Unit, integration, and evaluation tests |
| Lint / format | ruff | One tool, fast |
| Version control | Git | |
| CI | GitHub Actions (optional) | |

Not selected: Go, Laravel, React, Next.js, Computer Vision. They appear in the posting's skill chips, but the responsibilities and requirements text only demands Python, LLMs/GenAI, RAG, and integration. Add one only if the specification asks for it.

## MVP scope

An operations assistant for a small company:
- Answers questions from an internal knowledge base (RAG with citations).
- Looks up and updates business records (orders or tickets) through tools.
- Runs one automated workflow end to end (for example: classify an incoming ticket, look up the record, draft a reply, request human approval).
- Exposes everything through a documented HTTP API.
- Ships with an evaluation set and a test report.

## Development setup

Requires Python 3.12 and Git.

Windows (PowerShell):

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Run the checks (same on both platforms; CI runs them on Windows and Linux):

```bash
ruff check .
ruff format --check .
pytest
```

## Run the API

Windows (PowerShell):

```powershell
$env:AI_OPS_LLM_PROVIDER = "mock"
python -m ai_ops
```

Linux:

```bash
AI_OPS_LLM_PROVIDER=mock python -m ai_ops
```

Run it from the repository root, so that it finds the sample documents in `data/knowledge`. Then open `http://127.0.0.1:8000/health` or the generated docs at `http://127.0.0.1:8000/docs`. Stop the API with Ctrl+C.

Ask a question:

```bash
curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" -d "{\"question\": \"How long do card refunds take?\"}"
```

The response has the `answer`, whether anything relevant was `found`, the `sources` it used, token `usage`, and `latency_ms`. The mock provider does not answer questions: it echoes the question with a `[mock]` prefix, so it only demonstrates retrieval and the response format. Real answers need `openai_compatible`.

Documents are loaded from `AI_OPS_KNOWLEDGE_DIR` at startup and kept in memory. With a real provider, every restart pays to embed them again.

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `AI_OPS_LLM_PROVIDER` | Yes | | `mock` (offline, no key) or `openai_compatible` |
| `AI_OPS_LLM_BASE_URL` | With `openai_compatible` | | API root, such as `https://api.example.com/v1` |
| `AI_OPS_LLM_API_KEY` | With `openai_compatible` | | Provider key. Set it only in the environment, never in a file in this repository |
| `AI_OPS_LLM_CHAT_MODEL` | With `openai_compatible` | | Model for text generation |
| `AI_OPS_LLM_EMBEDDING_MODEL` | With `openai_compatible` | | Model for embeddings |
| `AI_OPS_LLM_TIMEOUT_SECONDS` | No | `30` | Time limit for each network step of one attempt |
| `AI_OPS_LLM_MAX_RETRIES` | No | `2` | Retries for temporary failures, from 0 to 5 |
| `AI_OPS_KNOWLEDGE_DIR` | No | `data/knowledge` | Directory of `.md` and `.txt` documents |
| `AI_OPS_RAG_CHUNK_SIZE` | No | `800` | Maximum characters per chunk |
| `AI_OPS_RAG_CHUNK_OVERLAP` | No | `100` | Characters repeated between chunks |
| `AI_OPS_RAG_TOP_K` | No | `4` | Maximum chunks used per question |
| `AI_OPS_RAG_MIN_SCORE` | No | `0.15` | Lowest similarity counted as relevant. Chosen for the mock provider, so check it when using a real embedding model |
| `AI_OPS_LOG_LEVEL` | No | `INFO` | `DEBUG`, `INFO`, `WARNING`, or `ERROR` |
| `AI_OPS_HOST` | No | `127.0.0.1` | Bind address |
| `AI_OPS_PORT` | No | `8000` | Bind port |

The API exits with a clear message when a required variable is missing or a value is invalid. Logs are JSON lines on stderr. Every response has an `X-Request-ID` header that matches the logs.

## Quick-start (planned)

The exact commands will be added as the features are built.

1. Run the tests and the evaluation set.
2. Add an API key through environment variables only when using a real provider.

## Repository status

Milestones 0 to 3: the project skeleton, tooling, CI, the API foundation (`GET /health`, settings, structured logging, request ids, and consistent errors), the LLM layer (provider interface, mock provider, OpenAI-compatible provider, retries, and prompt templates), and RAG (`POST /ask`, document loading, chunking, and an in-memory vector store) exist. The OpenAI-compatible provider has only been tested against a simulated API, so how a real model handles prompt injection is not verified. The agent and workflow features are not implemented yet.

## Commit messages

Commits follow [Conventional Commits](https://www.conventionalcommits.org/). CI checks every push to `main` and every pull request with commitlint; the rules live in `commitlint.config.mjs`.

```text
<type>(<scope>): <subject>
```

- **Type:** `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`, `revert`.
- **Scope (optional):** `api`, `services`, `llm`, `retrieval`, `agents`, `integrations`, `evaluation`, `frontend`, `docs`, `ci`, `deps`.
- **Subject:** imperative, lower case, no trailing period. The header is at most 72 characters.

Examples: `feat(retrieval): add chunk overlap setting`, `fix(api): return 504 on provider timeout`.

## Document map

| File | Purpose |
|---|---|
| `docs/PROJECT_SPECIFICATION.md` | What to build, requirements, out of scope |
| `docs/ARCHITECTURE.md` | Components, module boundaries, API contract |
| `docs/ROADMAP.md` | Milestones and acceptance checkboxes |
| `docs/ACCEPTANCE_CRITERIA.md` | Verification checklist |
| `docs/PYTHON_LEARNING_PATH.md` | Learning stages tied to project problems |
| `docs/OOP_DESIGN_NOTES.md` | Record of implemented design decisions |
| `docs/UI_UX_DESIGN.md` | Chat and dashboard UI specification |
| `docs/OPTIONAL_VUE_FRONTEND.md` | Optional Vue track |
| `docs/PORTFOLIO_POSITIONING.md` | Honest description and limits |
