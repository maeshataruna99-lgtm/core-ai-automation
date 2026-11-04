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

## Quick-start (planned)

The exact commands will be added once the API exists.

1. Run the API in local-only mode with the mock LLM provider.
2. Run the tests and the evaluation set.
3. Add an API key through environment variables only when using a real provider.

## Repository status

Milestone 0: the Python project skeleton, tooling, and CI exist. There is no application code yet, and it does not claim that the software has been implemented.

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
