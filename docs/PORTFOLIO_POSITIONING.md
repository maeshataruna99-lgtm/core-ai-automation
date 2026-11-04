# Portfolio positioning

## Honest project description

> Educational AI operations assistant in Python and FastAPI: retrieval-augmented generation with source citations, a tool-using agent where a person must approve every write, an idempotent ticket workflow, SQLite integration with migrations, and an evaluation suite that gates CI. Tested with a mock LLM provider and synthetic data.

## Resume bullets

Each bullet below is supported by the row with the same number in the evidence table.

1. Built a Python and FastAPI service with typed contracts, structured JSON logs with request ids, and one error format across all endpoints.
2. Designed an LLM provider interface with a deterministic mock and an OpenAI-compatible HTTP client, with bounded retries and timeouts, so that tests need no network or API key.
3. Implemented a RAG pipeline with chunking, cosine similarity search, source citations, and a "not found" path that skips the model when nothing relevant is retrieved.
4. Built a hand-written agent loop with a typed tool registry and step and time limits, where write tools can only create a pending approval, and verified this with a fake model that always attempts the write.
5. Built an end-to-end ticket workflow with resumable steps and idempotency keys, so that retries do not repeat finished steps or send a message twice.
6. Integrated SQLite through a repository with parameterised queries and transactional migrations, plus a third-party HTTP adapter that surfaces failures without leaking keys.
7. Built an evaluation suite with a golden question set, stored reports, and a CI regression gate, and used it to measure a retrieval optimisation and its trade-offs.
8. Set up CI on Windows and Linux for tests, lint, evaluation, and commit messages, with a one-command demo that checks each result end to end.
9. Built a dependency-free web interface for each feature, which renders untrusted text safely, works by keyboard and at phone width, and can only write through the approval flow.

## Evidence

| # | Claim | Where to see it |
|---|---|---|
| 1 | API, logs, and errors | `src/ai_ops/api/app.py`, `src/ai_ops/api/errors.py`, `src/ai_ops/logging.py`, `tests/test_api.py` |
| 2 | Provider interface and retries | `src/ai_ops/llm/provider.py`, `src/ai_ops/llm/retry.py`, `tests/test_llm_openai_and_retry.py` |
| 3 | RAG and "not found" | `src/ai_ops/services/rag.py`, `src/ai_ops/retrieval/chunking.py`, `tests/test_rag.py`, `tests/test_retrieval.py` |
| 4 | Agent and approval | `src/ai_ops/agents/loop.py`, `src/ai_ops/agents/registry.py`, `src/ai_ops/services/approval.py`, `tests/test_agent.py` |
| 5 | Ticket workflow | `src/ai_ops/services/workflow.py`, `tests/test_workflow.py` |
| 6 | SQLite and adapter | `src/ai_ops/integrations/records.py`, `src/ai_ops/integrations/database.py`, `src/ai_ops/integrations/outbound.py`, `tests/test_records.py`, `tests/test_outbound.py` |
| 7 | Evaluation | `src/ai_ops/evaluation/runner.py`, `data/evaluation/golden.json`, `docs/EVALUATION.md`, `.github/workflows/ci.yml` |
| 8 | CI and demo | `.github/workflows/ci.yml`, `.github/workflows/commitlint.yml`, `scripts/demo.py`, `tests/test_demo_and_docs.py` |
| 9 | Web interface | `src/ai_ops/ui/app.js`, `tests/test_ui.py`, `docs/UI_UX_DESIGN.md`, `docs/demo/ui-demo.mp4` |

## What not to claim

- **Production use.** It has not been deployed, and has had no real users or customer data.
- **Employer or client experience.** This is a personal learning project.
- **Accuracy.** The evaluation scores describe retrieval with a mock that matches exact words, on synthetic data. Do not quote them as model accuracy.
- **Experience with a real LLM in this project.** It has only run against a mock and simulated responses. Add this only after a real run.
- **Resistance to prompt injection.** Tests prove that a manipulated model cannot change data without approval. They do not prove that a real model ignores an attack.
- **Scale or latency.** The measured latency does not include a model call.

## Before using these bullets

- **Be able to explain each one.** Read the code and be ready to walk through it without notes. `docs/PYTHON_LEARNING_PATH.md` lists the concepts, and each function documents why it exists and where it is used.
- **Be accurate about how it was built.** If you are asked how the work was done or which tools you used, answer truthfully.

## Mapping to the job posting

Responsibilities and requirements in the AI Engineer listing map to `docs/PROJECT_SPECIFICATION.md`, section 4. Only quote what the repository can show.

## Suggested demo sequence

`python scripts/demo.py` runs this sequence and checks each result. `docs/demo/ui-demo.mp4` shows the same features in the web interface:

1. Start the API with the mock provider.
2. Ask a question, and show the sources.
3. Ask something outside the knowledge base, and show the "not found" answer.
4. Ask about an order, and show the tool call.
5. Ask for a change, and show that it waits for approval and the record has not changed.
6. Approve it, and show the record change.
7. Run a ticket through the workflow.
8. Show the tool calls in the audit log.
9. Run the evaluation, and show the report.

In an interview, also open `src/ai_ops/agents/registry.py` to show how a write tool's handler can only be reached through approval, and point out what the mock cannot show: real answers.
