# Optional Vue Frontend Track

## 1. Purpose

A Vue 3 dashboard can make the portfolio demo clearer, but it is not required for the API, the RAG pipeline, the agent, or the evaluation suite. Vue and Node.js are already in the author's skill set, so this track reuses that experience.

**Architecture decision:** keep the frontend a separate client of the HTTP API. Do not move business logic, prompts, or model keys into the browser.

## 2. Proposed architecture

```text
Python API (FastAPI)
  └── HTTP/JSON
        ├── curl / OpenAPI docs (primary)
        └── Optional Vue 3 app (Vite)
```

- The browser never holds an LLM API key. Only the Python API talks to model providers.
- Configure the API base URL through an environment variable.
- Enable CORS for the local development origin only.

## 3. Scope

MVP screens:
- Chat with sources.
- Approvals list.
- Record viewer.
- Diagnostics view.

Avoid a design system build-out. Clear states and honest labels matter more than visual fidelity.

## 4. Mapping API data to UI

| API data | UI representation |
|---|---|
| `answer` and `sources` | Message with a source list |
| `tool` and tool-call log | Diagnostics entry |
| Pending approval id | Approval card with approve and reject |
| 422 / 502 / 504 errors | Distinct error messages |
| Latency and tokens | Footer metrics |

The UI shows API state. It does not invent state of its own.

## 5. Interaction boundary

- The UI calls the same endpoints as any other client.
- Write actions go through `POST /approvals/{id}`; the UI cannot bypass it.
- Render all model and document text as plain text.

## 6. Performance and reliability

- Keep the bundle small; no heavy UI frameworks without need.
- Handle network errors, timeouts, and cancelled requests.
- Provide a mock API mode so the UI can run without the backend.

## 7. Suggested implementation sequence

1. Finish the API and its tests (milestones 1 to 6).
2. Complete `docs/UI_UX_DESIGN.md` and review the wireframe.
3. Create the Vue app with a mock API.
4. Connect to the real API.
5. Add approvals and diagnostics.
6. Test on the target browsers and document limitations.

## 8. Acceptance criteria

- [ ] The UI runs against a mock API on its own.
- [ ] Sources are visible next to answers.
- [ ] Error, timeout, and no-answer states are distinct.
- [ ] Write actions require an approval click and cannot bypass the API.
- [ ] No API key or secret is present in the frontend bundle.
- [ ] The API and tests work when the frontend is absent.
- [ ] README labels the frontend as optional.

## 9. Portfolio guidance

In an interview, first demonstrate the API, the grounded answers, and the evaluation report. Then show the Vue app as a client. This shows that the AI logic is separate from the interface.
