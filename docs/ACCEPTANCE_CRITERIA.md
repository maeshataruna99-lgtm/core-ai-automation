# Acceptance Criteria and Verification Plan

## Build and portability
- [x] Project installs and tests pass on Windows.
- [x] Project installs and tests pass on Linux.
- [ ] No OS-specific code in core logic.
- [x] Setup instructions match the tested commands.
- [x] CI runs lint and tests.

## API behaviour
- [x] Valid requests return documented responses.
- [x] Invalid requests are rejected with 422 and a clear message.
- [x] Request size and agent step limits are enforced.
- [x] Upstream LLM failures return 502; timeouts return 504.
- [x] The API binds to 127.0.0.1 by default.

## LLM layer
- [x] Application runs fully with the mock provider.
- [x] A real provider can be selected through configuration only.
- [x] Retries are bounded and tested.
- [x] No API key appears in logs, errors, or the repository.

## RAG
- [x] Answers cite source documents.
- [x] A question without context returns an explicit "not found" answer.
- [x] Chunking and retrieval are deterministic in tests.
- [ ] Prompt-injection text in a document does not alter system behaviour.

## Agent and workflow
- [x] The agent stops at the step limit.
- [x] Write tools run only after approval.
- [x] Rejected actions change no data.
- [x] Tool calls are logged.
- [x] Workflow retries do not duplicate side effects.

## Integrations
- [x] SQL access uses parameterised queries.
- [x] Repositories are tested against a temporary database.
- [ ] Cache failure does not break the application.
- [x] Adapter errors are surfaced.

## Evaluation
- [x] A golden set exists with expected sources and facts.
- [x] One command runs evaluation and prints a report.
- [x] A regression threshold fails the run when scores fall.
- [x] Reported metrics come from recorded runs.

## UI (optional)
- [ ] Sources are visible next to answers.
- [ ] Loading, error, no-answer, and pending-approval states are visible.
- [ ] Write actions cannot bypass approval.
- [ ] The API remains usable without the UI.

## Portfolio readiness
- [ ] README includes setup and run steps.
- [ ] Architecture diagram exists.
- [ ] API examples are documented.
- [ ] Evaluation report is included.
- [ ] Demo video or GIF is available.
- [ ] Tests can be run by reviewers.
- [ ] Limitations are clearly stated.
- [ ] CV bullet points describe only completed and verified features.
