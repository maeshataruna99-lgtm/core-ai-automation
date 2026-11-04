# Acceptance Criteria and Verification Plan

## Build and portability
- [x] Project installs and tests pass on Windows.
- [x] Project installs and tests pass on Linux.
- [ ] No OS-specific code in core logic.
- [ ] Setup instructions match the tested commands.
- [x] CI runs lint and tests.

## API behaviour
- [ ] Valid requests return documented responses.
- [ ] Invalid requests are rejected with 422 and a clear message.
- [ ] Request size and agent step limits are enforced.
- [ ] Upstream LLM failures return 502; timeouts return 504.
- [ ] The API binds to 127.0.0.1 by default.

## LLM layer
- [ ] Application runs fully with the mock provider.
- [ ] A real provider can be selected through configuration only.
- [ ] Retries are bounded and tested.
- [ ] No API key appears in logs, errors, or the repository.

## RAG
- [ ] Answers cite source documents.
- [ ] A question without context returns an explicit "not found" answer.
- [ ] Chunking and retrieval are deterministic in tests.
- [ ] Prompt-injection text in a document does not alter system behaviour.

## Agent and workflow
- [ ] The agent stops at the step limit.
- [ ] Write tools run only after approval.
- [ ] Rejected actions change no data.
- [ ] Tool calls are logged.
- [ ] Workflow retries do not duplicate side effects.

## Integrations
- [ ] SQL access uses parameterised queries.
- [ ] Repositories are tested against a temporary database.
- [ ] Cache failure does not break the application.
- [ ] Adapter errors are surfaced.

## Evaluation
- [ ] A golden set exists with expected sources and facts.
- [ ] One command runs evaluation and prints a report.
- [ ] A regression threshold fails the run when scores fall.
- [ ] Reported metrics come from recorded runs.

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
