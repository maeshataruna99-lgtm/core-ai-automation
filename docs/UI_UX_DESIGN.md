# UI/UX Design Specification

**Status:** built in `src/ai_ops/ui/` as plain HTML, CSS, and JavaScript, with no build step and no dependencies. See `docs/demo/ui-demo.mp4` for a recording. Open it at `http://127.0.0.1:8000` while the API runs.

**How it was checked:** on 2026-11-04, a script drove the real page in Microsoft Edge against the running API, and each criterion in section 10 passed. It covered light and dark themes, 375 px and 1280 px widths, keyboard-only use, text contrast of at least 4.5:1, and a document containing script tags. That script is not in this repository, so `tests/test_ui.py` only covers what needs no browser. Other browsers and screen readers were not tested.

## 1. Design goals

Create a clear chat and approval interface for an educational AI operations assistant.

Priorities:
1. Answers always show where they came from.
2. The user can tell loading, failure, and "no answer" apart.
3. Actions that change data always show a pending approval step.
4. AI-generated content is labelled as such.
5. The UI works at common desktop and phone widths.
6. Visual polish supports the demonstration; it does not hide system behaviour.

## 2. User personas

### Operator
Asks questions, checks records, and approves or rejects drafted actions.

### Portfolio reviewer
Needs to see the system working quickly and inspect sources, tool calls, and evaluation results.

## 3. Information architecture

- Chat
- Approvals
- Records
- Diagnostics (tool-call log, latency, token counts)
- About / limitations

For the MVP, a single page with a sidebar or tabs is sufficient.

## 4. Main layout

```text
┌──────────────────────────────────────────────────────────────────────┐
│ AI Operations Assistant            Mock provider • Connected         │
├─────────────┬────────────────────────────────────────────────────────┤
│ Navigation  │ Chat                                                   │
│             │                                                        │
│ Chat        │  You: Status of ORD-1001?                              │
│ Approvals   │  Assistant (AI-generated): Dispatched on 2 Nov.        │
│ Records     │    Sources: orders/ORD-1001                            │
│ Diagnostics │                                                        │
│ About       │  ┌──────────────────────────────────────────────────┐  │
│             │  │ Pending approval: update ORD-1002 to "shipped"   │  │
│             │  │ [Approve] [Reject]                               │  │
│             │  └──────────────────────────────────────────────────┘  │
│             │  [ Type a message…                          ] [Send]  │
├─────────────┴────────────────────────────────────────────────────────┤
│ Last request: 420 ms • 312 tokens • request id abc123                 │
└──────────────────────────────────────────────────────────────────────┘
```

## 5. Screen specifications

### Chat
- Message list with user and assistant messages.
- Each assistant message shows its sources and an "AI-generated" label.
- Input with length limit and a visible counter.

### Approvals
- List of pending, approved, and rejected actions with target, change, and time.
- Approve and reject require a clear target and a confirmation click.

### Records
- Read-only record viewer by id.

### Diagnostics
- Tool-call log with inputs, outcome, and latency.
- Token counts and estimated cost per request.
- Read-only.

## 6. Visual language

- Neutral light and dark themes with high-contrast text.
- Blue for information, amber for warnings, red for errors, green only for confirmed success.
- Do not communicate status by colour alone; pair colour with label or icon.
- Consistent spacing, typography, and units.
- Avoid decorative animation.

## 7. Interaction rules

- Show a loading state while a request runs, with a cancel option.
- Show success only after the API confirms it.
- On timeout, display "no confirmation received"; do not claim success.
- Disable send and approve controls while a request is pending.
- A "no answer" result is shown as such, not as an error and not as an invented answer.
- Retrieved text is rendered as plain text, never as HTML.

## 8. Accessibility and usability

- Keyboard navigation for chat, approve, and reject.
- Visible focus states and adequate contrast.
- Minimum practical hit targets.
- Status labels alongside colours.
- Reduced-motion support.
- Layout works at 375 px and 1280 px widths.

## 9. UI implementation order

1. Wireframe and static layout.
2. API client with a mock mode.
3. Chat with sources.
4. Approvals.
5. Records and diagnostics.
6. Usability review and demo polish.

## 10. UI acceptance criteria

- [x] Every assistant answer shows sources or an explicit "no sources" note.
- [x] Loading, error, no-answer, and pending-approval states are distinguishable.
- [x] Write actions cannot run without an approval click.
- [x] Retrieved text is rendered safely.
- [x] Core operation works by keyboard.
- [x] Layout is usable at 375 px and 1280 px.
