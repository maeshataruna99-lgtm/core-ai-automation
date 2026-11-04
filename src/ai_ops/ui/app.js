/**
 * The web interface: a thin client of the HTTP API.
 *
 * Why: Plain JavaScript needs no Node.js or build step, so anyone who can run the API can use this
 * page. It holds no business logic, no keys, and no state of its own beyond the current browser
 * session: every status it shows comes from the API. So it cannot run a write without an approval,
 * and it does not claim success before the API confirms it.
 *
 * All text from the API goes on the page through `textContent` (see `el`), never as HTML, because
 * answers and documents are untrusted.
 *
 * Used by: index.html.
 */

const REQUEST_TIMEOUT_MS = 90_000;

const PAGES = [
  { id: "ask", label: "Ask", render: renderAsk },
  { id: "chat", label: "Chat", render: renderChat },
  { id: "tickets", label: "Tickets", render: renderTickets },
  { id: "approvals", label: "Approvals", render: renderApprovals },
  { id: "records", label: "Records", render: renderRecords },
  { id: "insert", label: "Insert Data", render: renderInsert },
  { id: "diagnostics", label: "Diagnostics", render: renderDiagnostics },
  { id: "settings", label: "Settings", render: renderSettings },
  { id: "about", label: "About", render: renderAbout },
];

/** Status labels and colours. Each status has words, so colour is never the only signal. */
const APPROVAL_STATUS = {
  pending: { text: "Waiting for approval", tone: "warning" },
  executing: { text: "Running", tone: "info" },
  executed: { text: "Done", tone: "success" },
  rejected: { text: "Rejected", tone: "" },
  failed: { text: "Failed, can be retried", tone: "error" },
};

const TICKET_STATUS = {
  pending_approval: { text: "Waiting for approval", tone: "warning" },
  sending: { text: "Sending", tone: "info" },
  sent: { text: "Sent", tone: "success" },
  rejected: { text: "Rejected, not sent", tone: "" },
  send_failed: { text: "Sending failed, can be retried", tone: "error" },
};

const PROVIDER_NAMES = {
  mock: "Mock provider",
  deepseek: "DeepSeek",
  openai_compatible: "OpenAI-compatible provider",
};

const ORDER_STATUSES = ["processing", "shipped", "delivered", "cancelled"];

const STOP_REASON_NOTICE = {
  max_steps: "The agent reached its step limit and did not finish.",
  timeout: "The agent reached its time limit and did not finish.",
};

/**
 * What the current browser session remembers. It is lost when the page reloads.
 * Why: The API keeps no chat history or request log, so these are kept here only for display.
 * Used by: the chat and diagnostics pages, and `callApi`.
 */
const session = { chat: [], requests: [] };

/**
 * Create an element with properties and children.
 * Why: This is the only way this file adds content to the page. Strings become text nodes, so text
 * from the API can never become markup.
 * Used by: every render function.
 * @param {string} tag The element name.
 * @param {object} [properties] Properties to set, and `class` for the class name.
 * @param {...(Node|string|null|undefined|false)} children Child nodes. Empty values are skipped, so
 *   callers can write `condition && child`.
 * @returns {HTMLElement} The new element.
 */
function el(tag, properties = {}, ...children) {
  const element = document.createElement(tag);
  for (const [name, value] of Object.entries(properties)) {
    if (name === "class") element.className = value;
    else if (name.startsWith("aria-")) element.setAttribute(name, value);
    else element[name] = value;
  }
  element.append(...children.filter((child) => child !== null && child !== undefined && child !== false));
  return element;
}

/**
 * Scroll a new result into view.
 * Why: The result appears below the form, which can be off screen on a small window. Without this, a
 * person could press a button and see nothing change.
 * Used by: the ask, chat, and tickets pages.
 * @param {HTMLElement} element The result.
 */
function reveal(element) {
  element.scrollIntoView({ block: "nearest" });
}

/**
 * Build a status badge.
 * Why: Keeps the label and colour together, wherever a status is shown.
 * Used by: the ticket and approval cards.
 * @param {{text: string, tone: string}} status The label and colour.
 * @returns {HTMLElement} The badge.
 */
function badge({ text, tone }) {
  return el("span", { class: `badge ${tone ? `badge-${tone}` : ""}` }, text);
}

/**
 * Build a label that marks text as written by a model.
 * Why: The project rules require AI output to be labelled, so nobody mistakes it for a fact that
 * the system checked.
 * Used by: the ask, chat, and ticket pages.
 * @returns {HTMLElement} The label.
 */
function aiLabel() {
  return badge({ text: "AI-generated", tone: "info" });
}

/**
 * Build a coloured message box.
 * Why: Error, warning, and information messages must look different from each other.
 * Used by: every page that reports a result.
 * @param {"info"|"warning"|"error"|"success"} tone The kind of message.
 * @param {string} text The message.
 * @returns {HTMLElement} The box.
 */
function notice(tone, text) {
  return el("p", { class: `notice notice-${tone}`, role: tone === "error" ? "alert" : "status" }, text);
}

/**
 * Call the API and record the outcome.
 * Why: One place handles timeouts, network failures, and the shared error format, so each page only
 * deals with a clear result. Network failures do not throw, because they are normal here.
 * Used by: every page that talks to the API.
 * @param {string} path The URL path. It is relative to the site root.
 * @param {object} [options] `method`, `body`, an AbortController `signal`, and `background` for a call
 *   the person did not ask for. Those are not recorded, so they do not hide the metrics of a real one.
 * @returns {Promise<object>} `ok` with `data`, or `problem` with a message for a person.
 */
async function callApi(path, { method = "GET", body, signal, background = false } = {}) {
  const started = performance.now();
  const timeout = AbortSignal.timeout(REQUEST_TIMEOUT_MS);
  const signals = signal ? AbortSignal.any([signal, timeout]) : timeout;
  const result = { ok: false, status: null, requestId: null, data: null, problem: null };
  try {
    const response = await fetch(path, {
      method,
      signal: signals,
      headers: body ? { "Content-Type": "application/json" } : undefined,
      body: body ? JSON.stringify(body) : undefined,
    });
    result.status = response.status;
    result.requestId = response.headers.get("X-Request-ID");
    // A 204 has no body, so reading it as JSON would make a success look like a network failure.
    const data = response.status === 204 ? null : await response.json();
    if (response.ok) {
      result.ok = true;
      result.data = data;
    } else {
      result.problem = describeError(response.status, data.error);
    }
  } catch (error) {
    if (signal?.aborted) {
      result.cancelled = true;
      result.problem = "Cancelled. The server may still have finished this request.";
    } else if (timeout.aborted) {
      result.problem = "No confirmation received in time. The request may still have worked, so check before trying again.";
    } else {
      result.problem = "Could not reach the API. Check that it is running.";
    }
  }
  if (!background) recordRequest(path, method, result, performance.now() - started);
  return result;
}

/**
 * Turn an API error into a message for a person.
 * Why: Validation, upstream, and timeout errors need different actions from the person.
 * Used by: `callApi`.
 * @param {number} status The HTTP status.
 * @param {object} [error] The API's error body.
 * @returns {string} The message.
 */
function describeError(status, error) {
  if (status === 422 && error?.details?.length) {
    return `Check your input. ${error.details.map((detail) => detail.message).join(". ")}.`;
  }
  if (status === 504) return "The model provider did not answer in time. Nothing was confirmed. You can try again.";
  return error?.message ?? `The request failed with status ${status}.`;
}

/**
 * Save a request for the diagnostics page, and update the status bar.
 * Why: Latency and token counts show what each request cost. Numbers that the API did not report
 * stay empty. The browser's own timing is shown only as a fallback, and is labelled as such.
 * Used by: `callApi`.
 * @param {string} path The URL path.
 * @param {string} method The HTTP method.
 * @param {object} result The outcome.
 * @param {number} elapsed The time the browser measured, in milliseconds.
 */
function recordRequest(path, method, result, elapsed) {
  const entry = {
    time: new Date().toLocaleTimeString(),
    request: `${method} ${path}`,
    status: result.status ?? (result.cancelled ? "cancelled" : "no response"),
    latency: result.data?.latency_ms ?? null,
    browserMs: Math.round(elapsed),
    usage: result.data?.usage ?? null,
    requestId: result.requestId,
    toolCalls: result.data?.tool_calls ?? [],
  };
  session.requests.unshift(entry);
  const tokens = entry.usage ? ` • ${entry.usage.input_tokens + entry.usage.output_tokens} tokens` : "";
  const timing = entry.latency !== null ? `${entry.latency} ms` : `${entry.browserMs} ms in the browser`;
  document.querySelector("#last-request").textContent =
    `Last request: ${entry.request} • ${entry.status} • ${timing}${tokens}` +
    (entry.requestId ? ` • request id ${entry.requestId}` : "");
}

/**
 * Show one question in a modal dialog, and wait for the answer.
 * Why: The specification asks for a confirmation click with a clear target on every approval. A
 * native dialog handles keyboard focus and the Esc key. The safe choice has focus first, so a stray
 * Enter does not approve anything.
 * Used by: `decide`.
 * @param {string} title The question.
 * @param {string} detail What exactly will happen.
 * @param {string} acceptLabel The text on the confirm button.
 * @returns {Promise<boolean>} True if the person confirmed.
 */
function confirmAction(title, detail, acceptLabel) {
  const dialog = document.querySelector("#confirm-dialog");
  const cancel = document.querySelector("#confirm-cancel");
  const accept = document.querySelector("#confirm-accept");
  document.querySelector("#confirm-title").textContent = title;
  document.querySelector("#confirm-detail").textContent = detail;
  accept.textContent = acceptLabel;
  return new Promise((resolve) => {
    const finish = (answer) => {
      accept.onclick = cancel.onclick = dialog.onclose = null;
      dialog.close();
      resolve(answer);
    };
    accept.onclick = () => finish(true);
    cancel.onclick = () => finish(false);
    dialog.onclose = () => finish(false);
    dialog.showModal();
    cancel.focus();
  });
}

/**
 * Describe what an approval will do, in words.
 * Why: A person should not need to read JSON to know what they are approving. An unknown kind still
 * shows its raw data, so that nothing is hidden.
 * Used by: the approval cards and the confirmation dialog.
 * @param {object} approval The approval from the API.
 * @returns {string} The description.
 */
function describeApproval(approval) {
  const { kind, payload } = approval;
  if (kind === "update_record") return `Change order ${payload.record_id} to "${payload.status}".`;
  if (kind === "send_ticket_reply") return `Send this reply to ticket ${payload.ticket_id}:\n\n${payload.reply}`;
  return `${kind}: ${JSON.stringify(payload)}`;
}

/**
 * Ask for confirmation, then send a decision to the API.
 * Why: This is the only code in the interface that can trigger a write, and it always goes through
 * `POST /approvals/{id}`.
 * Used by: `approvalButtons`.
 * @param {object} approval The approval.
 * @param {"approve"|"reject"} decision The decision.
 * @returns {Promise<object|null>} The API result, or null if the person went back.
 */
async function decide(approval, decision) {
  const approving = decision === "approve";
  const confirmed = await confirmAction(
    approving ? "Approve this action?" : "Reject this action?",
    describeApproval(approval),
    approving ? "Yes, approve" : "Yes, reject",
  );
  if (!confirmed) return null;
  const result = await callApi(`/approvals/${encodeURIComponent(approval.id)}`, {
    method: "POST",
    body: { decision },
  });
  refreshPendingCount();
  return result;
}

/**
 * Build the approve and reject buttons for an approval.
 * Why: The same controls appear on the chat, tickets, and approvals pages. Buttons are disabled while
 * a request runs, so a decision cannot be sent twice. A failed action can only be approved again,
 * since a decision that has started cannot be rejected.
 * Used by: `approvalCard`.
 * @param {object} approval The approval.
 * @param {(result: object) => void} onDecided Called with the API result.
 * @returns {HTMLElement|null} The buttons, or null if no decision is possible.
 */
function approvalButtons(approval, onDecided) {
  if (approval.status !== "pending" && approval.status !== "failed") return null;
  const retrying = approval.status === "failed";
  const approve = el("button", { type: "button", class: "button button-primary" }, retrying ? "Retry" : "Approve");
  const reject = !retrying && el("button", { type: "button", class: "button" }, "Reject");
  const buttons = el("div", { class: "actions" }, approve, reject);
  const run = async (decision) => {
    approve.disabled = true;
    if (reject) reject.disabled = true;
    const result = await decide(approval, decision);
    approve.disabled = false;
    if (reject) reject.disabled = false;
    if (result) onDecided(result);
  };
  approve.onclick = () => run("approve");
  if (reject) reject.onclick = () => run("reject");
  return buttons;
}

/**
 * Fetch an approval's current state and show it as a card that updates itself.
 * Why: Cards on the chat and tickets pages only know an id. Asking the API each time means that the
 * page never shows a status of its own making.
 * Used by: the chat and tickets pages.
 * @param {string} approvalId The approval id.
 * @param {() => void} [onChanged] Called after a decision.
 * @returns {HTMLElement} A container that fills in once the API answers.
 */
function liveApprovalCard(approvalId, onChanged) {
  const holder = el("div", {}, el("p", { class: "hint" }, "Loading approval…"));
  const load = async (problem) => {
    const result = await callApi("/approvals", { background: true });
    const approval = result.data?.find((item) => item.id === approvalId);
    if (!approval) {
      holder.replaceChildren(notice("error", result.problem ?? "This approval no longer exists. The API may have restarted."));
      return;
    }
    holder.replaceChildren(
      approvalCard(approval, async (decision) => {
        await load(decision.ok ? null : decision.problem);
        onChanged?.();
      }, problem),
    );
  };
  load();
  return holder;
}

/**
 * Build a card for one approval.
 * Why: Shows the target, the change, the time, and the status, as the specification asks.
 * Used by: the approvals page and `liveApprovalCard`.
 * @param {object} approval The approval.
 * @param {(result: object) => void} onDecided Called with the API result of a decision.
 * @param {string|null} [problem] An error to show on the card.
 * @returns {HTMLElement} The card.
 */
function approvalCard(approval, onDecided, problem = null) {
  const status = APPROVAL_STATUS[approval.status] ?? { text: approval.status, tone: "" };
  return el(
    "article",
    { class: "card" },
    el(
      "div",
      { class: "card-head" },
      el("h3", {}, approval.kind === "update_record" ? "Change an order" : "Send a ticket reply"),
      badge(status),
    ),
    el("p", { class: "preserve-lines" }, describeApproval(approval)),
    el("p", { class: "hint" }, `Requested ${new Date(approval.created_at).toLocaleString()}`),
    problem && notice("error", problem),
    approval.status === "executed" &&
      approval.result &&
      el("p", { class: "hint mono" }, `Result: ${JSON.stringify(approval.result)}`),
    approvalButtons(approval, onDecided),
  );
}

/**
 * Build a text box with a live character count.
 * Why: The API rejects long input, so the limit should be visible while typing. `maxLength` stops
 * the input at the limit. Ctrl+Enter sends, because Enter alone starts a new line.
 * Used by: the ask, chat, and tickets pages.
 * @param {object} options The `id`, `label`, `limit`, and optional `onSend`.
 * @returns {{field: HTMLElement, input: HTMLTextAreaElement}} The field and its input.
 */
function textField({ id, label, limit, onSend }) {
  const input = el("textarea", { id, maxLength: limit, required: true });
  const counter = el("span", { class: "hint", id: `${id}-count` }, `0 / ${limit}`);
  const update = () => {
    counter.textContent = `${input.value.length} / ${limit}`;
  };
  input.addEventListener("input", update);
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && event.ctrlKey) onSend?.();
  });
  const field = el(
    "div",
    { class: "field" },
    el("label", { htmlFor: id }, label),
    input,
    el("div", { class: "hint" }, counter, " • Ctrl+Enter to send"),
  );
  return { field, input };
}

/**
 * Build buttons that fill in example text.
 * Why: A new user can try each feature without knowing what is in the sample data.
 * Used by: the ask, chat, and tickets pages.
 * @param {string[]} examples The example texts.
 * @param {HTMLTextAreaElement} input The field to fill.
 * @returns {HTMLElement} The buttons.
 */
function exampleButtons(examples, input) {
  return el(
    "div",
    { class: "examples", "aria-label": "Examples" },
    el("span", { class: "hint" }, "Try:"),
    ...examples.map((text) => {
      const button = el("button", { type: "button", class: "example" }, text);
      button.onclick = () => {
        input.value = text;
        input.dispatchEvent(new Event("input"));
        input.focus();
      };
      return button;
    }),
  );
}

/**
 * Connect a form to an API call, with loading, cancel, and disabled states.
 * Why: Each page needs the same behaviour: show progress, offer a cancel, stop double sends, and show
 * only what the API confirmed.
 * Used by: the ask, chat, tickets, and records pages.
 * @param {object} options The `area` for results, the `submit` button, and the async `send` function,
 *   which receives an AbortSignal.
 * @returns {() => void} A function that starts the request.
 */
function requestRunner({ area, submit, send }) {
  let running = false;
  return async () => {
    if (running) return;
    running = true;
    submit.disabled = true;
    const controller = new AbortController();
    const cancel = el("button", { type: "button", class: "button" }, "Cancel");
    cancel.onclick = () => controller.abort();
    const loading = el("div", { class: "loading", role: "status" }, el("span", { class: "spinner" }), "Working…", cancel);
    area.prepend(loading);
    try {
      await send(controller.signal);
    } finally {
      loading.remove();
      submit.disabled = false;
      running = false;
    }
  };
}

/** Build the Ask page. Why: Grounded answers with sources. Used by: PAGES. */
function renderAsk() {
  const area = el("div", {});
  const submit = el("button", { type: "button", class: "button button-primary" }, "Ask");
  let send;
  const { field, input } = textField({ id: "ask-question", label: "Question", limit: 1000, onSend: () => send() });
  send = requestRunner({
    area,
    submit,
    send: async (signal) => {
      const question = input.value.trim();
      if (!question) {
        area.replaceChildren(notice("warning", "Type a question first."));
        return;
      }
      const result = await callApi("/ask", { method: "POST", body: { question }, signal });
      area.replaceChildren(result.ok ? askResult(result.data) : notice(result.cancelled ? "warning" : "error", result.problem));
      reveal(area);
    },
  });
  submit.onclick = send;
  return [
    el("h2", {}, "Ask the knowledge base"),
    el("p", { class: "lead" }, "Answers come only from the company documents, with their sources."),
    el(
      "div",
      { class: "form" },
      field,
      exampleButtons(["How long do card refunds take?", "How much do I pay for express delivery?", "What is the capital of France?"], input),
      el("div", { class: "actions" }, submit),
    ),
    area,
  ];
}

/**
 * Show the result of a question.
 * Why: "No answer" is a normal result, so it is shown as information, not as an error. Sources are
 * always visible, so a person can check the answer. A model can still fail to answer when `found` is
 * true, so the label only says what the system knows: that relevant text was found.
 * Used by: `renderAsk`.
 * @param {object} data The API response.
 * @returns {HTMLElement} The result.
 */
function askResult(data) {
  if (!data.found) {
    return el(
      "article",
      { class: "card" },
      el("div", { class: "card-head" }, el("h3", {}, "No answer found"), badge({ text: "No sources", tone: "" })),
      el("p", {}, data.answer),
      el("p", { class: "hint" }, "Nothing in the documents was relevant enough, so the model was not asked."),
    );
  }
  return el(
    "article",
    { class: "card" },
    el("div", { class: "card-head" }, el("h3", {}, "Answer"), aiLabel()),
    el("p", { class: "preserve-lines" }, data.answer),
    el("h3", {}, `Sources (${data.sources.length})`),
    ...data.sources.map((source) =>
      el(
        "details",
        {},
        el("summary", {}, el("code", {}, source.id), ` • similarity ${source.score.toFixed(2)}`),
        el("p", { class: "preserve-lines hint" }, source.text),
      ),
    ),
  );
}

/** Build the Chat page. Why: The agent and its tools. Used by: PAGES. */
function renderChat() {
  const area = el("div", {});
  const history = el("div", {});
  const submit = el("button", { type: "button", class: "button button-primary" }, "Send");
  let send;
  const { field, input } = textField({ id: "chat-message", label: "Message", limit: 2000, onSend: () => send() });
  const draw = () => history.replaceChildren(...session.chat.map(chatCard));
  send = requestRunner({
    area,
    submit,
    send: async (signal) => {
      const message = input.value.trim();
      if (!message) {
        area.replaceChildren(notice("warning", "Type a message first."));
        return;
      }
      area.replaceChildren();
      const result = await callApi("/chat", { method: "POST", body: { message }, signal });
      if (!result.ok) {
        area.replaceChildren(notice(result.cancelled ? "warning" : "error", result.problem));
        return;
      }
      session.chat.unshift({ message, data: result.data });
      input.value = "";
      input.dispatchEvent(new Event("input"));
      draw();
      reveal(history.firstElementChild);
      refreshPendingCount();
    },
  });
  submit.onclick = send;
  draw();
  return [
    el("h2", {}, "Chat with the agent"),
    el(
      "p",
      { class: "lead" },
      "The agent can search documents and read orders. It can only propose changes, and you decide. Each message starts fresh, because the agent keeps no history.",
    ),
    el(
      "div",
      { class: "form" },
      field,
      exampleButtons(["What is the status of ORD-1002?", "How long do card refunds take?", "Please update ORD-1002 to shipped"], input),
      el("div", { class: "actions" }, submit),
    ),
    area,
    history,
  ];
}

/**
 * Show one chat exchange.
 * Why: The tool list comes from the API, and shows what really happened. A model's reply can claim
 * something else, so the two are shown separately.
 * Used by: `renderChat`.
 * @param {{message: string, data: object}} exchange The message and the API response.
 * @returns {HTMLElement} The exchange.
 */
function chatCard({ message, data }) {
  return el(
    "section",
    { class: "card" },
    el("div", { class: "card card-user" }, el("h3", {}, "You"), el("p", { class: "preserve-lines" }, message)),
    el("div", { class: "card-head" }, el("h3", {}, "Assistant"), aiLabel()),
    STOP_REASON_NOTICE[data.stop_reason] && notice("warning", STOP_REASON_NOTICE[data.stop_reason]),
    el("p", { class: "preserve-lines" }, data.reply),
    data.tool_calls.length > 0 && el("h3", {}, "What the agent did"),
    ...data.tool_calls.map((call) =>
      el(
        "p",
        { class: "mono" },
        `${call.name} ${JSON.stringify(call.arguments)} `,
        badge(toolStatus(call.status)),
      ),
    ),
    ...data.pending_approvals.map((approvalId) => liveApprovalCard(approvalId)),
  );
}

/**
 * Give a tool call's status a label.
 * Why: Uses the same words wherever tool calls appear.
 * Used by: the chat and diagnostics pages.
 * @param {string} status The status from the API.
 * @returns {{text: string, tone: string}} The label and colour.
 */
function toolStatus(status) {
  if (status === "ok") return { text: "Done", tone: "success" };
  if (status === "pending_approval") return { text: "Only proposed", tone: "warning" };
  return { text: "Failed", tone: "error" };
}

/** Build the Tickets page. Why: The ticket workflow. Used by: PAGES. */
function renderTickets() {
  const area = el("div", {});
  const submit = el("button", { type: "button", class: "button button-primary" }, "Run workflow");
  const ticketId = el("input", { id: "ticket-id", value: `T-${Date.now() % 100000}`, maxLength: 64 });
  let send;
  const { field, input } = textField({ id: "ticket-message", label: "Ticket text", limit: 4000, onSend: () => send() });
  const runWorkflow = async (signal) => {
    const message = input.value.trim();
    if (!message) {
      area.replaceChildren(notice("warning", "Type the ticket text first."));
      return;
    }
    const body = { ticket_id: ticketId.value.trim(), message };
    const result = await callApi("/workflows/ticket", { method: "POST", body, signal });
    area.replaceChildren(
      result.ok
        ? ticketCard(result.data, () => runWorkflow())
        : notice(result.cancelled ? "warning" : "error", result.problem),
    );
    reveal(area);
    refreshPendingCount();
  };
  send = requestRunner({ area, submit, send: runWorkflow });
  submit.onclick = send;
  return [
    el("h2", {}, "Run a ticket"),
    el(
      "p",
      { class: "lead" },
      "Classifies the ticket, finds the order, and drafts a reply. Nothing is sent until you approve it. Running the same ticket again is safe.",
    ),
    el(
      "div",
      { class: "form" },
      el(
        "div",
        { class: "field" },
        el("label", { htmlFor: "ticket-id" }, "Ticket id"),
        ticketId,
        el("span", { class: "hint" }, "Letters, numbers, dashes, and underscores."),
      ),
      field,
      exampleButtons(["Where is my delivery for ORD-1001?", "How long does a refund take?"], input),
      el("div", { class: "actions" }, submit),
    ),
    area,
  ];
}

/**
 * Show a ticket's state.
 * Why: Shows the draft next to its approval, so the person reads what they are about to send.
 * Used by: `renderTickets`.
 * @param {object} data The API response.
 * @param {() => void} reload Asks the API for the new state after a decision.
 * @returns {HTMLElement} The card.
 */
function ticketCard(data, reload) {
  return el(
    "article",
    { class: "card" },
    el(
      "div",
      { class: "card-head" },
      el("h3", {}, `Ticket ${data.ticket_id}`),
      badge(TICKET_STATUS[data.status] ?? { text: data.status, tone: "" }),
    ),
    el(
      "dl",
      { class: "facts" },
      el("dt", {}, "Category"),
      el("dd", {}, data.category),
      el("dt", {}, "Order"),
      el("dd", {}, data.record_id ?? "No matching order"),
    ),
    el("div", { class: "card-head" }, el("h3", {}, "Draft reply"), aiLabel()),
    el("p", { class: "preserve-lines" }, data.draft),
    data.status === "sent" && notice("success", "The API confirmed that this reply was sent."),
    liveApprovalCard(data.approval_id, reload),
  );
}

/** Build the Approvals page. Why: One place to decide on actions. Used by: PAGES. */
function renderApprovals() {
  const area = el("div", {}, el("p", { class: "hint" }, "Loading…"));
  const refresh = el("button", { type: "button", class: "button" }, "Refresh");
  const load = async (problem = null) => {
    refresh.disabled = true;
    const result = await callApi("/approvals");
    refresh.disabled = false;
    if (!result.ok) {
      area.replaceChildren(notice("error", result.problem));
      return;
    }
    refreshPendingCount();
    const onDecided = (decision) => load(decision.ok ? null : decision.problem);
    const pending = result.data.filter((item) => item.status === "pending" || item.status === "failed");
    const decided = result.data.filter((item) => !pending.includes(item));
    // `el` skips empty values. `replaceChildren` would show `false` as text, so it gets one wrapper.
    area.replaceChildren(
      el(
        "div",
        {},
        problem && notice("error", problem),
        el("h3", { class: "section-title" }, `Needs a decision (${pending.length})`),
        pending.length
          ? el("div", {}, ...pending.map((item) => approvalCard(item, onDecided)))
          : el("p", { class: "empty" }, "Nothing is waiting for a decision."),
        decided.length > 0 && el("h3", { class: "section-title" }, `Decided (${decided.length})`),
        ...decided.map((item) => approvalCard(item, onDecided)),
      ),
    );
  };
  refresh.onclick = () => load();
  load();
  return [
    el("h2", {}, "Approvals"),
    el(
      "p",
      { class: "lead" },
      "Actions that change data or send a message wait here. The list is kept in memory, so it is empty after the API restarts.",
    ),
    el("div", { class: "actions" }, refresh),
    area,
  ];
}

/** Build the Records page. Why: Read-only view of an order. Used by: PAGES. */
function renderRecords() {
  const area = el("div", {});
  const input = el("input", { id: "record-id", value: "ORD-1001", maxLength: 32 });
  const submit = el("button", { type: "button", class: "button button-primary" }, "Look up");
  const send = requestRunner({
    area,
    submit,
    send: async (signal) => {
      const recordId = input.value.trim();
      const result = await callApi(`/records/${encodeURIComponent(recordId)}`, { signal });
      area.replaceChildren(
        result.ok
          ? el(
              "article",
              { class: "card" },
              el("h3", {}, `Order ${result.data.id}`),
              el(
                "dl",
                { class: "facts" },
                el("dt", {}, "Customer"),
                el("dd", {}, result.data.customer),
                el("dt", {}, "Item"),
                el("dd", {}, result.data.item),
                el("dt", {}, "Status"),
                el("dd", {}, result.data.status),
              ),
            )
          : notice(result.cancelled ? "warning" : "error", result.problem),
      );
    },
  });
  submit.onclick = send;
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter") send();
  });
  send();
  return [
    el("h2", {}, "Records"),
    el("p", { class: "lead" }, "Read-only. Orders only change through an approved action."),
    el(
      "div",
      { class: "form" },
      el(
        "div",
        { class: "field" },
        el("label", { htmlFor: "record-id" }, "Order id"),
        input,
        el("span", { class: "hint" }, "The sample orders are ORD-1001, ORD-1002, and ORD-1003."),
      ),
      el("div", { class: "actions" }, submit),
    ),
    area,
  ];
}

/**
 * Build a labelled text input.
 * Why: Keeps each label tied to its input, for keyboard and screen reader users.
 * Used by: the Insert Data page.
 * @param {string} id The input's id.
 * @param {string} label The label.
 * @param {object} [properties] Extra input properties.
 * @returns {{field: HTMLElement, input: HTMLInputElement}} The field and its input.
 */
function labelledInput(id, label, properties = {}) {
  const input = el("input", { id, ...properties });
  return { field: el("div", { class: "field" }, el("label", { htmlFor: id }, label), input), input };
}

/**
 * Turn a file's name into one the API accepts.
 * Why: Files often have spaces or other characters, and a person should not have to rename them. The
 * API still checks the result, so this is only a convenience.
 * Used by: the document form.
 * @param {string} fileName The original name.
 * @returns {string} The suggested name.
 */
function suggestDocumentName(fileName) {
  const dot = fileName.lastIndexOf(".");
  const stem = (dot > 0 ? fileName.slice(0, dot) : fileName).replace(/[^A-Za-z0-9_-]+/g, "-");
  const ending = dot > 0 ? fileName.slice(dot).toLowerCase() : ".md";
  return `${stem.replace(/^[-_]+/, "").slice(0, 64) || "document"}${ending}`;
}

/**
 * Build the document half of the Insert Data page.
 * Why: Adding a document makes it searchable at once. Files are read in the browser and sent as text,
 * so there is no separate upload path to secure.
 * Used by: `renderInsert`.
 * @returns {HTMLElement[]} The elements.
 */
function documentSection() {
  const area = el("div", {});
  const list = el("div", {}, el("p", { class: "hint" }, "Loading…"));
  const submit = el("button", { type: "button", class: "button button-primary" }, "Add document");
  const name = labelledInput("document-name", "File name", { placeholder: "holiday-hours.md", maxLength: 67 });
  let send;
  const { field, input } = textField({ id: "document-text", label: "Text", limit: 50000, onSend: () => send() });
  const file = el("input", { id: "document-file", type: "file", accept: ".md,.txt" });

  const drawList = async () => {
    const result = await callApi("/documents", { background: true });
    if (!result.ok) {
      list.replaceChildren(notice("error", result.problem));
      return;
    }
    list.replaceChildren(
      el(
        "div",
        { class: "table-wrap" },
        el(
          "table",
          {},
          el("thead", {}, el("tr", {}, ...["Document", "Chunks", ""].map((label) => el("th", {}, label)))),
          el(
            "tbody",
            {},
            ...result.data.map((item) => {
              const remove = el("button", { type: "button", class: "button" }, "Remove");
              remove.onclick = async () => {
                const confirmed = await confirmAction(
                  "Remove this document?",
                  `${item.id} will be deleted from this computer, and can no longer be found.`,
                  "Yes, remove",
                );
                if (!confirmed) return;
                const removed = await callApi(`/documents/${encodeURIComponent(item.id.replace("uploads/", ""))}`, {
                  method: "DELETE",
                });
                area.replaceChildren(
                  removed.ok
                    ? notice("success", `The API confirmed that ${item.id} was removed.`)
                    : notice("error", removed.problem),
                );
                drawList();
                refreshConnection();
              };
              return el(
                "tr",
                {},
                el("td", { class: "mono" }, item.id),
                el("td", {}, String(item.chunks)),
                el("td", {}, item.removable ? remove : el("span", { class: "hint" }, "Part of the project")),
              );
            }),
          ),
        ),
      ),
    );
  };

  file.onchange = async () => {
    const chosen = file.files[0];
    if (!chosen) return;
    input.value = await chosen.text();
    input.dispatchEvent(new Event("input"));
    name.input.value = suggestDocumentName(chosen.name);
  };

  send = requestRunner({
    area,
    submit,
    send: async (signal) => {
      const body = { name: name.input.value.trim(), text: input.value };
      const result = await callApi("/documents", { method: "POST", body, signal });
      if (result.ok) {
        area.replaceChildren(
          notice("success", `The API confirmed that ${result.data.id} was added, in ${result.data.chunks} chunk(s). You can ask about it now.`),
        );
        name.input.value = "";
        input.value = "";
        file.value = "";
        input.dispatchEvent(new Event("input"));
        drawList();
        refreshConnection();
      } else {
        area.replaceChildren(notice(result.cancelled ? "warning" : "error", result.problem));
      }
      reveal(area);
    },
  });
  submit.onclick = send;
  drawList();
  return [
    el("h3", { class: "section-title" }, "Add a document"),
    el(
      "p",
      { class: "hint" },
      "Stored on this computer in a folder that Git ignores. When a question matches it, that part of the text is sent to the model provider, so do not add anything you may not share.",
    ),
    el(
      "div",
      { class: "form" },
      el("div", { class: "field" }, el("label", { htmlFor: "document-file" }, "Read from a file (optional, .md or .txt)"), file),
      name.field,
      el("span", { class: "hint" }, "Letters, numbers, dashes, and underscores, ending in .md or .txt."),
      field,
      el("div", { class: "actions" }, submit),
    ),
    area,
    list,
  ];
}

/**
 * Build the order half of the Insert Data page.
 * Why: The model only fills in the form. Nothing is stored until a person reads it and presses Save,
 * because a model can misread text, and pasted text may come from outside.
 * Used by: `renderInsert`.
 * @returns {HTMLElement[]} The elements.
 */
function orderSection() {
  const extractArea = el("div", {});
  const saveArea = el("div", {});
  const list = el("div", {}, el("p", { class: "hint" }, "Loading…"));
  const orderId = labelledInput("order-id", "Order id", { placeholder: "ORD-2001", maxLength: 32 });
  const customer = labelledInput("order-customer", "Customer", { maxLength: 100 });
  const item = labelledInput("order-item", "Item", { maxLength: 100 });
  const status = el(
    "select",
    { id: "order-status" },
    ...ORDER_STATUSES.map((value) => el("option", { value }, value)),
  );
  const save = el("button", { type: "button", class: "button button-primary" }, "Save order");
  const read = el("button", { type: "button", class: "button" }, "Fill the form with AI");
  let sendRead;
  const { field: textBox, input: freeText } = textField({
    id: "order-text",
    label: "Text that describes an order",
    limit: 2000,
    onSend: () => sendRead(),
  });

  const drawList = async () => {
    const result = await callApi("/records", { background: true });
    if (!result.ok) {
      list.replaceChildren(notice("error", result.problem));
      return;
    }
    list.replaceChildren(
      el(
        "div",
        { class: "table-wrap" },
        el(
          "table",
          {},
          el("thead", {}, el("tr", {}, ...["Order", "Customer", "Item", "Status"].map((label) => el("th", {}, label)))),
          el(
            "tbody",
            {},
            ...result.data.map((record) =>
              el(
                "tr",
                {},
                el("td", { class: "mono" }, record.id),
                el("td", {}, record.customer),
                el("td", {}, record.item),
                el("td", {}, record.status),
              ),
            ),
          ),
        ),
      ),
    );
  };

  sendRead = requestRunner({
    area: extractArea,
    submit: read,
    send: async (signal) => {
      const result = await callApi("/records/extract", {
        method: "POST",
        body: { text: freeText.value.trim() },
        signal,
      });
      if (!result.ok) {
        extractArea.replaceChildren(notice(result.cancelled ? "warning" : "error", result.problem));
        return;
      }
      if (!result.data.found) {
        extractArea.replaceChildren(notice("warning", result.data.message));
        return;
      }
      const { draft } = result.data;
      orderId.input.value = draft.id ?? "";
      customer.input.value = draft.customer ?? "";
      item.input.value = draft.item ?? "";
      if (draft.status) status.value = draft.status;
      extractArea.replaceChildren(
        el(
          "p",
          { class: "notice notice-info", role: "status" },
          aiLabel(),
          " The form was filled in by the model, and nothing has been saved. Check each field, then press Save order.",
        ),
      );
    },
  });
  read.onclick = sendRead;

  const sendSave = requestRunner({
    area: saveArea,
    submit: save,
    send: async (signal) => {
      const body = {
        id: orderId.input.value.trim(),
        customer: customer.input.value.trim(),
        item: item.input.value.trim(),
        status: status.value,
      };
      const result = await callApi("/records", { method: "POST", body, signal });
      if (result.ok) {
        saveArea.replaceChildren(notice("success", `The API confirmed that order ${result.data.id} was saved.`));
        for (const { input } of [orderId, customer, item]) input.value = "";
        drawList();
      } else {
        saveArea.replaceChildren(notice(result.cancelled ? "warning" : "error", result.problem));
      }
      reveal(saveArea);
    },
  });
  save.onclick = sendSave;
  drawList();
  return [
    el("h3", { class: "section-title" }, "Add an order"),
    el("p", { class: "hint" }, "Type the details, or let the model read them from text. This text is sent to the model provider."),
    el(
      "div",
      { class: "form" },
      textBox,
      exampleButtons(["Order ORD-2001 for Sample Customer D: 40 valves, already shipped."], freeText),
      el("div", { class: "actions" }, read),
      extractArea,
      orderId.field,
      customer.field,
      item.field,
      el("div", { class: "field" }, el("label", { htmlFor: "order-status" }, "Status"), status),
      el("div", { class: "actions" }, save),
    ),
    saveArea,
    list,
  ];
}

/** Build the Insert Data page. Why: Add documents and orders without a restart. Used by: PAGES. */
function renderInsert() {
  return [
    el("h2", {}, "Insert data"),
    el("p", { class: "lead" }, "Add documents the assistant can answer from, and orders it can look up."),
    ...documentSection(),
    ...orderSection(),
  ];
}

/**
 * Show the result of a connection check, on the Settings page and in the header.
 * Why: The answer must be visible from any page, and it must match what the API reported.
 * Used by: `renderSettings` and startup.
 * @param {object} report The API response.
 */
function showLlmStatus(report) {
  const badgeElement = document.querySelector("#llm-status");
  badgeElement.textContent = report.connected ? "Provider OK" : "Provider problem";
  badgeElement.className = `badge badge-${report.connected ? "success" : "error"}`;
}

/** Build the Settings page. Why: See the model and whether it really works. Used by: PAGES. */
function renderSettings() {
  const details = el("div", {}, el("p", { class: "hint" }, "Loading…"));
  const statusArea = el("div", {});
  const check = el("button", { type: "button", class: "button" }, "Check connection");
  const testMessage = el("button", { type: "button", class: "button" }, "Send a test message");

  const runCheck = (sendMessage) =>
    requestRunner({
      area: statusArea,
      submit: sendMessage ? testMessage : check,
      send: async (signal) => {
        const result = await callApi("/settings/llm/check", {
          method: "POST",
          body: { send_message: sendMessage },
          signal,
        });
        if (!result.ok) {
          statusArea.replaceChildren(notice(result.cancelled ? "warning" : "error", result.problem));
          return;
        }
        const report = result.data;
        showLlmStatus(report);
        // `el` skips empty values. `replaceChildren` would show `false` as text.
        statusArea.replaceChildren(
          el(
            "div",
            {},
            notice(report.connected ? "success" : "error", report.detail),
            report.message_sent &&
              el(
                "p",
                { class: "hint" },
                `Answered by ${report.model_answered}, using ${report.usage.input_tokens} input and ${report.usage.output_tokens} output tokens.`,
              ),
          ),
        );
        if (report.remote_models.length) drawDetails();
      },
    });

  const drawDetails = async () => {
    const result = await callApi("/settings/llm", { background: true });
    if (!result.ok) {
      details.replaceChildren(notice("error", result.problem));
      return;
    }
    const data = result.data;
    const facts = el(
      "dl",
      { class: "facts" },
      el("dt", {}, "Provider"),
      el("dd", {}, PROVIDER_NAMES[data.provider] ?? data.provider),
      el("dt", {}, "Model"),
      el("dd", { id: "current-model" }, data.model ?? "None, because the mock is not a real model"),
      el("dt", {}, "API key"),
      el("dd", {}, data.key_configured ? "Set in your .env file. It is never shown here." : "Not set"),
      el("dt", {}, "Address"),
      el("dd", { class: "mono" }, data.base_url ?? "None, it runs offline"),
      el("dt", {}, "Embeddings"),
      el("dd", {}, data.provider === "deepseek" ? "Local word matching, because DeepSeek lists no embeddings API" : data.embeddings),
    );
    const parts = [el("article", { class: "card" }, facts)];
    if (data.selectable) {
      const select = el(
        "select",
        { id: "model-select" },
        ...data.models.map((model) => el("option", { value: model, selected: model === data.model }, model)),
      );
      const save = el("button", { type: "button", class: "button button-primary" }, "Use this model");
      const saved = el("div", {});
      save.onclick = async () => {
        save.disabled = true;
        const response = await callApi("/settings/llm", { method: "PUT", body: { model: select.value } });
        save.disabled = false;
        if (response.ok) {
          await drawDetails();
          statusArea.replaceChildren(
            notice("success", `The API confirmed that ${response.data.model} is now in use, and will stay selected after a restart.`),
          );
          refreshConnection();
        } else {
          saved.replaceChildren(notice("error", response.problem));
        }
      };
      parts.push(
        el(
          "div",
          { class: "form" },
          el("div", { class: "field" }, el("label", { htmlFor: "model-select" }, "Model"), select),
          el("div", { class: "actions" }, save),
          saved,
        ),
      );
    }
    if (data.provider === "mock") {
      parts.push(
        notice(
          "info",
          "To use DeepSeek, copy .env.example to .env, paste your key after AI_OPS_LLM_API_KEY=, and restart the API. A key cannot be entered on this page, because this API has no login.",
        ),
      );
    }
    details.replaceChildren(...parts);
  };

  check.onclick = runCheck(false);
  testMessage.onclick = runCheck(true);
  drawDetails();
  check.onclick();
  return [
    el("h2", {}, "Settings"),
    el("p", { class: "lead" }, "Which model answers, and whether the connection really works."),
    details,
    el("h3", { class: "section-title" }, "Connection"),
    el(
      "p",
      { class: "hint" },
      "Check connection is free, and proves that the address and key work. It cannot see an empty balance: only a test message can, and that uses a few tokens.",
    ),
    el("div", { class: "actions" }, check, testMessage),
    el("div", { style: "margin-top: 12px" }, statusArea),
  ];
}

/** Build the Diagnostics page. Why: What each request did and cost. Used by: PAGES. */
function renderDiagnostics() {
  const rows = session.requests.map((entry) =>
    el(
      "tr",
      {},
      el("td", {}, entry.time),
      el("td", { class: "mono" }, entry.request),
      el("td", {}, String(entry.status)),
      el("td", {}, entry.latency !== null ? `${entry.latency} ms` : `${entry.browserMs} ms (browser)`),
      el("td", {}, entry.usage ? `${entry.usage.input_tokens} in, ${entry.usage.output_tokens} out` : "None reported"),
      el("td", { class: "mono" }, entry.requestId ?? "None"),
    ),
  );
  const toolCalls = session.requests.flatMap((entry) => entry.toolCalls.map((call) => ({ entry, call })));
  return [
    el("h2", {}, "Diagnostics"),
    el(
      "p",
      { class: "lead" },
      "Requests from this browser session, newest first. They are cleared when the page reloads. Use the request id to find the matching lines in the API's logs. With the mock provider, tokens are word counts. Cost is not shown here, because it needs your provider's prices.",
    ),
    el("h3", { class: "section-title" }, "Tool calls"),
    toolCalls.length
      ? el(
          "div",
          {},
          ...toolCalls.map(({ entry, call }) =>
            el("p", { class: "mono" }, `${entry.time} ${call.name} ${JSON.stringify(call.arguments)} `, badge(toolStatus(call.status))),
          ),
        )
      : el("p", { class: "empty" }, "No tool calls yet. Try the Chat page."),
    el("h3", { class: "section-title" }, "Requests"),
    rows.length
      ? el(
          "div",
          { class: "table-wrap" },
          el(
            "table",
            {},
            el(
              "thead",
              {},
              el("tr", {}, ...["Time", "Request", "Status", "Latency", "Tokens", "Request id"].map((name) => el("th", {}, name))),
            ),
            el("tbody", {}, ...rows),
          ),
        )
      : el("p", { class: "empty" }, "No requests yet."),
  ];
}

/** Build the About page. Why: States the limits honestly. Used by: PAGES. */
function renderAbout() {
  const limits = [
    "This is an educational project with synthetic data. It has not been deployed to production.",
    "With the mock provider, answers only repeat your text. Retrieval, tools, approvals, and workflows are real.",
    "It has only had a small evaluation with one real model, on synthetic data. See docs/EVALUATION.md for the limits.",
    "There is no login. Anyone who can reach the API can read orders and approve actions, so keep it on this computer.",
    "Approvals and workflow progress are lost when the API restarts. Orders are kept.",
    "Messages go through a stub by default, so nothing leaves the application.",
  ];
  return [
    el("h2", {}, "About"),
    el("p", { class: "lead" }, "A simple interface over the same HTTP API that any other client uses."),
    el("h3", { class: "section-title" }, "Known limits"),
    el("ul", { class: "plain-list" }, ...limits.map((text) => el("li", {}, text))),
    el("h3", { class: "section-title" }, "For developers"),
    el("p", {}, el("a", { href: "/docs" }, "Interactive API docs"), " (needs internet access for its interface)."),
  ];
}

/**
 * Show the page named in the URL.
 * Why: Keeping the page in the URL hash makes Back and reload work, without any router library. After
 * a page change, focus moves to the content, so a keyboard or screen reader user lands on it. On first
 * load, focus stays at the top, or Tab would skip the navigation.
 * Used by: the `hashchange` listener and startup.
 * @param {boolean} moveFocus Whether to move focus to the new content.
 */
function showPage(moveFocus) {
  const id = location.hash.slice(1);
  const current = PAGES.find((page) => page.id === id) ?? PAGES[0];
  const container = document.querySelector("#page");
  container.replaceChildren(...current.render());
  container.scrollTop = 0;
  document.title = `${current.label} - AI Operations Assistant`;
  for (const button of document.querySelectorAll(".nav-button")) {
    if (button.dataset.page === current.id) button.setAttribute("aria-current", "page");
    else button.removeAttribute("aria-current");
  }
  if (moveFocus) container.focus({ preventScroll: true });
}

/**
 * Build the navigation.
 * Why: Buttons change the hash, and `showPage` does the rest, so there is one path to each page.
 * Used by: startup.
 */
function buildNavigation() {
  const list = document.querySelector("#navigation");
  for (const page of PAGES) {
    const button = el("button", { type: "button", class: "nav-button" }, page.label);
    button.dataset.page = page.id;
    if (page.id === "approvals") button.append(el("span", { class: "count", id: "pending-count", hidden: true }));
    button.onclick = () => {
      location.hash = page.id;
    };
    list.append(el("li", {}, button));
  }
}

/**
 * Update the number of waiting approvals in the navigation.
 * Why: A proposed action is easy to miss, so the count stays visible on every page.
 * Used by: every place where approvals can change, and startup.
 */
async function refreshPendingCount() {
  const count = document.querySelector("#pending-count");
  try {
    const response = await fetch("/approvals");
    const approvals = await response.json();
    const waiting = approvals.filter((item) => item.status === "pending").length;
    count.textContent = String(waiting);
    count.hidden = waiting === 0;
    count.setAttribute("aria-label", `${waiting} waiting`);
  } catch {
    count.hidden = true;
  }
}

/**
 * Show whether the API is reachable, and which provider it uses.
 * Why: A person must be able to see at a glance that answers come from a mock.
 * Used by: startup, and then on a timer.
 */
async function refreshConnection() {
  const badgeElement = document.querySelector("#connection");
  const mockNotice = document.querySelector("#mock-notice");
  try {
    const response = await fetch("/health");
    const health = await response.json();
    const provider = PROVIDER_NAMES[health.provider] ?? health.provider;
    const settings = await (await fetch("/settings/llm")).json();
    const model = settings.model ? ` • ${settings.model}` : "";
    badgeElement.textContent = `${provider}${model} • ${health.documents} documents • Connected`;
    badgeElement.className = "badge badge-success";
    mockNotice.hidden = health.provider !== "mock";
  } catch {
    badgeElement.textContent = "Not connected";
    badgeElement.className = "badge badge-error";
  }
}

buildNavigation();
window.addEventListener("hashchange", () => showPage(true));
showPage(false);
refreshConnection();
refreshPendingCount();
// This request is free, and only runs once per page load, so it costs nothing.
callApi("/settings/llm/check", { method: "POST", body: {}, background: true }).then((result) => {
  if (result.ok) showLlmStatus(result.data);
});
setInterval(refreshConnection, 15_000);
