# API guide

The API listens on `http://127.0.0.1:8000`. The web interface is at `/`, which redirects to `/ui/`. While it runs, interactive docs are at `/docs`. That page loads its interface from a public CDN, so it needs internet access. The same contract is stored in `docs/openapi.json`, which needs nothing to read.

Examples show real responses from the mock provider. Ids and timings differ on each run, and the mock provider echoes questions instead of answering them. Long text is shortened with `...`.

Each example is shown for bash on Linux. On Windows, `curl` in Windows PowerShell 5.1 is an alias for a different command, so use the PowerShell version. Both forms work the same way:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/ask -ContentType application/json -Body '{"question": "How long do card refunds take?"}' | ConvertTo-Json -Depth 6
```

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Check that the API is up, and how many documents loaded |
| POST | `/ask` | Answer a question from the knowledge base, with sources |
| POST | `/chat` | Run the agent, which can use tools |
| POST | `/workflows/ticket` | Run the ticket workflow, or read a ticket's state |
| GET | `/approvals` | List all approvals, newest first |
| POST | `/approvals/{approval_id}` | Approve or reject a pending action |
| GET | `/records/{record_id}` | Read one order |
| GET, POST | `/records` | List orders, or add one |
| POST | `/records/extract` | Read a draft order from text with the model. This saves nothing |
| GET, POST | `/documents` | List knowledge base documents, or add one |
| DELETE | `/documents/{name}` | Remove an added document |
| GET, PUT | `/settings/llm` | See the provider, or choose the model |
| POST | `/settings/llm/check` | Check that the provider really works |

## Check the API

```bash
curl http://127.0.0.1:8000/health
```

```json
{"status": "ok", "documents": 5, "provider": "mock"}
```

## Ask a question

```bash
curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" -d '{"question": "How long do card refunds take?"}'
```

```json
{
  "answer": "[mock] How long do card refunds take?",
  "found": true,
  "sources": [
    {
      "id": "refund-policy.md#0",
      "document_id": "refund-policy.md",
      "position": 0,
      "score": 0.2227,
      "text": "# Refund policy ... Card refunds take up to 10 business days."
    }
  ],
  "usage": {"input_tokens": 188, "output_tokens": 7},
  "latency_ms": 0.4
}
```

When nothing relevant is found, the response is still a 200, with `found` set to `false`, an empty `sources`, and the answer `I could not find this in the knowledge base.` The model is not called in that case.

## Use the agent

```bash
curl -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" -d '{"message": "Please update ORD-1002 to shipped"}'
```

```json
{
  "reply": "[mock] {\"status\": \"pending_approval\", \"approval_id\": \"0f89...\", \"message\": \"Nothing has changed yet. A person must approve this action.\"}",
  "stop_reason": "completed",
  "tool_calls": [
    {
      "name": "update_record",
      "arguments": {"record_id": "ORD-1002", "status": "shipped"},
      "status": "pending_approval",
      "approval_id": "0f89..."
    }
  ],
  "pending_approvals": ["0f89..."],
  "usage": {"input_tokens": 189, "output_tokens": 16},
  "latency_ms": 5.4
}
```

Read `tool_calls` and `pending_approvals` to see what the agent did. The `reply` comes from a model, which can be wrong about its own actions. `stop_reason` is `completed`, `max_steps`, or `timeout`.

## Approve or reject an action

Use an id from `pending_approvals`. `decision` is `approve` or `reject`.

```bash
curl -X POST http://127.0.0.1:8000/approvals/APPROVAL_ID -H "Content-Type: application/json" -d '{"decision": "approve"}'
```

```json
{
  "id": "0f89...",
  "kind": "update_record",
  "payload": {"record_id": "ORD-1002", "status": "shipped"},
  "status": "executed",
  "result": {"id": "ORD-1002", "status": "shipped"},
  "created_at": "2026-11-04T10:49:18+00:00"
}
```

Approving twice is safe, and the action only runs once. A failed action can be retried by approving again. A decision cannot be reversed.

`GET /approvals` returns a list of these objects, newest first. `status` is `pending`, `executing`, `executed`, `rejected`, or `failed`. The list is kept in memory, so it is empty after a restart.

## Read an order

```bash
curl http://127.0.0.1:8000/records/ORD-1002
```

```json
{"id": "ORD-1002", "customer": "Sample Customer B", "item": "Bearings", "status": "shipped"}
```

The sample orders are `ORD-1001`, `ORD-1002`, and `ORD-1003`.

## Insert data

Add a document, which can be found at once. It is stored in `data/uploads/`, which Git ignores. `name` may hold letters, digits, dashes, and underscores, and must end in `.md` or `.txt`. `text` can be up to 50,000 characters. A repeated name returns a 409, and nothing is replaced.

```bash
curl -X POST http://127.0.0.1:8000/documents -H "Content-Type: application/json" -d '{"name": "holiday-hours.md", "text": "The warehouse closes on Friday afternoons."}'
```

```json
{"id": "uploads/holiday-hours.md", "chunks": 1, "removable": true}
```

Add an order. A repeated id returns a 409. To change an existing order, use the agent and an approval.

```bash
curl -X POST http://127.0.0.1:8000/records -H "Content-Type: application/json" -d '{"id": "ORD-2001", "customer": "Sample Customer D", "item": "Valves", "status": "processing"}'
```

Read a draft from text. This sends the text to the provider, and saves nothing. `found` is `false` when no order could be read, which is what the mock provider always gives. Any field in `draft` can be `null`.

```bash
curl -X POST http://127.0.0.1:8000/records/extract -H "Content-Type: application/json" -d '{"text": "Order ORD-2001 for Sample Customer D: 40 valves, already shipped."}'
```

## Settings

`GET /settings/llm` shows the provider and model, and whether a key is set. It never shows the key.

```json
{
  "provider": "deepseek",
  "model": "deepseek-flash",
  "models": ["deepseek-flash", "deepseek-v4-pro"],
  "selectable": true,
  "key_configured": true,
  "embeddings": "local-word-hash",
  "base_url": "https://api.deepseek.com"
}
```

Choose a model. The next model call uses it, and it stays selected after a restart.

```bash
curl -X PUT http://127.0.0.1:8000/settings/llm -H "Content-Type: application/json" -d '{"model": "deepseek-v4-pro"}'
```

Check the connection. This is free, and calls the provider's model list. Add `"send_message": true` to also send one short message, which uses a few tokens, and is the only way to detect an empty balance. A failure is still a 200, with `connected` set to `false`.

```bash
curl -X POST http://127.0.0.1:8000/settings/llm/check -H "Content-Type: application/json" -d '{}'
```

```json
{
  "connected": false,
  "detail": "The API key was rejected. Check the key in your .env file, then restart the API.",
  "remote_models": [],
  "message_sent": false,
  "model_answered": null,
  "usage": null
}
```

## Run the ticket workflow

```bash
curl -X POST http://127.0.0.1:8000/workflows/ticket -H "Content-Type: application/json" -d '{"ticket_id": "T-1", "message": "Where is my delivery for ORD-1001?"}'
```

```json
{
  "ticket_id": "T-1",
  "status": "pending_approval",
  "category": "delivery",
  "record_id": "ORD-1001",
  "draft": "[mock] Where is my delivery for ORD-1001?",
  "approval_id": "25cf..."
}
```

Approve it with `/approvals/APPROVAL_ID` to send the reply. Posting the same ticket again is safe: it continues after a failure, or else reports the current state. `status` is `pending_approval`, `sending`, `sent`, `rejected`, or `send_failed`.

## Errors

Every error has the same shape, and `request_id` matches the `X-Request-ID` response header and the logs.

```json
{
  "error": {
    "code": "validation_error",
    "message": "The request is invalid.",
    "request_id": "a232...",
    "details": [{"field": "body.question", "message": "String should have at least 1 character"}]
  }
}
```

| Status | Code | Meaning |
|---|---|---|
| 404 | `record_not_found`, `approval_not_found`, `document_not_found`, `not_found` | The record, approval, document, or route does not exist |
| 409 | `approval_conflict`, `ticket_conflict`, `record_exists`, `document_exists`, `model_not_selectable` | A decision was reversed, an id or name is already used, or this provider's model cannot be changed |
| 413 | Depends on the Python version | The body is larger than 256 KiB |
| 422 | `validation_error`, `unknown_model` | The input is invalid, or the model is not on the list |
| 500 | `internal_error` | Unexpected failure. Details are only in the logs |
| 502 | `upstream_error`, `action_failed` | The LLM provider failed, or an approved action failed |
| 504 | `upstream_timeout` | The LLM provider did not answer in time |

## Update the OpenAPI file

After changing an endpoint, run this from the repository root. A test fails if the file does not match the code.

```bash
python -c "import json; from ai_ops.api.app import create_app; from ai_ops.config import Settings; open('docs/openapi.json', 'w', encoding='utf-8', newline='\n').write(json.dumps(create_app(Settings(llm_provider='mock')).openapi(), indent=2) + '\n')"
```
