# API guide

The API listens on `http://127.0.0.1:8000`. While it runs, interactive docs are at `/docs`. That page loads its interface from a public CDN, so it needs internet access. The same contract is stored in `docs/openapi.json`, which needs nothing to read.

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
| POST | `/approvals/{approval_id}` | Approve or reject a pending action |
| GET | `/records/{record_id}` | Read one order |

## Check the API

```bash
curl http://127.0.0.1:8000/health
```

```json
{"status": "ok", "documents": 5}
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
  "result": {"id": "ORD-1002", "status": "shipped"}
}
```

Approving twice is safe, and the action only runs once. A failed action can be retried by approving again. A decision cannot be reversed.

## Read an order

```bash
curl http://127.0.0.1:8000/records/ORD-1002
```

```json
{"id": "ORD-1002", "customer": "Sample Customer B", "item": "Bearings", "status": "shipped"}
```

The sample orders are `ORD-1001`, `ORD-1002`, and `ORD-1003`.

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
| 404 | `record_not_found`, `approval_not_found`, `not_found` | The record, approval, or route does not exist |
| 409 | `approval_conflict`, `ticket_conflict` | A decision was reversed, or a ticket id was reused with different text |
| 413 | Depends on the Python version | The body is larger than 64 KiB |
| 422 | `validation_error` | The input is invalid |
| 500 | `internal_error` | Unexpected failure. Details are only in the logs |
| 502 | `upstream_error`, `action_failed` | The LLM provider failed, or an approved action failed |
| 504 | `upstream_timeout` | The LLM provider did not answer in time |

## Update the OpenAPI file

After changing an endpoint, run this from the repository root. A test fails if the file does not match the code.

```bash
python -c "import json; from ai_ops.api.app import create_app; from ai_ops.config import Settings; open('docs/openapi.json', 'w', encoding='utf-8', newline='\n').write(json.dumps(create_app(Settings(llm_provider='mock')).openapi(), indent=2) + '\n')"
```
