# Configuration

All settings come from environment variables. The API exits with a clear message when a required variable is missing or a value is invalid. Keys belong only in the environment, never in a file in this repository.

Set a variable before starting the API:

```powershell
$env:AI_OPS_LLM_PROVIDER = "mock"
```

```bash
export AI_OPS_LLM_PROVIDER=mock
```

## LLM provider

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `AI_OPS_LLM_PROVIDER` | Yes | | `mock` (offline, no key) or `openai_compatible` |
| `AI_OPS_LLM_BASE_URL` | With `openai_compatible` | | API root, such as `https://api.example.com/v1` |
| `AI_OPS_LLM_API_KEY` | With `openai_compatible` | | Provider key |
| `AI_OPS_LLM_CHAT_MODEL` | With `openai_compatible` | | Model for text generation |
| `AI_OPS_LLM_EMBEDDING_MODEL` | With `openai_compatible` | | Model for embeddings |
| `AI_OPS_LLM_TIMEOUT_SECONDS` | No | `30` | Time limit for each network step of one attempt |
| `AI_OPS_LLM_MAX_RETRIES` | No | `2` | Retries for temporary failures, from 0 to 5 |

## Knowledge base and retrieval

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `AI_OPS_KNOWLEDGE_DIR` | No | `data/knowledge` | Directory of `.md` and `.txt` documents, loaded at startup |
| `AI_OPS_RAG_CHUNK_SIZE` | No | `800` | Maximum characters per chunk |
| `AI_OPS_RAG_CHUNK_OVERLAP` | No | `100` | Characters repeated between chunks |
| `AI_OPS_RAG_TOP_K` | No | `2` | Maximum chunks used per question |
| `AI_OPS_RAG_MIN_SCORE` | No | `0.10` | Lowest similarity counted as relevant. Measured with the mock provider, so measure it again for a real embedding model |

## Agent

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `AI_OPS_AGENT_MAX_STEPS` | No | `5` | Maximum model calls for one chat message, from 1 to 20 |
| `AI_OPS_AGENT_TIMEOUT_SECONDS` | No | `60` | Time limit for one chat message |

## Storage and outbound messages

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `AI_OPS_DATABASE_PATH` | No | `data/ai_ops.db` | SQLite file for orders. Created on first start. Delete it to return to the sample data |
| `AI_OPS_OUTBOUND_PROVIDER` | No | `stub` | `stub` (sends nothing) or `http` |
| `AI_OPS_OUTBOUND_BASE_URL` | With `http` | | API root of the messaging service |
| `AI_OPS_OUTBOUND_API_KEY` | With `http` | | Messaging service key |
| `AI_OPS_OUTBOUND_TIMEOUT_SECONDS` | No | `10` | Time limit for each network step of a send |

## Server and logs

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `AI_OPS_HOST` | No | `127.0.0.1` | Bind address. The API has no login, so keep this default |
| `AI_OPS_PORT` | No | `8000` | Bind port |
| `AI_OPS_LOG_LEVEL` | No | `INFO` | `DEBUG`, `INFO`, `WARNING`, or `ERROR` |
