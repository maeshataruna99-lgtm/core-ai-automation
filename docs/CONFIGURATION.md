# Configuration

Settings come from a local `.env` file and from environment variables, and real environment variables win. The API exits with a clear message when a required variable is missing or a value is invalid. That message never repeats the value.

## Use DeepSeek

You need a key from the [DeepSeek console](https://platform.deepseek.com/api_keys). Using it costs money, and your questions and the retrieved document text are sent to DeepSeek, so only use it with data you may share. Do not use real customer data.

1. Create your local settings file from the repository root.

   Windows (PowerShell):

   ```powershell
   Copy-Item .env.example .env
   notepad .env
   ```

   Linux:

   ```bash
   cp .env.example .env
   ```

2. Paste your key after `AI_OPS_LLM_API_KEY=`, and save the file.
3. Run `python -m ai_ops`. The header of the web interface now says "Real provider".

Only the key is needed. The URL `https://api.deepseek.com` and the model `deepseek-flash` are the defaults, and thinking mode is switched off because the agent does not keep the reasoning text that it would need to send back.

DeepSeek's documentation lists no embeddings API, so retrieval runs locally with the same word matching as the mock provider. Answers come from DeepSeek, but retrieval quality does not improve.

## Keep keys secret

- **Only put keys in `.env`.** Git ignores it, and it stays on your computer. `.env.example` is committed, so it must never hold a value.
- **Do not paste a key into a chat, an issue, or a screenshot,** and avoid typing it in a shell, where it stays in the command history.
- **Turn on the commit check once per clone.** After this, a commit that holds a secret or a `.env` file is stopped, and the output never shows the value:

  ```bash
  git config core.hooksPath .githooks
  ```

- **To scan by hand,** run `python scripts/check_secrets.py`. CI also scans the whole history on each push with `gitleaks`.
- **If a key is ever exposed, revoke it in the DeepSeek console and create a new one.** Deleting it from a file or a later commit is not enough, because Git history and any copies keep it.

## All settings

To use another settings file, set `AI_OPS_ENV_FILE` to its path. Set it to an empty value to ignore `.env`, which is what `scripts/demo.py` does, so that the demo can never use a real key.

## LLM provider

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `AI_OPS_LLM_PROVIDER` | Yes | | `mock` (offline, no key), `deepseek`, or `openai_compatible` |
| `AI_OPS_LLM_BASE_URL` | With `openai_compatible` | `https://api.deepseek.com` with `deepseek` | API root. It must use https, except to this computer |
| `AI_OPS_LLM_API_KEY` | With `deepseek` or `openai_compatible` | | Provider key |
| `AI_OPS_LLM_CHAT_MODEL` | With `openai_compatible` | `deepseek-flash` with `deepseek` | Model for text generation |
| `AI_OPS_LLM_EMBEDDING_MODEL` | With `openai_compatible` | Always local with `deepseek` | Model for embeddings |
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
| `AI_OPS_OUTBOUND_BASE_URL` | With `http` | | API root of the messaging service. It must use https, except to this computer |
| `AI_OPS_OUTBOUND_API_KEY` | With `http` | | Messaging service key |
| `AI_OPS_OUTBOUND_TIMEOUT_SECONDS` | No | `10` | Time limit for each network step of a send |

## Server and logs

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `AI_OPS_HOST` | No | `127.0.0.1` | Bind address. The API has no login, so keep this default |
| `AI_OPS_PORT` | No | `8000` | Bind port |
| `AI_OPS_LOG_LEVEL` | No | `INFO` | `DEBUG`, `INFO`, `WARNING`, or `ERROR` |
