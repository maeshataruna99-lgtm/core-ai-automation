# Evaluation

The evaluation runs a fixed question set through the same RAG service as `POST /ask`, and reports retrieval quality, answer quality, cost, and latency.

**What these numbers are:** they measure this pipeline on a small synthetic knowledge base, with retrieval that only matches exact words. The questions were written by the author of the documents, and the settings were tuned on the same questions. They do not predict results on real data.

## Run it

Windows (PowerShell):

```powershell
$env:AI_OPS_LLM_PROVIDER = "mock"
python -m ai_ops.evaluation
```

Linux:

```bash
AI_OPS_LLM_PROVIDER=mock python -m ai_ops.evaluation
```

The command prints a report and stores it as JSON in `evaluation-runs/`, which Git ignores. Use `--output-dir` and `--label` for a run worth keeping. CI runs the same command on Windows and Linux.

| Exit code | Meaning |
|---|---|
| 0 | Passed |
| 1 | A score is below its minimum in `data/evaluation/thresholds.json` |
| 2 | The run could not complete, for example because of a missing setting |

## The question set

`data/evaluation/golden.json` holds 24 questions the knowledge base can answer, 8 it cannot, and 2 prompt injection attacks. The questions were written before the first run and were not changed after seeing the results. One expected fact was corrected later, because `warranty-wear` repeated its answer in the question, which would let a model pass by echoing it. Retrieval scores did not change, but the stored reports hold the earlier fingerprint. They were written by the same person who wrote the documents, so they may be easier than real questions.

## Metrics

| Metric | Question it answers | Needs a real model |
|---|---|---|
| `hit_rate` | Did the right document appear in the results? | No |
| `precision` | How much of what was retrieved is relevant? | No |
| `recall` | How much of what is relevant was retrieved? | No |
| `context_fact_recall` | Does the retrieved text contain the expected fact? | No |
| `not_found_accuracy` | Does an unanswerable question get "not found"? | No |
| `answer_fact_recall` | Does the answer state the expected fact? | Yes |
| `groundedness` | How many of the answer's words appear in the retrieved text? | Yes |
| `citation_rate` | Does the answer cite a retrieved source? | Yes |
| `injection_resistance` | How many attacks did the model ignore? | Yes |

Metrics that need a real model print "not measured" with the mock provider, because it only echoes the question. Scoring that echo would produce numbers that look real and mean nothing. The mock's token counts are word counts, so only compare them with each other. Its latency does not include a model call, and it says nothing about real response times.

These checks have known limits. Facts are matched as exact phrases, so a correct answer in other words counts as a miss. `groundedness` only compares words, so it cannot detect a false statement made from words that are in the context.

## Optimisation

The goal was to make sure the model sees the facts it needs, without paying for more text than necessary. Retrieval settings were measured on the question set, with chunk size 800 and overlap 100 throughout. Each row is a stored report in `docs/evaluation/runs/`, recorded on 2026-11-04 with the mock provider.

| Step | `top_k` | `min_score` | `hit_rate` | `precision` | `recall` | `context_fact_recall` | `not_found_accuracy` | Input tokens |
|---|---|---|---|---|---|---|---|---|
| 1. Baseline | 4 | 0.15 | 0.792 | 0.628 | 0.792 | 0.792 | 0.625 | 4913 |
| 2. Fewer chunks | 2 | 0.15 | 0.792 | 0.646 | 0.792 | 0.792 | 0.625 | 4630 |
| 3. Lower threshold | 2 | 0.10 | 0.958 | 0.667 | 0.958 | 0.958 | 0.500 | 6183 |

**Step 2 has no downside.** No score dropped, precision rose, and input tokens fell by 5.8%.

**Step 3 is a trade-off.** The share of answerable questions whose fact reached the model rose from 19 of 24 to 23 of 24. In exchange, 4 of 8 unanswerable questions now reach the model, up from 3 of 8, and input tokens rose by 33.5% compared with step 2.

This trade-off was accepted because a missing fact cannot be fixed later: the model never sees it. An irrelevant document can still be handled by the model, since the prompt tells it to say when the context has no answer. That second part has not been tested with a real model, so step 3 needs measuring again with one.

The defaults in `src/ai_ops/config.py` use step 3. The minimums in `data/evaluation/thresholds.json` are set to those measured scores, so CI fails on any change that lowers them.

### Limits of this result

- **It was tuned on the set it was tested on.** There is no separate test set, so these scores are likely better than results on new questions.
- **It does not carry over to real embeddings.** `min_score` depends on the embedding model and must be measured again for any real one.
- **The sample is small.** With 24 and 8 cases, one question moves a score by 0.042 or 0.125.

### Remaining misses at the chosen settings

- `shipping-cutoff`: the question shares too few exact words with the shipping document, so the wrong document ranks first.
- `none-ceo`, `none-singapore`, `none-crypto`, and `none-loyalty`: these share ordinary words with a document, such as "company" and "ship", so the exact-word mock counts them as matches.

## Results with DeepSeek

Three runs on 2026-11-04 with `deepseek-flash`, at `top_k` 2 and `min_score` 0.10. Each is a stored report in `docs/evaluation/runs/`, including every answer. Embeddings are local word matching, because DeepSeek's documentation lists no embeddings API, so the retrieval scores match the mock's.

| Metric | Run 1 | Run 2 | Run 3 |
|---|---|---|---|
| `answer_fact_recall` | 0.958 | 0.958 | 0.958 |
| `citation_rate` | 0.958 | 1.000 | 0.958 |
| `groundedness` | 0.924 | 0.916 | 0.910 |
| `injection_resistance` (2 attacks) | 1.000 | 1.000 | 1.000 |
| `latency_p50_ms` | 852 | 762 | 824 |
| `latency_p95_ms` | 1206 | 1140 | 1153 |
| `estimated_cost` (US dollars) | 0.0034 | 0.0035 | 0.0034 |

The cost uses the peak `deepseek-flash` prices listed in DeepSeek's documentation on that date, $0.30 per million input tokens with no cache hit and $1.20 per million output tokens. Input tokens include local word counts for embeddings, so this is a rough upper estimate.

What reading the answers showed:

- **Unanswerable questions:** 4 of 8 pass the relevance threshold and reach the model, and it said it could not find the answer in all 12 of those answers across the three runs. So the trade-off in step 3 above held here. This is not a metric: `not_found_accuracy` stays at 0.500, because it measures retrieval, and these were read by a person.
- **The one miss:** `shipping-cutoff` is the same in each run. Retrieval returns the wrong document, and the model said that it could not find the answer. In run 2 it cited a source while saying so, which is why that run's citation rate is 1.000. It is a retrieval failure, and not an invented answer.
- **Prompt injection:** the model ignored both document attacks in all three runs, so 6 of 6 tries. That is a small sample of two simple attacks, on one model. It does not show that a determined attacker would fail, and attacks through tool results were not tested. Separately, `tests/test_agent.py` shows that even a fully manipulated model cannot change data without approval.
- **`groundedness`** is below 1 mainly because normal linking words count as unsupported, so it cannot be read as 8% invented text.

The answers were not checked by anyone other than the author, and answer quality on real questions is unknown. Answer metrics are not in CI, because CI has no key, and they vary between runs.

## With a real model

This is needed to close the open prompt injection criterion, and to measure answer quality. Set up your key as described in `docs/CONFIGURATION.md`, then run:

```bash
python -m ai_ops.evaluation --label deepseek --output-dir docs/evaluation/runs
```

To get a cost estimate, add `--input-price-per-million` and `--output-price-per-million` with your provider's current prices. There are no default prices, because they differ by vendor and change over time. The token counts of a real run are the provider's, except that local embeddings count words.

With DeepSeek, retrieval still uses local word matching, so its retrieval thresholds match the mock's. The run measures answers, citations, and prompt injection resistance.

Thresholds for the answer metrics are left out of `data/evaluation/thresholds.json`, because three runs are too few to set a fair minimum.
