# Evaluation

The evaluation runs a fixed question set through the same RAG service as `POST /ask`, and reports retrieval quality, answer quality, cost, and latency.

**What these numbers are:** the results below come from the mock provider, whose embeddings only match exact words. They measure this retrieval pipeline on this small synthetic knowledge base. They are not the accuracy of an AI model, and they do not predict results on real data.

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

`data/evaluation/golden.json` holds 24 questions the knowledge base can answer, 8 it cannot, and 2 prompt injection attacks. The questions were written before the first run and were not changed after seeing the results. They were written by the same person who wrote the documents, so they may be easier than real questions.

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

Metrics that need a real model print "not measured" with the mock, because it only echoes the question. Scoring that echo would produce numbers that look real and mean nothing. The mock's token counts are word counts, so only compare them with each other. Its latency does not include a model call, and it says nothing about real response times.

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

## With a real model

This is needed to close the open prompt injection criterion, and to measure answer quality. Set the `AI_OPS_LLM_*` variables from the README, then run:

```bash
python -m ai_ops.evaluation --label real-model --output-dir docs/evaluation/runs
```

To get a cost estimate, add `--input-price-per-million` and `--output-price-per-million` with your provider's current prices. There are no default prices, because they differ by vendor and change over time.

A real run has not been done yet. Its thresholds in `data/evaluation/thresholds.json` are empty until there are real numbers to base them on.
