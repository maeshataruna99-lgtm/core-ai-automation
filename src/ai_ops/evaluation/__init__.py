"""Evaluation: measure retrieval, answers, cost, and latency against a fixed question set.

Why: AI behaviour changes when a prompt, a model, or a chunk size changes, and normal unit tests do
not notice. A golden set turns "it seems fine" into a number that can be tracked and enforced.

Used by: `python -m ai_ops.evaluation`, locally and in `.github/workflows/ci.yml`.
"""
