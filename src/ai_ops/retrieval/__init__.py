"""Retrieval layer: document loading, chunking, and vector search.

Why: These parts do not depend on HTTP or on a model vendor, so each can be tested alone
and replaced alone.

Used by: `ai_ops.services.rag` and `ai_ops.api.app`.
"""
