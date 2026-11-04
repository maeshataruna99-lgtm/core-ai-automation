"""LLM layer: the provider interface, its implementations, and prompt templates.

Why: Services talk to models only through this package, so a model vendor can be replaced
without touching business logic.

Used by: `ai_ops.api.app.create_app` today, and the services from Milestone 3 onwards.
"""
