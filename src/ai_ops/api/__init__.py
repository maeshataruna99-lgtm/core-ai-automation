"""HTTP layer: the FastAPI app, routes, schemas, middleware, and error mapping.

Why: This layer only translates between HTTP and the application. Business logic goes in
`ai_ops.services`, so it can be tested and reused without a web server.

Used by: `ai_ops.__main__`, which runs the app, and `tests/test_api.py`.
"""
