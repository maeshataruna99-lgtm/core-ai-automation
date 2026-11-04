"""FastAPI application factory.

Why: A factory builds the app from settings passed as an argument. Importing this module
has no side effects, and each test can build its own app without environment variables.

Used by: `ai_ops.__main__.main` and `tests/test_api.py`.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from ai_ops.api.errors import register_error_handlers
from ai_ops.api.middleware import RequestContextMiddleware
from ai_ops.api.routes import router
from ai_ops.config import Settings
from ai_ops.llm.factory import build_provider
from ai_ops.llm.provider import LLMProvider
from ai_ops.logging import configure_logging


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Close the LLM provider when the app stops.

    Why: FastAPI runs the code after `yield` at shutdown, which is the right moment to
    release the provider's open connections.

    Used by: FastAPI, through the `lifespan` argument in `create_app`.

    Args:
        app: The running app.

    Yields:
        Control back to FastAPI while the app serves requests.
    """
    yield
    await app.state.llm_provider.aclose()


def create_app(settings: Settings, llm_provider: LLMProvider | None = None) -> FastAPI:
    """Build the FastAPI app.

    Why: This is the one place where the API is assembled, so the full setup can be read
    from top to bottom. The provider is stored in `app.state`, FastAPI's place for objects
    shared across requests.

    Used by: `ai_ops.__main__.main` and the fixtures in `tests/test_api.py`.

    Args:
        settings: Validated application settings.
        llm_provider: Replaces the configured provider. Tests pass a failing fake.

    Returns:
        The configured app. It is not running yet.

    Side effects:
        Configures the `ai_ops` logger.
    """
    configure_logging(settings.log_level)
    app = FastAPI(title="AI Operations Assistant", version="0.1.0", lifespan=_lifespan)
    app.state.llm_provider = llm_provider or build_provider(settings)
    app.add_middleware(RequestContextMiddleware)
    register_error_handlers(app)
    app.include_router(router)
    return app
