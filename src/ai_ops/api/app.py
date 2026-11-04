"""FastAPI application factory.

Why: A factory builds the app from settings passed as an argument. Importing this module
has no side effects, and each test can build its own app without environment variables.

Used by: `ai_ops.__main__.main` and `tests/test_api.py`.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from ai_ops.api.errors import register_error_handlers
from ai_ops.api.middleware import BodySizeLimitMiddleware, RequestContextMiddleware
from ai_ops.api.routes import router
from ai_ops.config import Settings
from ai_ops.llm.factory import build_provider
from ai_ops.llm.provider import LLMProvider
from ai_ops.logging import configure_logging
from ai_ops.retrieval.documents import load_documents
from ai_ops.retrieval.vector_store import InMemoryVectorStore
from ai_ops.services.rag import RagService

# Far more than any valid request needs today. A question is limited to 1000 characters.
MAX_BODY_BYTES = 64 * 1024


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the knowledge base at startup and close the LLM provider at shutdown.

    Why: FastAPI runs the code before `yield` once before serving, and the code after it at
    shutdown. Ingesting here means the first question does not have to wait for it, and a
    failing provider stops the startup with a visible error.

    Used by: FastAPI, through the `lifespan` argument in `create_app`.

    Args:
        app: The app being started.

    Yields:
        Control back to FastAPI while the app serves requests.

    Raises:
        LLMError: Embedding the documents failed. The app does not start.
    """
    documents = load_documents(app.state.knowledge_dir)
    await app.state.rag_service.ingest(documents)
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
    app.state.knowledge_dir = settings.knowledge_dir
    app.state.rag_service = RagService(
        provider=app.state.llm_provider,
        store=InMemoryVectorStore(),
        chunk_size=settings.rag_chunk_size,
        chunk_overlap=settings.rag_chunk_overlap,
        top_k=settings.rag_top_k,
        min_score=settings.rag_min_score,
    )
    # The middleware added last runs first, so the request id exists before the body is read.
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=MAX_BODY_BYTES)
    app.add_middleware(RequestContextMiddleware)
    register_error_handlers(app)
    app.include_router(router)
    return app
