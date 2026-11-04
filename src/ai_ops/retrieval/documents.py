"""Load knowledge base documents from a directory.

Why: Ingestion needs plain text plus a stable id. Reading files is kept apart from chunking
and embedding, so each step can be tested alone.

Used by: the lifespan function in `ai_ops.api.app`.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

from ai_ops.logging import LOGGER_NAME

logger = logging.getLogger(f"{LOGGER_NAME}.retrieval")

SUPPORTED_SUFFIXES = {".md", ".txt"}


@dataclass(frozen=True)
class Document:
    """A loaded document.

    Why: The id is the path relative to the knowledge directory, with forward slashes, so it
    is the same on Windows and Linux and citations stay stable.

    Used by: `load_documents` and `ai_ops.services.rag.RagService.ingest`.
    """

    id: str
    text: str


def load_documents(directory: Path) -> list[Document]:
    """Read all `.md` and `.txt` files under a directory.

    Why: Files are sorted, because directory listing order differs between systems and the
    results must be deterministic. Problems are logged and skipped, so one bad file or a
    missing directory does not stop the API. `GET /health` shows how many documents loaded.

    Used by: the lifespan function in `ai_ops.api.app`.

    Args:
        directory: The knowledge directory.

    Returns:
        The documents, sorted by id.

    Side effects:
        Reads files, and logs a warning for a missing directory or an undecodable file.
    """
    if not directory.is_dir():
        logger.warning("knowledge directory not found", extra={"path": str(directory)})
        return []

    documents = []
    paths = (path for path in directory.rglob("*") if path.suffix.lower() in SUPPORTED_SUFFIXES)
    for path in sorted(paths, key=lambda path: path.relative_to(directory).as_posix()):
        document_id = path.relative_to(directory).as_posix()
        try:
            documents.append(Document(id=document_id, text=path.read_text(encoding="utf-8")))
        except UnicodeDecodeError:
            logger.warning("skipped file that is not UTF-8", extra={"path": document_id})
    return documents
