"""Split document text into overlapping chunks.

Why: A whole document embeds poorly, because one vector cannot represent many topics, and it
would use too many tokens as context. Small chunks give precise matches and short prompts.

Used by: `ai_ops.services.rag.RagService.ingest`.
"""


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Split text with a sliding window, preferring to break at whitespace.

    Why: Overlap keeps a fact that crosses a boundary whole in at least one chunk. Sizes are
    in characters, not tokens, because that needs no tokenizer and works for any model. The
    trade-off is that the overlap can begin in the middle of a word.

    This is a pure function, so it is easy to test and has no side effects.

    Used by: `ai_ops.services.rag.RagService.ingest`.

    Args:
        text: The document text.
        size: Maximum characters per chunk. Must be positive.
        overlap: Characters repeated from the previous chunk. Must be from 0 to `size - 1`.

    Returns:
        The non-blank chunks in document order, with surrounding whitespace removed.

    Raises:
        ValueError: `size` or `overlap` would stop the window from moving forward.
    """
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("size must be positive and overlap must be from 0 to size - 1")

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            # Only look past the overlap, so that the next window still moves forward.
            search_from = start + overlap + 1
            cut = max(text.rfind(" ", search_from, end), text.rfind("\n", search_from, end))
            if cut != -1:
                end = cut
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = end - overlap
    return chunks
