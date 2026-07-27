from dataclasses import dataclass

DEFAULT_CHUNK_WORD_SIZE = 180
DEFAULT_CHUNK_OVERLAP_WORDS = 30


@dataclass
class SlideChunk:
    slide_id: str
    chunk_index: int
    text: str
    content_hash: str


def chunk_text(
    text: str,
    chunk_word_size: int = DEFAULT_CHUNK_WORD_SIZE,
    overlap_words: int = DEFAULT_CHUNK_OVERLAP_WORDS,
) -> list[str]:
    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_word_size:
        return [" ".join(words)]

    chunks: list[str] = []
    step = max(1, chunk_word_size - overlap_words)
    start = 0
    while start < len(words):
        end = min(start + chunk_word_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += step
    return chunks


def chunk_slide_text(slide_id: str, content_hash: str, text: str) -> list[SlideChunk]:
    return [
        SlideChunk(slide_id=slide_id, chunk_index=i, text=chunk, content_hash=content_hash)
        for i, chunk in enumerate(chunk_text(text))
    ]
