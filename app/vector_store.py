"""
Per-project embedded-chunk store, one SQLite database per project
(`<data_dir>/<project_id>.db`), mirroring the one-IndexedDB-per-project
design the runtime used when everything ran in the browser. Records are
keyed by slideId and carry the content hash they were embedded from, so a
re-preview after edits only needs to re-embed the slides that changed.
"""

import json
import sqlite3
from dataclasses import dataclass, field

import numpy as np

from app.config import settings


@dataclass
class StoredChunk:
    slide_id: str
    chunk_index: int
    text: str
    embedding: list[float]


@dataclass
class _SlideRecord:
    content_hash: str
    chunks: list[StoredChunk] = field(default_factory=list)


def _open_db(project_id: str) -> sqlite3.Connection:
    db_path = settings.data_dir / f"{project_id}.db"
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS slides (slide_id TEXT PRIMARY KEY, content_hash TEXT)"
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS chunks (
            slide_id TEXT,
            chunk_index INTEGER,
            text TEXT,
            embedding TEXT,
            PRIMARY KEY (slide_id, chunk_index)
        )"""
    )
    conn.commit()
    return conn


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class VectorStore:
    def __init__(self, project_id: str):
        self.project_id = project_id
        self._conn = _open_db(project_id)
        self._records: dict[str, _SlideRecord] = {}
        self._load()

    def _load(self) -> None:
        slide_rows = self._conn.execute("SELECT slide_id, content_hash FROM slides").fetchall()
        self._records = {slide_id: _SlideRecord(content_hash=content_hash) for slide_id, content_hash in slide_rows}
        for slide_id, chunk_index, text, embedding_json in self._conn.execute(
            "SELECT slide_id, chunk_index, text, embedding FROM chunks"
        ):
            record = self._records.get(slide_id)
            if record is None:
                continue
            record.chunks.append(
                StoredChunk(
                    slide_id=slide_id,
                    chunk_index=chunk_index,
                    text=text,
                    embedding=json.loads(embedding_json),
                )
            )

    def get_stale_or_missing_slide_ids(self, signatures: dict[str, str]) -> list[str]:
        """signatures: {slideId: contentHash}. Returns ids that are new or whose stored hash differs."""
        return [
            slide_id
            for slide_id, content_hash in signatures.items()
            if self._records.get(slide_id) is None
            or self._records[slide_id].content_hash != content_hash
        ]

    def upsert_slide_chunks(
        self, slide_id: str, content_hash: str, chunks: list[StoredChunk]
    ) -> None:
        self._records[slide_id] = _SlideRecord(content_hash=content_hash, chunks=chunks)
        self._conn.execute(
            "INSERT OR REPLACE INTO slides (slide_id, content_hash) VALUES (?, ?)",
            (slide_id, content_hash),
        )
        self._conn.execute("DELETE FROM chunks WHERE slide_id = ?", (slide_id,))
        self._conn.executemany(
            "INSERT INTO chunks (slide_id, chunk_index, text, embedding) VALUES (?, ?, ?, ?)",
            [
                (slide_id, chunk.chunk_index, chunk.text, json.dumps(chunk.embedding))
                for chunk in chunks
            ],
        )
        self._conn.commit()

    def remove_slide(self, slide_id: str) -> None:
        self._records.pop(slide_id, None)
        self._conn.execute("DELETE FROM slides WHERE slide_id = ?", (slide_id,))
        self._conn.execute("DELETE FROM chunks WHERE slide_id = ?", (slide_id,))
        self._conn.commit()

    def prune_to_slide_ids(self, valid_slide_ids: list[str]) -> None:
        valid = set(valid_slide_ids)
        for slide_id in [sid for sid in self._records if sid not in valid]:
            self.remove_slide(slide_id)

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[StoredChunk]:
        all_chunks = [chunk for record in self._records.values() for chunk in record.chunks]
        if not all_chunks:
            return []
        query = np.array(query_embedding)
        scored = [
            (cosine_similarity(query, np.array(chunk.embedding)), chunk) for chunk in all_chunks
        ]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [chunk for _, chunk in scored[:top_k]]

    @property
    def is_empty(self) -> bool:
        return len(self._records) == 0


_stores: dict[str, VectorStore] = {}


def get_vector_store(project_id: str) -> VectorStore:
    if project_id not in _stores:
        _stores[project_id] = VectorStore(project_id)
    return _stores[project_id]
