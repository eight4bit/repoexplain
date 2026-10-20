"""A tiny vector store: unit-length float32 vectors in a NumPy array.

Search is exact cosine similarity (a dot product, since vectors are normalised).
For repositories with thousands of chunks this is instant, needs no server, and
works identically on macOS, Windows and Linux.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np

INDEX_DIRNAME = ".repoexplain"
FORMAT_VERSION = 1


def _empty() -> np.ndarray:
    return np.zeros((0, 0), dtype=np.float32)


def normalize(vectors: np.ndarray) -> np.ndarray:
    if vectors.size == 0:
        return vectors.astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (vectors / norms).astype(np.float32)


class VectorStore:
    def __init__(self, directory: Path):
        self.dir = Path(directory)
        self.chunks: List[Dict[str, Any]] = []
        self.vectors: np.ndarray = _empty()
        self.meta: Dict[str, Any] = {}

    # ---- persistence -------------------------------------------------------
    def load(self) -> bool:
        meta_p, chunks_p, vec_p = (self.dir / n for n in ("meta.json", "chunks.jsonl", "vectors.npy"))
        if not (meta_p.is_file() and chunks_p.is_file() and vec_p.is_file()):
            return False
        try:
            self.meta = json.loads(meta_p.read_text(encoding="utf-8"))
            with open(chunks_p, encoding="utf-8") as handle:
                self.chunks = [json.loads(line) for line in handle if line.strip()]
            self.vectors = np.load(vec_p)
        except (OSError, ValueError):
            self.reset({})
            return False
        if self.meta.get("version") != FORMAT_VERSION or len(self.chunks) != self.vectors.shape[0]:
            self.reset({})
            return False
        return True

    def save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / ".gitignore").write_text("*\n", encoding="utf-8")  # never commit the index
        self._atomic_write("chunks.jsonl", lambda f: f.write(
            "".join(json.dumps(c, ensure_ascii=False) + "\n" for c in self.chunks).encode("utf-8")))
        self._atomic_write("vectors.npy", lambda f: np.save(f, self.vectors))
        self._atomic_write("meta.json", lambda f: f.write(
            json.dumps(self.meta, indent=2, ensure_ascii=False).encode("utf-8")))

    def _atomic_write(self, name: str, writer) -> None:
        tmp = self.dir / (name + ".tmp")
        with open(tmp, "wb") as handle:
            writer(handle)
        os.replace(tmp, self.dir / name)

    # ---- state -------------------------------------------------------------
    def reset(self, settings: Dict[str, Any]) -> None:
        self.chunks = []
        self.vectors = _empty()
        self.meta = dict(settings, files={}, version=FORMAT_VERSION)

    def compatible(self, settings: Dict[str, Any]) -> bool:
        return all(self.meta.get(key) == value for key, value in settings.items())

    def replace(self, drop_paths: Set[str], new_chunks: List[Dict[str, Any]], new_vectors: np.ndarray) -> None:
        """Remove all chunks of `drop_paths`, then append the new chunks."""
        keep = [i for i, c in enumerate(self.chunks) if c["path"] not in drop_paths]
        kept_chunks = [self.chunks[i] for i in keep]
        kept_vecs = self.vectors[keep] if (keep and self.vectors.size) else _empty()
        parts = [v for v in (kept_vecs, new_vectors) if v.shape[0] > 0]
        self.chunks = kept_chunks + list(new_chunks)
        self.vectors = np.vstack(parts).astype(np.float32) if parts else _empty()

    # ---- search ------------------------------------------------------------
    def search(self, query_vec, top_k: int = 4, min_score: float = 0.0) -> List[Tuple[float, Dict[str, Any]]]:
        if self.vectors.shape[0] == 0:
            return []
        q = np.asarray(query_vec, dtype=np.float32)
        if q.shape[0] != self.vectors.shape[1]:
            raise ValueError(
                "The index was built with a different embedding model. "
                "Run `repoexplain index <path> --rebuild`."
            )
        norm = np.linalg.norm(q)
        if norm:
            q = q / norm
        scores = self.vectors @ q
        k = min(top_k, scores.shape[0])
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(float(scores[i]), self.chunks[i]) for i in top if scores[i] >= min_score]
