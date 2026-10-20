"""Build or update the vector index and the project overview for a repository."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from .analyzer import analyze_file, is_code
from .chunking import Chunk, chunk_file
from .manifests import is_manifest
from .overview import build_overview, load_overview, save_overview
from .scanner import scan_repo
from .store import INDEX_DIRNAME, VectorStore, normalize
from .summarize import add_summaries

README_NAMES = {"readme.md", "readme.rst", "readme.txt", "readme"}


def read_text(path: Path) -> str:
    text = path.read_bytes().decode("utf-8", errors="replace")
    return text.replace("\r\n", "\n").replace("\r", "\n")  # same hash on every OS


def _hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def embed_texts(embedder, model: str, texts: List[str], batch_size: int, log: Callable[[str], None]) -> np.ndarray:
    parts = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        parts.append(np.asarray(embedder.embed(model, batch), dtype=np.float32))
        log(f"  embedded {min(start + batch_size, len(texts))}/{len(texts)} chunks")
    return normalize(np.vstack(parts)) if parts else np.zeros((0, 0), dtype=np.float32)


def index_repo(root: Path, cfg: Dict[str, Any], embedder, rebuild: bool = False,
               log: Callable[[str], None] = print, summarize: bool = False,
               summarizer=None) -> Dict[str, Any]:
    root = Path(root).resolve()
    icfg, ecfg = cfg["indexing"], cfg["embeddings"]
    store = VectorStore(root / INDEX_DIRNAME)
    settings = {
        "embed_model": ecfg["model"],
        "chunk_chars": icfg["chunk_chars"],
        "overlap_lines": icfg["overlap_lines"],
    }

    previous_overview = None if rebuild else load_overview(store.dir)
    loaded = (not rebuild) and store.load()
    if loaded and not store.compatible(settings):
        log("Index settings or embedding model changed; rebuilding from scratch.")
        loaded = False
    if not loaded:
        store.reset(settings)

    scan = scan_repo(root, icfg["max_file_kb"], icfg.get("extra_ignore", []))
    old_hashes: Dict[str, str] = dict(store.meta.get("files", {}))
    new_hashes: Dict[str, str] = {}
    changed: List[str] = []
    chunks: List[Chunk] = []
    analyses: Dict[str, Dict[str, Any]] = {}
    manifest_texts: Dict[str, str] = {}
    code_texts: Dict[str, str] = {}
    readme_path: Optional[str] = None
    readme_text = ""
    summary_chars = cfg["summarizer"]["max_chars"]

    for rel in scan.files:
        text = read_text(root / rel)
        digest = _hash(text)
        new_hashes[rel] = digest
        info = analyze_file(rel, text)
        info["hash"] = digest
        analyses[rel] = info
        if is_manifest(rel):
            manifest_texts[rel] = text
        if rel.rsplit("/", 1)[-1].lower() in README_NAMES and (readme_path is None or rel.count("/") < readme_path.count("/")):
            readme_path, readme_text = rel, text
        if summarize and is_code(info.get("lang")):
            code_texts[rel] = text[:summary_chars]
        if old_hashes.get(rel) != digest:
            changed.append(rel)
            chunks.extend(chunk_file(rel, text, icfg["chunk_chars"], icfg["overlap_lines"]))

    removed = set(old_hashes) - set(new_hashes)
    log(f"Found {len(scan.files)} files: {len(changed)} new/changed, "
        f"{len(scan.files) - len(changed)} unchanged, {len(removed)} removed.")
    if scan.skipped:
        log("Skipped: " + ", ".join(f"{count} {reason}" for reason, count in sorted(scan.skipped.items())))

    if chunks:
        texts = [f"File: {c.path}\n{c.text}" for c in chunks]
        vectors = embed_texts(embedder, ecfg["model"], texts, ecfg["batch_size"], log)
    else:
        vectors = np.zeros((0, 0), dtype=np.float32)

    store.replace(removed | set(changed), [c.to_dict() for c in chunks], vectors)
    store.meta["files"] = new_hashes
    store.meta["dim"] = int(store.vectors.shape[1]) if store.vectors.size else 0
    store.save()

    if cfg["memory"]["unload_embedder"] and chunks:
        embedder.unload(ecfg["model"], kind="embed")

    # Project overview: deterministic facts about languages, dependencies, entry points...
    overview = build_overview(root, analyses, manifest_texts, readme_text)
    summaries_made = 0
    if summarize:
        provider = summarizer if summarizer is not None else embedder
        previous = {}
        if previous_overview:
            previous = {p: (i.get("hash", ""), i.get("summary", ""))
                        for p, i in previous_overview["files"].items() if i.get("summary")}
        log("Writing one-line file summaries with the small model...")
        summaries_made = add_summaries(overview, code_texts, previous, provider, cfg, log)
    save_overview(store.dir, overview)

    project = overview["project"]
    return {
        "files": len(scan.files),
        "changed": len(changed),
        "removed": len(removed),
        "new_chunks": len(chunks),
        "total_chunks": len(store.chunks),
        "approx_tokens": sum(len(c["text"]) for c in store.chunks) // 4,
        "skipped": scan.skipped,
        "languages": [l["language"] for l in project["languages"][:3]],
        "dependencies": len(project["dependencies"]),
        "technologies": len(project["technologies"]),
        "entry_points": len(project["entry_points"]),
        "summaries": summaries_made,
    }
