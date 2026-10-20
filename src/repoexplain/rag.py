"""Retrieval-augmented answering over the indexed repository."""
from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional, Tuple

from .store import VectorStore

Hit = Tuple[float, Dict[str, Any]]

NOT_FOUND = "I couldn't find that in the indexed code."

SYSTEM_PROMPT = (
    "You are a code assistant that explains a software repository.\n"
    "Answer ONLY from the code excerpts in CONTEXT. Do not invent files, functions or behaviour.\n"
    "Cite the excerpts you used like [1] or [2], matching the numbers in CONTEXT.\n"
    f"If the excerpts do not contain the answer, reply exactly: \"{NOT_FOUND}\" "
    "and suggest what to search for instead.\n"
    "Be concise and refer to file names and function names."
)


def retrieve(question: str, store: VectorStore, embedder, cfg: Dict[str, Any],
             top_k: Optional[int] = None, query_text: Optional[str] = None) -> List[Hit]:
    ecfg = cfg["embeddings"]
    query = (ecfg.get("query_prefix") or "") + (query_text or question)
    vector = embedder.embed(ecfg["model"], [query])[0]
    if cfg["memory"]["unload_embedder"]:
        embedder.unload(ecfg["model"], kind="embed")  # free RAM before the LLM loads
    rcfg = cfg["retrieval"]
    return store.search(vector, top_k or rcfg["top_k"], rcfg["min_score"])


def build_messages(question: str, hits: List[Hit]) -> List[Dict[str, str]]:
    blocks = []
    for number, (_, chunk) in enumerate(hits, start=1):
        blocks.append(f"[{number}] {chunk['path']}:{chunk['start']}-{chunk['end']}\n```\n{chunk['text']}\n```")
    context = "\n\n".join(blocks)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION: {question}"},
    ]


def answer_stream(question: str, hits: List[Hit], llm, cfg: Dict[str, Any]) -> Iterator[str]:
    if not hits:
        yield NOT_FOUND
        return
    lcfg = cfg["llm"]
    options = {k: lcfg[k] for k in ("num_ctx", "num_predict", "temperature")}
    yield from llm.chat_stream(lcfg["model"], build_messages(question, hits), options)
