"""Decide how to answer a question: the 'planning' step of the agent.

Rules first (fast, predictable, explainable). Optionally the LLM classifies
instead (`routing.mode: llm`), falling back to the rules if its reply is unusable.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

ROUTES = ("overview", "deps", "locate", "explain", "search")

STOPWORDS = {
    "what", "where", "which", "when", "does", "this", "that", "with", "from", "have", "your", "about",
    "there", "their", "into", "work", "works", "used", "uses", "using", "code", "file", "files", "function",
    "functions", "class", "classes", "project", "repo", "tool", "explain", "please", "tell", "show", "give",
    "list", "happen", "happens", "handled", "handle", "defined", "implemented", "located", "find", "like",
    "these", "those", "would", "could", "should", "also", "each", "much", "many", "some", "than", "then",
}

DEPS_RX = re.compile(
    r"\bdepend(s|ed|ing|ency|encies)?\b|\blibrar(y|ies)\b|\bpackages?\b|\brequirements?\b|\btech(nolog(y|ies))?\b"
    r"|\bstack\b|\bframeworks?\b|\bwritten in\b|\bbuilt with\b|\bprogramming languages?\b|\bwhich languages?\b"
    r"|\bwhat languages?\b|\bthird[- ]party\b|\bpip install\b|\bnpm\b", re.I)

OVERVIEW_RX = re.compile(
    r"what (does|is|can|do) (this|the|it|these)\b.*\b(project|repo|repository|app|application|tool|codebase|program|software|package|library)\b"
    r"|what (can|does) (this|it) do\b|what can (this|it|the)\b|\boverview\b|\bsummar(y|ise|ize)\b|\bfeatures?\b|\bpurpose\b"
    r"|\bcapabilit|\barchitecture\b|\bproject structure\b|\bfolder structure\b|\bdirectory structure\b|\blayout\b|\bentry ?points?\b"
    r"|\bruns? first\b|\bstarts? first\b|\bfirst (function|thing|file)\b|\bwhere (does|do|did) .*\b(start|begin)\b"
    r"|\bhow (does|do|is) .*\b(start|launch(ed)?|bootstrap(ped)?)\b|\bmain function\b|\bstart ?up\b|\bwhat is this\b", re.I)

LOCATE_RX = re.compile(
    r"\bwhere\b|\blocat(e|ed|ion)\b"
    r"|\bwhich (file|files|module|modules|function|functions|class|classes|folder|script)\b.*\b(handles?|contains?|implements?|defines?|does|is responsible|has|holds|manages?|runs?|parses|loads|builds)\b"
    r"|\bfind (the )?(code|function|file|class|logic)\b|\b(defined|implemented|handled|lives?|stored|configured|declared)\b", re.I)

LLM_PROMPT = (
    "Classify the user's question about a code repository into exactly one label:\n"
    "overview - what the project is or does, its features, structure or entry point\n"
    "deps - dependencies, libraries, languages or technologies used\n"
    "locate - where something is implemented or defined\n"
    "explain - explain a specific named file, function or class\n"
    "search - anything else about how the code works\n"
    "Reply with the label only."
)


@dataclass
class Decision:
    route: str
    reason: str
    paths: List[str] = field(default_factory=list)
    symbols: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)


def _symbol_index(overview: Dict[str, Any]) -> Dict[str, str]:
    index: Dict[str, str] = {}
    for info in overview["files"].values():
        for sym in info.get("symbols", []):
            full = sym["name"].lower()
            index.setdefault(full, sym["name"])
            index.setdefault(full.split(".")[-1], sym["name"].split(".")[-1])
    return index


def extract_targets(question: str, overview: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    files = list(overview["files"])
    paths: List[str] = []
    candidates = re.findall(r"[A-Za-z0-9_./\\-]+", question)
    for token in candidates:
        token = token.strip(".,;:?!)(`'\"").replace("\\", "/")
        if not token or not ("/" in token or re.search(r"\.[A-Za-z0-9]{1,5}$", token)):
            continue
        base = token.rsplit("/", 1)[-1]
        for rel in files:
            if rel == token or rel.endswith("/" + token) or (("/" not in token) and rel.rsplit("/", 1)[-1] == base):
                if rel not in paths:
                    paths.append(rel)
    lowered = question.lower()
    if re.search(r"\b(file|module|script|source)\b", lowered):
        for word in re.findall(r"[a-z_][a-z0-9_]{3,}", lowered):
            stems = [rel for rel in files if rel.rsplit("/", 1)[-1].rsplit(".", 1)[0].lower() == word]
            for rel in stems[:3]:
                if rel not in paths:
                    paths.append(rel)

    index = _symbol_index(overview)
    symbols: List[str] = []
    raw: List[str] = re.findall(r"`([^`\n]+)`", question)
    raw += re.findall(r"([A-Za-z_][\w.]*)\(", question)
    for token in re.findall(r"[A-Za-z_][A-Za-z0-9_.]*", question):
        if "_" in token or re.search(r"[a-z][A-Z]", token) or re.match(r"^[A-Z][a-z]+[A-Z]", token):
            raw.append(token)
    for token in raw:
        key = token.strip("`().,;:?! ").lower()
        if key in index and index[key] not in symbols:
            symbols.append(index[key])
    return paths[:3], symbols[:4]


def _keywords(question: str) -> List[str]:
    words = re.findall(r"[a-z][a-z0-9_]{3,}", question.lower())
    seen: List[str] = []
    for w in words:
        if w not in STOPWORDS and w not in seen:
            seen.append(w)
    return seen[:8]


def _rule_route(question: str, has_target: bool) -> Tuple[str, str]:
    if DEPS_RX.search(question) and not has_target:
        return "deps", "mentions dependencies, libraries, languages or technologies"
    if OVERVIEW_RX.search(question) and not has_target:
        return "overview", "asks what the project is or does, or how it starts"
    if LOCATE_RX.search(question):
        return "locate", "asks where something is implemented"
    if has_target:
        return "explain", "names a specific file or symbol"
    return "search", "general question about the code"


def _llm_route(question: str, llm, cfg: Dict[str, Any]) -> Optional[str]:
    lcfg = cfg["llm"]
    messages = [{"role": "system", "content": LLM_PROMPT}, {"role": "user", "content": question}]
    try:
        reply = "".join(llm.chat_stream(lcfg["model"], messages,
                                        {"num_ctx": lcfg["num_ctx"], "num_predict": 6, "temperature": 0.0}))
    except Exception:
        return None
    words = re.findall(r"[a-z]+", reply.lower())
    return words[0] if words and words[0] in ROUTES else None


def classify(question: str, overview: Dict[str, Any], mode: str = "rules", llm=None,
             cfg: Optional[Dict[str, Any]] = None) -> Decision:
    paths, symbols = extract_targets(question, overview)
    has_target = bool(paths or symbols)
    route, reason = _rule_route(question, has_target)
    if mode == "llm" and llm is not None and cfg is not None:
        chosen = _llm_route(question, llm, cfg)
        if chosen:
            route, reason = chosen, "chosen by the LLM router"
    if route == "explain" and not has_target:
        route, reason = "search", "no specific file or symbol found, using code search"
    return Decision(route, reason, paths, symbols, _keywords(question))
