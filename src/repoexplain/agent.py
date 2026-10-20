"""The agent: plan (route) -> gather evidence with tools -> answer -> (verify).

Small local models are unreliable at free-form tool calling, so the *plan* is made by
a transparent router and the tools are deterministic. The LLM does what it is good
at: turning well-chosen evidence into a readable explanation.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

from .overview import file_line, render_deps_report, render_overview
from .rag import NOT_FOUND, retrieve
from .router import Decision, ROUTES, classify
from .store import VectorStore
from .tools import RepoTools
from .verify import collect_known_names

RULES = (
    "Use ONLY the numbered EVIDENCE below. Never invent files, functions, libraries or behaviour.\n"
    "Cite evidence like [1] or [2]. Put file, function and class names in `backticks`.\n"
    f'If the evidence does not contain the answer, reply exactly: "{NOT_FOUND}" and suggest where to look.\n'
)
SYSTEMS = {
    "search": "You are a code assistant explaining a software repository.\n" + RULES + "Be concise.",
    "overview": ("You explain what a software project is and what it can do.\n" + RULES +
                 "Describe capabilities and structure in plain language. Take entry points, technologies and "
                 "dependencies from the PROJECT OVERVIEW facts; use code excerpts for details. "
                 "Mention limitations or missing features only if the user asks."),
    "locate": ("You tell developers where things are implemented.\n" + RULES +
               "Answer with file paths and function or class names, with line numbers when SYMBOL MATCHES "
               "provide them. Then add one sentence on how it works."),
    "explain": ("You explain code to a learner in simple words.\n" + RULES +
                "Say what the code is for, then walk through how it works step by step, then name the key "
                "functions. Avoid jargon, and define any term you have to use."),
}


@dataclass
class Turn:
    question: str
    answer: str


@dataclass
class Evidence:
    header: str
    text: str
    path: Optional[str] = None
    start: Optional[int] = None
    end: Optional[int] = None
    score: Optional[float] = None
    kind: str = "code"  # "code" is shown in a fence, "facts" as plain text


@dataclass
class Result:
    route: str
    reason: str
    steps: List[str]
    evidence: List[Evidence]
    stream: Iterator[str]
    known_names: Set[str]
    direct: bool = False  # answered from facts, no LLM involved

    @property
    def context_text(self) -> str:
        return "\n".join(e.text for e in self.evidence)


def source_label(e: Evidence) -> str:
    label = e.header
    if e.path and e.start:
        label = f"{e.path}:{e.start}-{e.end}"
    elif e.path:
        label = e.path
    return label + (f"  (score {e.score:.2f})" if e.score is not None else "")


def fit_budget(evidence: List[Evidence], max_chars: int) -> List[Evidence]:
    kept: List[Evidence] = []
    remaining = max_chars
    for e in evidence:
        if remaining <= 200:
            break
        text = e.text
        if len(text) > remaining:
            text = text[:remaining].rstrip() + "\n... (truncated)"
        kept.append(replace(e, text=text))
        remaining -= len(text) + len(e.header) + 20
    return kept


def build_messages(system: str, evidence: List[Evidence], question: str, history: List[Turn],
                   max_turns: int) -> List[Dict[str, str]]:
    blocks = []
    for number, e in enumerate(evidence, start=1):
        body = f"```\n{e.text}\n```" if e.kind == "code" else e.text
        blocks.append(f"[{number}] {e.header}\n{body}")
    messages = [{"role": "system", "content": system}]
    for turn in history[-max_turns:]:
        messages.append({"role": "user", "content": turn.question})
        messages.append({"role": "assistant", "content": turn.answer[:400]})
    messages.append({"role": "user", "content": "EVIDENCE:\n" + "\n\n".join(blocks) + f"\n\nQUESTION: {question}"})
    return messages


def chunk_evidence(hits) -> List[Evidence]:
    return [Evidence(f"{c['path']}:{c['start']}-{c['end']}", c["text"], c["path"], c["start"], c["end"], score)
            for score, c in hits]


class Agent:
    def __init__(self, root: Path, cfg: Dict[str, Any], store: VectorStore, overview: Dict[str, Any],
                 embedder, llm):
        self.root, self.cfg, self.store, self.overview = Path(root), cfg, store, overview
        self.embedder, self.llm = embedder, llm
        self.tools = RepoTools(root, overview)
        self.known = collect_known_names(overview)

    # ------------------------------------------------------------- tools
    def _search(self, question: str, history: List[Turn], steps: List[str], top_k: Optional[int] = None):
        query = question
        if history and len(question.split()) < 7:  # short follow-up: borrow the previous question's words
            query = history[-1].question + " " + question
        hits = retrieve(question, self.store, self.embedder, self.cfg, top_k=top_k, query_text=query)
        steps.append(f"search_code(top_k={top_k or self.cfg['retrieval']['top_k']}) -> {len(hits)} chunks")
        return hits

    # ------------------------------------------------------------ routes
    def _route_search(self, question, d: Decision, history, steps):
        return chunk_evidence(self._search(question, history, steps)), SYSTEMS["search"], None

    def _route_overview(self, question, d, history, steps):
        steps.append("project_overview() -> facts from manifests, README and file analysis")
        evidence = [Evidence("PROJECT OVERVIEW (facts extracted from the repo's files)",
                             render_overview(self.overview, 3200), kind="facts")]
        evidence += chunk_evidence(self._search(question, history, steps, top_k=3))
        return evidence, SYSTEMS["overview"], None

    def _route_locate(self, question, d, history, steps):
        syms = self.tools.find_symbols(d.symbols, d.keywords, limit=6)
        steps.append(f"find_symbols({d.symbols or d.keywords[:4]}) -> {len(syms)} matches")
        evidence: List[Evidence] = []
        if syms:
            lines = [f"- {s['name']} ({s['kind']}) in {s['path']}:{s['line']}" for s in syms]
            evidence.append(Evidence("SYMBOL MATCHES (functions and classes found by name)",
                                     "\n".join(lines), kind="facts"))
        evidence += chunk_evidence(self._search(question, history, steps, top_k=3))
        return evidence, SYSTEMS["locate"], None

    def _route_explain(self, question, d, history, steps):
        evidence: List[Evidence] = []
        if d.symbols:
            for sym in self.tools.find_symbols(d.symbols, (), limit=2):
                end = sym["end"] or sym["line"] + 40
                got = self.tools.read_file(sym["path"], sym["line"], end, max_chars=2200)
                steps.append(f"read_file({sym['path']}, lines {sym['line']}-{end})")
                if got:
                    evidence.append(Evidence(f"{got['path']}:{got['start']}-{got['end']} ({sym['name']})",
                                             got["text"], got["path"], got["start"], got["end"]))
        for path in d.paths[:2]:
            info = self.overview["files"][path]
            if info.get("lines", 0) <= 150:
                got = self.tools.read_file(path, 1, None, max_chars=3000)
                steps.append(f"read_file({path})")
                if got:
                    suffix = " (truncated)" if got["truncated"] else ""
                    evidence.append(Evidence(f"{got['path']}:{got['start']}-{got['end']}{suffix}",
                                             got["text"], got["path"], got["start"], got["end"]))
            else:
                hits = self._search(question, history, steps, top_k=12)
                mine = [h for h in hits if h[1]["path"] == path][:3]
                steps.append(f"kept {len(mine)} search results from {path}")
                evidence.append(Evidence(f"FILE SUMMARY: {path}", file_line(path, info), kind="facts"))
                evidence += chunk_evidence(mine)
        if not evidence:
            steps.append("nothing readable for the named targets; falling back to code search")
            return self._route_search(question, d, history, steps)
        return evidence, SYSTEMS["explain"], None

    def _route_deps(self, question, d, history, steps):
        steps.append("deps report built from manifests and file analysis (no LLM call)")
        sources = [Evidence(m, "", path=m, kind="facts") for m in self.overview["project"]["manifests"]]
        return sources, "", render_deps_report(self.overview)

    # ------------------------------------------------------------ public
    def answer(self, question: str, history: Optional[List[Turn]] = None,
               forced_route: Optional[str] = None) -> Result:
        history = list(history or [])
        decision = classify(question, self.overview, self.cfg["routing"]["mode"], self.llm, self.cfg)
        if forced_route:
            if forced_route not in ROUTES:
                raise ValueError(f"Unknown route '{forced_route}'. Choose from: {', '.join(ROUTES)}")
            decision.route, decision.reason = forced_route, "forced with --route"
        steps = [f"route: {decision.route} ({decision.reason})"]
        evidence, system, direct = getattr(self, "_route_" + decision.route)(question, decision, history, steps)

        if direct is not None:
            return Result(decision.route, decision.reason, steps, evidence, iter([direct]), self.known, direct=True)

        evidence = fit_budget(evidence, self.cfg["retrieval"]["max_context_chars"])
        if not evidence:
            steps.append("no evidence found -> answering 'not found' without calling the LLM")
            stream: Iterator[str] = iter([NOT_FOUND])
        else:
            lcfg = self.cfg["llm"]
            options = {k: lcfg[k] for k in ("num_ctx", "num_predict", "temperature")}
            messages = build_messages(system, evidence, question, history, self.cfg["history"]["turns"])
            stream = self.llm.chat_stream(lcfg["model"], messages, options)
        return Result(decision.route, decision.reason, steps, evidence, stream, self.known)
