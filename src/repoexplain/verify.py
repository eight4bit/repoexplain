"""Cheap, deterministic grounding check: flag things in an answer that the
evidence does not support. No extra LLM call, so it costs nothing on 8 GB RAM."""
from __future__ import annotations

import re
from typing import Iterable, List, Set

_IDENT = re.compile(r"^[A-Za-z_][\w./:-]*(\(\))?$")


def grounding_warnings(answer: str, context_text: str, known_names: Set[str], n_sources: int) -> List[str]:
    warnings: List[str] = []
    cited = sorted({int(n) for n in re.findall(r"\[(\d{1,2})\]", answer)})
    bad = [n for n in cited if n < 1 or n > n_sources]
    if bad:
        warnings.append("the answer cites source(s) " + ", ".join(f"[{n}]" for n in bad) +
                        f" but only {n_sources} were provided")

    for token in sorted(set(re.findall(r"`([^`\n]{2,80})`", answer))):
        name = token.strip()
        if not _IDENT.match(name):
            continue
        core = name[:-2] if name.endswith("()") else name
        tail = re.split(r"[.:/]", core)[-1]
        lowered = {k.lower() for k in known_names}
        if core in context_text or tail in context_text:
            continue
        if core.lower() in lowered or tail.lower() in lowered:
            continue
        warnings.append(f"`{name}` was not found in the code that was retrieved")
    return warnings


def collect_known_names(overview: dict) -> Set[str]:
    names: Set[str] = set()
    for path, info in overview["files"].items():
        base = path.rsplit("/", 1)[-1]
        names.update({path, base, base.rsplit(".", 1)[0]})
        for sym in info.get("symbols", []):
            names.add(sym["name"])
            names.add(sym["name"].split(".")[-1])
    project = overview["project"]
    names.update(d["name"] for d in project["dependencies"])
    names.update(t["label"] for t in project["technologies"])
    names.update(s["name"] for s in project["scripts"])
    return names
