"""Safe, deterministic tools the agent can use on an indexed repository.

`read_file` only opens files that were indexed (so never .env files, keys,
binaries or anything outside the repo), which also blocks path traversal.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .indexer import read_text


class RepoTools:
    def __init__(self, root: Path, overview: Dict[str, Any]):
        self.root = Path(root).resolve()
        self.files: Dict[str, Dict[str, Any]] = overview["files"]

    def read_file(self, path: str, start: int = 1, end: Optional[int] = None,
                  max_chars: int = 3000) -> Optional[Dict[str, Any]]:
        path = path.replace("\\", "/")
        if path not in self.files:  # whitelist: only indexed files
            return None
        full = (self.root / path).resolve()
        try:
            full.relative_to(self.root)
        except ValueError:
            return None
        try:
            lines = read_text(full).split("\n")
        except OSError:
            return None
        total = len(lines)
        start = max(1, start)
        end = min(end or total, total)
        if end < start:
            return None
        body = "\n".join(lines[start - 1:end])
        truncated = False
        if len(body) > max_chars:
            body = body[:max_chars]
            end = start + body.count("\n")
            truncated = True
        return {"path": path, "start": start, "end": end, "text": body,
                "truncated": truncated, "total_lines": total}

    def find_symbols(self, names: Iterable[str], keywords: Iterable[str] = (), limit: int = 6) -> List[Dict[str, Any]]:
        wanted = [n.lower() for n in names]
        words = [k.lower() for k in keywords if len(k) >= 4]
        scored = []
        for path, info in self.files.items():
            for sym in info.get("symbols", []):
                full = sym["name"].lower()
                tail = full.split(".")[-1]
                score = 0
                if full in wanted or tail in wanted:
                    score += 10
                score += sum(1 for w in words if w in full)
                if score:
                    scored.append((-score, path, sym["line"], sym))
        scored.sort(key=lambda t: t[:3])
        return [{"path": p, "name": s["name"], "kind": s["kind"], "line": s["line"], "end": s.get("end")}
                for _, p, _, s in scored[:limit]]

    def list_tree(self, prefix: str = "", depth: int = 2) -> str:
        prefix = prefix.strip("/")
        paths = [p for p in sorted(self.files) if not prefix or p.startswith(prefix + "/")]

        def walk(paths_: List[str], base: str, level: int) -> List[str]:
            out: List[str] = []
            dirs: Dict[str, List[str]] = {}
            for p in paths_:
                rest = p[len(base):] if base else p
                head, _, tail = rest.partition("/")
                if tail:
                    dirs.setdefault(head, []).append(p)
                else:
                    out.append("  " * level + head)
            for d in sorted(dirs):
                out.append("  " * level + f"{d}/ ({len(dirs[d])} files)")
                if level + 1 < depth:
                    out.extend(walk(dirs[d], (base + d + "/"), level + 1))
            return out

        return "\n".join(walk(paths, prefix + "/" if prefix else "", 0))
