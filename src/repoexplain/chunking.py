"""Split source files into chunks that keep functions and classes together.

Python files are split along function/class boundaries using the `ast` module.
Every other language falls back to blank-line blocks. Small neighbouring blocks
are merged up to `max_chars`; blocks that are too big are split on line
boundaries with a few lines of overlap. Line numbers are always preserved so
answers can cite `file:start-end`.
"""
from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

Range = Tuple[int, int]  # 1-based, inclusive


@dataclass
class Chunk:
    path: str
    start: int
    end: int
    text: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _size(lines: List[str], start: int, end: int) -> int:
    return sum(len(line) + 1 for line in lines[start - 1:end])


def _python_ranges(source: str, lines: List[str], max_chars: int) -> List[Range]:
    tree = ast.parse(source)
    ranges: List[Range] = []

    def node_range(node: ast.AST) -> Range:
        start = node.lineno  # type: ignore[attr-defined]
        for deco in getattr(node, "decorator_list", []):
            start = min(start, deco.lineno)
        return start, node.end_lineno  # type: ignore[attr-defined]

    def visit(body: List[ast.stmt], lo: int, hi: int) -> None:
        cursor = lo
        for node in body:
            start, end = node_range(node)
            if start < cursor:
                start = cursor
            if end < start:
                continue
            if start > cursor:
                ranges.append((cursor, start - 1))
            if isinstance(node, ast.ClassDef) and node.body and _size(lines, start, end) > max_chars:
                first = node_range(node.body[0])[0]
                if first > start:
                    ranges.append((start, first - 1))
                visit(node.body, max(first, start), end)
            else:
                ranges.append((start, end))
            cursor = end + 1
        if cursor <= hi:
            ranges.append((cursor, hi))

    visit(tree.body, 1, len(lines))
    return ranges


def _blank_line_ranges(lines: List[str]) -> List[Range]:
    ranges: List[Range] = []
    start = None
    for i, line in enumerate(lines, start=1):
        if line.strip():
            if start is None:
                start = i
        elif start is not None:
            ranges.append((start, i - 1))
            start = None
    if start is not None:
        ranges.append((start, len(lines)))
    return ranges


def _split_range(lines: List[str], start: int, end: int, max_chars: int, overlap: int) -> List[Range]:
    if _size(lines, start, end) <= max_chars:
        return [(start, end)]
    pieces: List[Range] = []
    a = start
    while a <= end:
        b, total = a, 0
        while b <= end and (total == 0 or total + len(lines[b - 1]) + 1 <= max_chars):
            total += len(lines[b - 1]) + 1
            b += 1
        b -= 1  # last line included
        pieces.append((a, b))
        if b >= end:
            break
        a = max(b + 1 - overlap, a + 1)
    return pieces


def _pack(lines: List[str], ranges: List[Range], max_chars: int, overlap: int) -> List[Range]:
    packed: List[Range] = []
    cur_start = cur_end = None
    cur_size = 0
    for rng_start, rng_end in ranges:
        for start, end in _split_range(lines, rng_start, rng_end, max_chars, overlap):
            size = _size(lines, start, end)
            if cur_start is not None and cur_size + size <= max_chars:
                cur_end, cur_size = end, cur_size + size
            else:
                if cur_start is not None:
                    packed.append((cur_start, cur_end))
                cur_start, cur_end, cur_size = start, end, size
    if cur_start is not None:
        packed.append((cur_start, cur_end))
    return packed


def chunk_file(path: str, text: str, max_chars: int = 1000, overlap_lines: int = 3) -> List[Chunk]:
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    if not lines:
        return []

    ranges = None
    if path.lower().endswith(".py"):
        try:
            ranges = _python_ranges(text, lines, max_chars)
        except (SyntaxError, ValueError, RecursionError):
            ranges = None
    if ranges is None:
        ranges = _blank_line_ranges(lines)

    chunks: List[Chunk] = []
    for start, end in _pack(lines, ranges, max_chars, overlap_lines):
        body = "\n".join(lines[start - 1:end])
        if not body.strip():
            continue
        if len(body) > max_chars * 2:  # e.g. a single enormous minified line
            body = body[: max_chars * 2]
        chunks.append(Chunk(path=path, start=start, end=end, text=body))
    return chunks
