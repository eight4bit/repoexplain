"""Static analysis of single files: language, symbols, entry points, imports.

Everything here is deterministic (no LLM), so the facts it produces are reliable.
Python is analysed with the `ast` module; other languages use light regexes.
"""
from __future__ import annotations

import ast
import importlib.util
import os
import re
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

LANGUAGES = {
    ".py": "Python", ".js": "JavaScript", ".jsx": "JavaScript", ".mjs": "JavaScript", ".cjs": "JavaScript",
    ".ts": "TypeScript", ".tsx": "TypeScript", ".vue": "Vue", ".java": "Java", ".kt": "Kotlin",
    ".scala": "Scala", ".go": "Go", ".rs": "Rust", ".c": "C", ".h": "C", ".cpp": "C++", ".cc": "C++",
    ".hpp": "C++", ".cs": "C#", ".rb": "Ruby", ".php": "PHP", ".swift": "Swift", ".dart": "Dart",
    ".lua": "Lua", ".r": "R", ".sh": "Shell", ".bash": "Shell", ".ps1": "PowerShell", ".sql": "SQL",
    ".html": "HTML", ".css": "CSS", ".scss": "CSS", ".md": "Markdown", ".rst": "Markdown",
    ".yml": "YAML", ".yaml": "YAML", ".json": "JSON", ".toml": "TOML",
}
NON_CODE = {"Markdown", "YAML", "JSON", "TOML"}
MAX_SYMBOLS = 80


def language_of(path: str) -> Optional[str]:
    name = path.rsplit("/", 1)[-1]
    if name == "Dockerfile":
        return "Dockerfile"
    return LANGUAGES.get(os.path.splitext(name)[1].lower())


def is_code(lang: Optional[str]) -> bool:
    return bool(lang) and lang not in NON_CODE


# ---------------------------------------------------------------- symbols
_JS = [
    ("function", re.compile(r"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?function\*?\s+([A-Za-z_$][\w$]*)")),
    ("class", re.compile(r"^\s*(?:export\s+)?(?:default\s+)?(?:abstract\s+)?class\s+([A-Za-z_$][\w$]*)")),
    ("function", re.compile(r"^\s*(?:export\s+)?const\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?(?:\([^)]*\)|[A-Za-z_$][\w$]*)\s*=>")),
]
_GO = [
    ("function", re.compile(r"^func\s+(?:\([^)]*\)\s*)?([A-Za-z_]\w*)")),
    ("type", re.compile(r"^type\s+([A-Za-z_]\w*)\s+(?:struct|interface)")),
]
_RUST = [
    ("function", re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?fn\s+([A-Za-z_]\w*)")),
    ("type", re.compile(r"^\s*(?:pub\s+)?(?:struct|enum|trait)\s+([A-Za-z_]\w*)")),
]
_JAVA = [("class", re.compile(
    r"^\s*(?:public\s+|private\s+|protected\s+|internal\s+)?(?:abstract\s+|final\s+|static\s+|sealed\s+)*"
    r"(?:class|interface|enum|record)\s+([A-Za-z_]\w*)"))]
_KOTLIN = [
    ("function", re.compile(r"^\s*(?:private\s+|public\s+|internal\s+|suspend\s+)*fun\s+(?:<[^>]*>\s*)?([A-Za-z_]\w*)")),
    ("class", re.compile(r"^\s*(?:data\s+|sealed\s+|abstract\s+|open\s+)*(?:class|interface|object)\s+([A-Za-z_]\w*)")),
]
_RUBY = [
    ("function", re.compile(r"^\s*def\s+(?:self\.)?([A-Za-z_]\w*[?!]?)")),
    ("class", re.compile(r"^\s*(?:class|module)\s+([A-Z]\w*)")),
]
_PHP = [
    ("function", re.compile(r"^\s*(?:public\s+|private\s+|protected\s+|static\s+)*function\s+([A-Za-z_]\w*)")),
    ("class", re.compile(r"^\s*(?:abstract\s+|final\s+)?class\s+([A-Za-z_]\w*)")),
]
_C = [("type", re.compile(r"^\s*(?:typedef\s+)?(?:struct|class|enum)\s+([A-Za-z_]\w*)"))]
_SWIFT = [
    ("function", re.compile(r"^\s*(?:public\s+|private\s+|internal\s+|static\s+)*func\s+([A-Za-z_]\w*)")),
    ("class", re.compile(r"^\s*(?:public\s+|final\s+)*(?:class|struct|enum|protocol)\s+([A-Za-z_]\w*)")),
]
_PATTERNS = {
    "JavaScript": _JS, "TypeScript": _JS, "Vue": _JS, "Go": _GO, "Rust": _RUST,
    "Java": _JAVA, "C#": _JAVA, "Scala": _JAVA, "Kotlin": _KOTLIN, "Ruby": _RUBY,
    "PHP": _PHP, "C": _C, "C++": _C, "Swift": _SWIFT,
}


def _leading_comment(text: str) -> str:
    for line in text.split("\n")[:15]:
        s = line.strip()
        if not s or s.startswith("#!"):
            continue
        if s.startswith(("//", "#", "/*", "*", "--")):
            body = s.lstrip("/#*- ").rstrip("*/ ").strip()
            low = body.lower()
            if len(body) >= 8 and "copyright" not in low and "license" not in low:
                return body[:160]
            continue
        break
    return ""


def _python_analysis(text: str) -> Tuple[List[Dict[str, Any]], str, Set[str]]:
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError, RecursionError):
        return [], "", set()
    doc = (ast.get_docstring(tree) or "").strip().split("\n")[0][:160]
    symbols: List[Dict[str, Any]] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            symbols.append({"name": node.name, "kind": "function", "line": node.lineno, "end": node.end_lineno})
        elif isinstance(node, ast.ClassDef):
            symbols.append({"name": node.name, "kind": "class", "line": node.lineno, "end": node.end_lineno})
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    symbols.append({"name": f"{node.name}.{sub.name}", "kind": "method",
                                    "line": sub.lineno, "end": sub.end_lineno})
    imports: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imports.add(node.module.split(".")[0])
    return symbols[:MAX_SYMBOLS], doc, imports


def _regex_symbols(lang: str, text: str) -> List[Dict[str, Any]]:
    patterns = _PATTERNS.get(lang)
    if not patterns:
        return []
    out: List[Dict[str, Any]] = []
    for number, line in enumerate(text.split("\n"), start=1):
        for kind, rx in patterns:
            match = rx.match(line)
            if match:
                out.append({"name": match.group(1), "kind": kind, "line": number, "end": None})
                break
        if len(out) >= MAX_SYMBOLS:
            break
    return out


# ------------------------------------------------------------ entry points
_ENTRY_NAMES = {"main.py", "__main__.py", "app.py", "manage.py", "index.js", "main.js", "server.js",
                "app.js", "index.ts", "main.ts", "server.ts", "main.go", "main.rs", "Main.java",
                "Program.cs", "main.c", "main.cpp", "main.kt"}


def detect_entry(path: str, text: str, lang: Optional[str]) -> List[str]:
    name = path.rsplit("/", 1)[-1]
    reasons: List[str] = []
    if lang == "Python":
        if name == "__main__.py":
            reasons.append("package entry file (runs with `python -m`)")
        if re.search(r"""^if\s+__name__\s*==\s*["']__main__["']""", text, re.M):
            reasons.append('has an `if __name__ == "__main__"` block')
        if name == "manage.py":
            reasons.append("Django management script")
        if re.search(r"\bFastAPI\(", text):
            reasons.append("creates a FastAPI app")
        if re.search(r"\bFlask\(\s*__name__", text):
            reasons.append("creates a Flask app")
    elif lang in ("JavaScript", "TypeScript"):
        if re.search(r"\.listen\(", text):
            reasons.append("starts an HTTP server (`.listen`)")
        if name in _ENTRY_NAMES:
            reasons.append("conventional entry file name")
    elif lang == "Go":
        if re.search(r"^package\s+main\b", text, re.M) and re.search(r"^func\s+main\(", text, re.M):
            reasons.append("Go `main` package")
    elif lang == "Rust":
        if re.search(r"^\s*fn\s+main\(", text, re.M):
            reasons.append("Rust `main` function")
    elif lang == "Java":
        if re.search(r"public\s+static\s+void\s+main\s*\(", text):
            reasons.append("Java `main` method")
    elif lang == "Kotlin":
        if re.search(r"^\s*fun\s+main\(", text, re.M):
            reasons.append("Kotlin `main` function")
    elif lang in ("C", "C++"):
        if re.search(r"^\s*int\s+main\s*\(", text, re.M):
            reasons.append("C/C++ `main` function")
    elif lang == "C#":
        if re.search(r"static\s+(?:async\s+)?(?:void|int|Task)\s+Main\s*\(", text):
            reasons.append("C# `Main` method")
    return reasons


# ----------------------------------------------------------------- public
def analyze_file(path: str, text: str) -> Dict[str, Any]:
    lang = language_of(path)
    lines = text.count("\n") + (0 if text.endswith("\n") or not text else 1)
    imports: Set[str] = set()
    if lang == "Python":
        symbols, doc, imports = _python_analysis(text)
    else:
        symbols, doc = _regex_symbols(lang or "", text), _leading_comment(text)
    info: Dict[str, Any] = {"lang": lang, "lines": lines, "doc": doc, "symbols": symbols}
    entry = detect_entry(path, text, lang)
    if entry:
        info["entry"] = entry
    if imports:
        info["imports"] = sorted(imports)
    return info


def is_stdlib(name: str) -> bool:
    names = getattr(sys, "stdlib_module_names", None)  # Python 3.10+
    if names is not None:
        return name in names
    if name in sys.builtin_module_names:
        return True
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError, AttributeError):
        return False
    if spec is None:
        return False
    origin = spec.origin or ""
    if origin in ("built-in", "frozen"):
        return True
    return bool(origin) and "site-packages" not in origin and "dist-packages" not in origin
