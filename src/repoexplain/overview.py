"""Build, save and render the project overview.

The overview is created at index time from facts only (manifests, file analysis,
README), so questions like "what technologies does it use?" or "where does it
start?" are answered from data, not guessed by a small model.
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from .analyzer import NON_CODE, is_code, is_stdlib
from .manifests import classify_dependency, detect_markers, parse_manifest

OVERVIEW_FILE = "overview.json"
OVERVIEW_VERSION = 1


# ------------------------------------------------------------ persistence
def save_overview(directory: Path, overview: Dict[str, Any]) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    tmp = directory / (OVERVIEW_FILE + ".tmp")
    tmp.write_text(json.dumps(overview, indent=1, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, directory / OVERVIEW_FILE)


def load_overview(directory: Path) -> Optional[Dict[str, Any]]:
    path = Path(directory) / OVERVIEW_FILE
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if data.get("version") == OVERVIEW_VERSION else None


# ---------------------------------------------------------------- helpers
def readme_purpose(text: str) -> str:
    """First plain-text paragraph of a README (skips headings, badges, code, tables)."""
    paragraph: List[str] = []
    in_fence = False
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        skip = (not line or line.startswith(("#", "[![", "![", "<", "|", ">", "---", "===")))
        if skip:
            if paragraph and not line.startswith("<"):
                break
            continue
        paragraph.append(line)
    cleaned = " ".join(paragraph)
    cleaned = re.sub(r"[*_`]", "", cleaned)
    cleaned = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", cleaned)
    return cleaned[:500].strip()


def resolve_module(module: str, files: List[str]) -> Optional[str]:
    tail = "/".join(module.split("."))
    for suffix in (tail + ".py", tail + "/__init__.py"):
        for rel in files:
            if rel == suffix or rel.endswith("/" + suffix):
                return rel
    return None


def _find_symbol_line(info: Dict[str, Any], name: str) -> Optional[int]:
    for symbol in info.get("symbols", []):
        if symbol["name"] == name:
            return symbol["line"]
    return None


def _is_test_file(rel: str) -> bool:
    low = rel.lower()
    name = low.rsplit("/", 1)[-1]
    parts = low.split("/")
    return (any(p in ("tests", "test", "__tests__", "spec", "specs") for p in parts[:-1])
            or name.startswith("test_") or name.endswith(("_test.py", ".test.js", ".test.ts", ".spec.js", ".spec.ts", "_test.go")))


def _top_dirs(files: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[str, List[str]] = {}
    for rel in files:
        parts = rel.split("/")
        if len(parts) == 1:
            key = "(root)"
        elif parts[0] == "src" and len(parts) > 2:
            key = f"src/{parts[1]}"
        else:
            key = parts[0]
        groups.setdefault(key, []).append(rel)
    out = []
    for key in sorted(groups):
        langs = Counter(files[r]["lang"] for r in groups[key] if is_code(files[r].get("lang")))
        out.append({"dir": key, "files": len(groups[key]),
                    "languages": [lang for lang, _ in langs.most_common(2)]})
    return out


# --------------------------------------------------------------- building
def build_overview(root: Path, analyses: Dict[str, Dict[str, Any]], manifest_texts: Dict[str, str],
                   readme_text: str = "") -> Dict[str, Any]:
    root = Path(root)
    files: Dict[str, Dict[str, Any]] = {}
    imports_counter: Counter = Counter()
    for rel, info in analyses.items():
        info = dict(info)
        for name in info.pop("imports", []):
            imports_counter[name] += 1
        files[rel] = info

    # languages
    lang_files: Counter = Counter()
    lang_lines: Counter = Counter()
    for info in files.values():
        lang = info.get("lang")
        if is_code(lang):
            lang_files[lang] += 1
            lang_lines[lang] += info.get("lines", 0)
    languages = [{"language": lang, "files": lang_files[lang], "lines": lang_lines[lang]}
                 for lang, _ in lang_lines.most_common()]

    # manifests
    deps: List[Dict[str, str]] = []
    scripts: List[Dict[str, str]] = []
    entries: List[Dict[str, Any]] = []
    meta: Dict[str, Any] = {}
    manifest_files: List[str] = []
    for rel, text in sorted(manifest_texts.items()):
        try:
            parsed = parse_manifest(rel, text)
        except Exception:  # a broken manifest must never break indexing
            continue
        manifest_files.append(rel)
        deps.extend(parsed["deps"])
        scripts.extend(parsed["scripts"])
        entries.extend(parsed["entries"])
        for key, value in parsed["meta"].items():
            meta.setdefault(key, value)
    seen = set()
    unique_deps = []
    for dep in deps:
        key = (dep["name"].lower(), dep["ecosystem"], dep["group"])
        if key not in seen:
            seen.add(key)
            unique_deps.append(dep)

    # technologies
    techs: Dict[str, Dict[str, Any]] = {}
    for dep in unique_deps:
        hit = classify_dependency(dep["name"], dep["ecosystem"])
        if hit:
            label, category = hit
            entry = techs.setdefault(label, {"label": label, "category": category, "evidence": []})
            if dep["name"] not in entry["evidence"]:
                entry["evidence"].append(dep["name"])
    for marker in detect_markers(list(files)):
        entry = techs.setdefault(marker["label"], {"label": marker["label"], "category": marker["category"], "evidence": []})
        entry["evidence"].append(marker["evidence"])
    for tool in meta.get("tools", []):
        techs.setdefault(tool, {"label": tool, "category": "configured in pyproject.toml", "evidence": ["pyproject.toml"]})

    # entry points: declared commands first, then files that look like entry points
    resolved: List[Dict[str, Any]] = []
    for script in scripts:
        if script["kind"] == "console_script" and ":" in script["command"]:
            module, func = script["command"].split(":", 1)
            func = func.strip()
            path = resolve_module(module.strip(), list(files))
            line = _find_symbol_line(files[path], func) if path else None
            where = f"{path}:{line}" if path and line else (path or module)
            resolved.append({"text": f"The `{script['name']}` command runs `{func}()` in {where} "
                                     f"(declared in {script['source']})",
                             "file": path, "line": line})
        elif script["kind"] == "npm_script" and script["name"] in ("start", "dev", "serve", "build", "test"):
            continue  # shown under scripts instead
    for entry in entries:
        resolved.append({"text": entry["text"], "file": entry.get("file"), "line": None})
    for rel in sorted(files):
        if _is_test_file(rel):
            continue  # test files have __main__ blocks too; they are not how the program starts
        for reason in files[rel].get("entry", []):
            resolved.append({"text": f"{rel} {reason}", "file": rel, "line": None})
    entry_points = resolved[:12]

    # python imports that are neither stdlib nor part of the repo
    local_names = set()
    for rel in files:
        parts = rel.split("/")
        local_names.add(parts[0])
        local_names.add(os.path.splitext(parts[-1])[0])
        if len(parts) > 2 and parts[0] == "src":
            local_names.add(parts[1])
    third_party = [{"name": n, "files": c} for n, c in imports_counter.most_common()
                   if n not in local_names and not is_stdlib(n)][:20]

    tests = sum(1 for rel in files if _is_test_file(rel))
    total_lines = sum(info.get("lines", 0) for info in files.values() if is_code(info.get("lang")))
    name = meta.get("name") or root.name
    purpose = readme_purpose(readme_text) or str(meta.get("description", ""))

    return {
        "version": OVERVIEW_VERSION,
        "project": {
            "name": name, "version": meta.get("version", ""), "description": meta.get("description", ""),
            "purpose": purpose, "languages": languages, "total_files": len(files), "total_code_lines": total_lines,
            "dependencies": unique_deps, "manifests": manifest_files, "technologies": list(techs.values()),
            "scripts": scripts, "entry_points": entry_points, "third_party_imports": third_party,
            "top_dirs": _top_dirs(files), "test_files": tests,
            "requires_python": meta.get("requires-python", ""),
        },
        "files": files,
    }


# --------------------------------------------------------------- rendering
def file_line(path: str, info: Dict[str, Any]) -> str:
    desc = info.get("summary") or info.get("doc") or ""
    if not desc:
        names = [s["name"] for s in info.get("symbols", []) if s["kind"] != "method"][:6]
        if names:
            desc = "defines " + ", ".join(names)
    return f"{path}: {desc}" if desc else path


def _dep_names(deps: List[Dict[str, str]], groups: tuple) -> str:
    names = [d["name"] for d in deps if d["group"] in groups]
    return ", ".join(names[:18]) + (f" (+{len(names) - 18} more)" if len(names) > 18 else "")


def render_overview(overview: Dict[str, Any], max_chars: int = 3200) -> str:
    p = overview["project"]
    lines: List[str] = []
    title = p["name"] + (f" v{p['version']}" if p.get("version") else "")
    lines.append(f"PROJECT: {title}")
    if p.get("purpose"):
        lines.append(f"PURPOSE (from README/manifest): {p['purpose']}")
    if p["languages"]:
        lines.append("LANGUAGES: " + ", ".join(f"{l['language']} ({l['files']} files, {l['lines']} lines)"
                                              for l in p["languages"][:5]))
    if p["technologies"]:
        lines.append("TECHNOLOGIES: " + "; ".join(f"{t['label']} ({t['category']})" for t in p["technologies"][:14]))
    runtime = _dep_names(p["dependencies"], ("runtime",))
    if runtime:
        lines.append("RUNTIME DEPENDENCIES: " + runtime)
    if p["entry_points"]:
        lines.append("ENTRY POINTS:")
        lines.extend(f"- {e['text']}" for e in p["entry_points"][:6])
    npm = [s for s in p["scripts"] if s["kind"] == "npm_script"][:6]
    if npm:
        lines.append("NPM SCRIPTS: " + "; ".join(f"{s['name']}: {s['command']}" for s in npm))
    lines.append("STRUCTURE: " + "; ".join(f"{d['dir']} ({d['files']} files)" for d in p["top_dirs"][:12]))

    header = "\n".join(lines)
    budget = max_chars - len(header) - 40
    code_files, other_files = [], []
    for rel, info in sorted(overview["files"].items()):
        entry_first = 0 if info.get("entry") else 1
        (code_files if is_code(info.get("lang")) or rel.lower().startswith("readme") else other_files).append(
            (entry_first, rel, info))
    ordered = [t for t in sorted(code_files, key=lambda t: (t[0], t[1]))]
    listing: List[str] = []
    used = 0
    for index, (_, rel, info) in enumerate(ordered):
        text = file_line(rel, info)[:170]
        if used + len(text) + 3 > budget:
            listing.append(f"... and {len(ordered) - index} more code files")
            break
        listing.append("- " + text)
        used += len(text) + 3
    if listing:
        header += "\nFILES:\n" + "\n".join(listing)
    if len(header) > max_chars:  # core facts alone exceed the budget: cut at a line boundary
        header = header[:max_chars].rsplit("\n", 1)[0]
    return header


def render_deps_report(overview: Dict[str, Any]) -> str:
    p = overview["project"]
    out: List[str] = []
    if p["languages"]:
        out.append("Languages: " + ", ".join(f"{l['language']} ({l['files']} files)" for l in p["languages"][:6]))
    if p.get("requires_python"):
        out.append(f"Requires Python {p['requires_python']}")

    if p["dependencies"]:
        src = ", ".join(p["manifests"])
        out.append(f"\nDependencies (declared in {src}):")
        grouped: Dict[tuple, List[Dict[str, str]]] = {}
        for dep in p["dependencies"]:
            grouped.setdefault((dep["ecosystem"], dep["group"]), []).append(dep)
        for (eco, group), items in sorted(grouped.items()):
            out.append(f"  {eco} / {group}:")
            for dep in items:
                hit = classify_dependency(dep["name"], dep["ecosystem"])
                spec = f" {dep['spec']}" if dep["spec"] else ""
                note = f" - {hit[0]}: {hit[1]}" if hit else ""
                out.append(f"    {dep['name']}{spec}{note}")
    else:
        out.append("\nNo dependency manifest found (looked for pyproject.toml, requirements.txt, package.json, "
                   "go.mod, Cargo.toml, pom.xml, Gemfile, composer.json).")

    if p["third_party_imports"]:
        out.append("\nThird-party packages imported by the Python code: " +
                   ", ".join(f"{i['name']} ({i['files']} files)" for i in p["third_party_imports"][:12]))
    if p["technologies"]:
        out.append("\nTechnologies detected:")
        for t in p["technologies"]:
            evidence = ", ".join(t["evidence"][:3])
            out.append(f"  - {t['label']} ({t['category']}) - from {evidence}")
    return "\n".join(out).strip()
