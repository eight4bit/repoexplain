"""Find the files worth indexing, and (importantly) skip the ones that must not be."""
from __future__ import annotations

import fnmatch
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List

import pathspec

ALWAYS_IGNORE_DIRS = {
    ".git", ".hg", ".svn", "node_modules", "venv", ".venv", "env", "__pycache__",
    ".mypy_cache", ".pytest_cache", ".tox", "dist", "build", "target", ".idea",
    ".vscode", ".gradle", ".next", ".repoexplain",
}

BINARY_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".webp", ".svg", ".psd",
    ".pdf", ".zip", ".tar", ".gz", ".tgz", ".7z", ".rar", ".jar", ".war",
    ".class", ".pyc", ".pyo", ".so", ".dll", ".dylib", ".exe", ".o", ".a",
    ".bin", ".dat", ".db", ".sqlite", ".woff", ".woff2", ".ttf", ".otf", ".eot",
    ".mp3", ".mp4", ".mov", ".avi", ".wav", ".map",
}

LOCKFILES = {
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock",
    "cargo.lock", "pipfile.lock", "composer.lock", "gemfile.lock",
}

# Conservative on purpose: private keys and credentials should never reach an index.
SECRET_PATTERNS = (
    ".env", ".env.*", "*.pem", "*.key", "*.p12", "*.pfx", "*.keystore", "*.jks",
    "id_rsa*", "id_ed25519*", "id_ecdsa*", "credentials*", "secrets.*", "*.kdbx",
    ".npmrc", ".pypirc", ".netrc",
)
SAFE_ENV_NAMES = {".env.example", ".env.sample", ".env.template"}


def looks_secret(filename: str) -> bool:
    name = filename.lower()
    if name in SAFE_ENV_NAMES:
        return False
    return any(fnmatch.fnmatch(name, pattern) for pattern in SECRET_PATTERNS)


@dataclass
class ScanResult:
    files: List[str] = field(default_factory=list)  # repo-relative, forward slashes
    skipped: Dict[str, int] = field(default_factory=dict)


def scan_repo(root: Path, max_kb: int = 200, extra_ignore: Iterable[str] = ()) -> ScanResult:
    root = Path(root).resolve()
    patterns = list(extra_ignore)
    gitignore = root / ".gitignore"
    if gitignore.is_file():
        try:
            patterns += gitignore.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            pass
    try:  # pathspec >= 1.0 renamed the pattern style; older versions only know "gitwildmatch"
        spec = pathspec.PathSpec.from_lines("gitignore", patterns)
    except (KeyError, LookupError, ValueError):
        spec = pathspec.PathSpec.from_lines("gitwildmatch", patterns)

    skipped: Dict[str, int] = defaultdict(int)
    files: List[str] = []

    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = Path(dirpath).relative_to(root)
        kept = []
        for d in sorted(dirnames):
            rel = (rel_dir / d).as_posix()
            full = os.path.join(dirpath, d)
            if d in ALWAYS_IGNORE_DIRS or os.path.islink(full) or spec.match_file(rel + "/"):
                skipped["ignored folders"] += 1
                continue
            kept.append(d)
        dirnames[:] = kept

        for name in sorted(filenames):
            full = os.path.join(dirpath, name)
            rel = (rel_dir / name).as_posix()
            ext = os.path.splitext(name)[1].lower()

            if os.path.islink(full):
                skipped["symlinks"] += 1
            elif looks_secret(name):
                skipped["secret-like files"] += 1
            elif spec.match_file(rel):
                skipped["gitignored"] += 1
            elif ext in BINARY_EXTENSIONS or name.lower() in LOCKFILES or name.lower().endswith(".min.js"):
                skipped["binary/lock/minified"] += 1
            else:
                try:
                    size = os.path.getsize(full)
                    if size == 0:
                        continue
                    if size > max_kb * 1024:
                        skipped["too large"] += 1
                        continue
                    with open(full, "rb") as handle:
                        if b"\x00" in handle.read(4096):
                            skipped["binary/lock/minified"] += 1
                            continue
                except OSError:
                    skipped["unreadable"] += 1
                    continue
                files.append(rel)

    return ScanResult(files=files, skipped=dict(skipped))
