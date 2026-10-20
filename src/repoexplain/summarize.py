"""Optional: one-line file summaries written by a small local model at index time."""
from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

from .analyzer import is_code
from .providers import BackendError

SYSTEM = (
    "You summarise source files for a developer. In one or two plain sentences "
    "(at most 35 words) say what the file is for and its main job. No preamble, no code."
)


def add_summaries(overview: Dict[str, Any], texts: Dict[str, str], previous: Dict[str, Tuple[str, str]],
                  provider, cfg: Dict[str, Any], log: Callable[[str], None] = print) -> int:
    """Add a 'summary' to file entries. Reuses cached summaries for unchanged files."""
    scfg = cfg["summarizer"]
    files = overview["files"]
    candidates = [p for p, i in files.items() if is_code(i.get("lang")) and i.get("lines", 0) >= 8 and p in texts]
    candidates.sort(key=lambda p: (0 if files[p].get("entry") else 1, -files[p].get("lines", 0), p))

    made = reused = 0
    options = {"num_ctx": 4096, "num_predict": 80, "temperature": 0.1}
    for path in candidates:
        info = files[path]
        cached = previous.get(path)
        if cached and cached[0] == info.get("hash") and cached[1]:
            info["summary"] = cached[1]
            reused += 1
            continue
        if made >= scfg["max_files"]:
            break
        messages = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": f"File: {path}\n\n{texts[path][:scfg['max_chars']]}"}]
        try:
            text = "".join(provider.chat_stream(scfg["model"], messages, options))
        except BackendError as exc:
            log(f"  summarizer stopped: {exc}")
            break
        text = " ".join(text.split())[:300]
        if text:
            info["summary"] = text
            made += 1
            log(f"  summarised {path} ({made}/{min(len(candidates), scfg['max_files'])})")
    if made:
        provider.unload(scfg["model"], kind="chat")
    if reused:
        log(f"  reused {reused} cached summaries")
    return made
