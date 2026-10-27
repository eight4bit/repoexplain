"""Configuration loading.

Order of precedence (later wins):
    built-in defaults < user config < <repo>/repoexplain.yaml < --config FILE
"""
from __future__ import annotations

import copy
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

DEFAULTS: Dict[str, Any] = {
    "llm": {
        "provider": "ollama",  # ollama | openai_compatible (LM Studio, llama.cpp server, ...)
        "model": "qwen3:4b-instruct-2507-q4_K_M",
        "base_url": "http://localhost:11434",
        "api_key": "",  # only needed by servers that require one; local servers usually don't
        "num_ctx": 4096,  # Ollama only; other servers set context size at launch
        "num_predict": 400,  # max tokens in an answer
        "temperature": 0.1,
    },
    "embeddings": {
        "provider": "ollama",
        "model": "qwen3-embedding:0.6b",
        "base_url": "http://localhost:11434",
        "api_key": "",
        "batch_size": 16,
        # Qwen3-Embedding works best when queries carry a task instruction.
        # Set to "" if you switch to an embedding model that does not use one.
        "query_prefix": (
            "Instruct: Given a question about a code repository, "
            "retrieve the code that answers it\nQuery: "
        ),
    },
    "indexing": {
        "chunk_chars": 1000,
        "overlap_lines": 3,
        "max_file_kb": 200,
        "extra_ignore": [],  # extra .gitignore-style patterns
    },
    "retrieval": {
        "top_k": 4,
        "min_score": 0.0,  # raise (e.g. 0.3) to refuse weakly related questions
        "max_context_chars": 6500,  # evidence budget; keeps prompts inside num_ctx
    },
    "summarizer": {
        # Optional (index --summarize): a small model writes a one-line summary per file.
        "enabled": False,
        "provider": "ollama",
        "model": "qwen2.5-coder:1.5b",
        "base_url": "http://localhost:11434",
        "api_key": "",
        "max_files": 60,
        "max_chars": 3000,
    },
    "routing": {"mode": "rules"},  # rules | llm
    "history": {"turns": 4},  # chat memory: how many previous turns to remember
    "verify": {"grounding": True},  # flag names in answers that were not in the evidence
    "memory": {
        # Unload the embedding model before generating, so only one model is in RAM.
        "unload_embedder": True,
    },
    "server": {
        # Start a local OpenAI-compatible server (e.g. llama.cpp) when none is running.
        # Only used for localhost URLs. Leave command empty to disable.
        "autostart": True,
        "command": "",
        "stop_on_exit": False,
        "startup_timeout": 60,
    },
}



class ConfigError(RuntimeError):
    pass


def user_config_dir() -> Path:
    if sys.platform.startswith("win"):
        base = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
        return Path(base) / "repoexplain"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "repoexplain"
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "repoexplain"


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def _read_yaml(path: Path) -> Dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"Could not read config file {path}: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"Config file {path} must contain a YAML mapping.")
    return data


def load_config(repo_root: Optional[Path] = None, explicit: Optional[Path] = None) -> Dict[str, Any]:
    cfg = copy.deepcopy(DEFAULTS)
    candidates = [user_config_dir() / "config.yaml"]
    if repo_root is not None:
        candidates.append(Path(repo_root) / "repoexplain.yaml")
    if explicit is not None:
        if not Path(explicit).is_file():
            raise ConfigError(f"Config file not found: {explicit}")
        candidates.append(Path(explicit))
    for path in candidates:
        if path.is_file():
            _deep_merge(cfg, _read_yaml(path))
    return cfg


def dump_defaults() -> str:
    return yaml.safe_dump(DEFAULTS, sort_keys=False, allow_unicode=True)
