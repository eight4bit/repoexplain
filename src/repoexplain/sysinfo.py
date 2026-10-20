"""`repoexplain doctor`: check the machine and the model backends."""
from __future__ import annotations

import os
import platform
import subprocess
import sys
from typing import Any, Dict, Optional

from .providers import BackendError, make_provider

MIN_RAM_GB = 8
OK, WARN, FAIL = "[ok]", "[!!]", "[xx]"


def total_ram_gb() -> Optional[float]:
    try:
        if sys.platform.startswith("win"):
            import ctypes

            class MemStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MemStatus()
            stat.dwLength = ctypes.sizeof(stat)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))  # type: ignore[attr-defined]
            return stat.ullTotalPhys / 1024 ** 3
        if sys.platform == "darwin":
            out = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5)
            return int(out.stdout.strip()) / 1024 ** 3
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024 ** 3
    except Exception:
        return None


def _model_available(wanted: str, installed: list) -> bool:
    return wanted in installed or f"{wanted}:latest" in installed


def run_doctor(cfg: Dict[str, Any]) -> int:
    problems = 0
    version = sys.version_info
    if version >= (3, 9):
        print(f"{OK} Python {version.major}.{version.minor}.{version.micro}")
    else:
        print(f"{FAIL} Python {version.major}.{version.minor} found; 3.9 or newer is required")
        problems += 1

    print(f"{OK} OS: {platform.system()} {platform.release()} ({platform.machine()})")

    ram = total_ram_gb()
    if ram is None:
        print(f"{WARN} Could not detect RAM (recommended: {MIN_RAM_GB} GB or more)")
    elif ram < MIN_RAM_GB - 0.5:
        print(f"{WARN} RAM: {ram:.1f} GB is below the recommended {MIN_RAM_GB} GB; expect slowness or failures")
    else:
        print(f"{OK} RAM: {ram:.1f} GB")

    checks = [("LLM", "llm"), ("Embeddings", "embeddings")]
    if cfg.get("summarizer", {}).get("enabled"):
        checks.append(("Summarizer", "summarizer"))
    for label, key in checks:
        section = cfg[key]
        wanted = section["model"]
        try:
            provider = make_provider(section)
            installed = provider.list_models()
        except BackendError as exc:
            print(f"{FAIL} {label} backend ({section['provider']} at {section['base_url']}): {exc}")
            print("     Start your model server (e.g. open Ollama or LM Studio) or edit repoexplain.yaml.")
            problems += 1
            continue
        if _model_available(wanted, installed):
            print(f"{OK} {label} model available: {wanted}")
        elif section["provider"] == "ollama":
            print(f"{FAIL} {label} model '{wanted}' is not installed. Run: ollama pull {wanted}")
            problems += 1
        else:
            print(f"{WARN} {label} model '{wanted}' not listed by the server; it may still work if loaded on demand")

    print()
    print("Everything looks good." if problems == 0 else f"{problems} problem(s) found. Fix them and run doctor again.")
    return 0 if problems == 0 else 1
