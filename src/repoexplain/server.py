"""Start the local model server on demand, so `rx` works without a manual step.

Auto-start only happens when ALL of these are true:
  * server.autostart is true (the default), and neither --no-autostart nor the
    REPOEXPLAIN_NO_AUTOSTART environment variable is set
  * server.command is set
  * the llm backend is an OpenAI-compatible server on localhost
  * nothing is answering at that address yet

It never touches a server that is already running, and it only ever stops a
server that it started itself.
"""
from __future__ import annotations

import atexit
import os
import shlex
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional
from urllib.parse import urlparse

import requests

from .providers import BackendError

LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}
_started: Optional[subprocess.Popen] = None


def log_path() -> Path:
    return Path(tempfile.gettempdir()) / "repoexplain-server.log"


def disabled_by_env() -> bool:
    value = os.environ.get("REPOEXPLAIN_NO_AUTOSTART", "").strip().lower()
    return value not in ("", "0", "false", "no")


def _models_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    return (base if base.endswith("/v1") else base + "/v1") + "/models"


def is_up(base_url: str, api_key: str = "") -> bool:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        return requests.get(_models_url(base_url), headers=headers, timeout=2).status_code < 500
    except requests.RequestException:
        return False


def reason_off(cfg: Dict[str, Any], no_autostart: bool = False) -> Optional[str]:
    """Why auto-start will not run, or None if it is eligible. Used by `doctor` too."""
    srv = cfg.get("server", {})
    llm = cfg["llm"]
    if no_autostart:
        return "disabled by --no-autostart"
    if disabled_by_env():
        return "disabled by REPOEXPLAIN_NO_AUTOSTART"
    if not srv.get("autostart", True):
        return "disabled by server.autostart: false"
    if not (srv.get("command") or "").strip():
        return "no server.command set"
    if llm.get("provider") == "ollama":
        return "llm provider is ollama"
    if urlparse(llm["base_url"]).hostname not in LOCAL_HOSTS:
        return "llm base_url is not a local address"
    return None


def ensure_server(cfg: Dict[str, Any], no_autostart: bool = False,
                  echo: Callable[[str], None] = print) -> bool:
    """Make sure the model server is reachable. Returns True if we started it."""
    global _started
    if reason_off(cfg, no_autostart) is not None:
        return False
    llm = cfg["llm"]
    if is_up(llm["base_url"], llm.get("api_key", "")):
        return False  # already running (ours or someone else's): leave it alone

    srv = cfg["server"]
    args = [os.path.expanduser(a) for a in shlex.split(srv["command"], posix=(os.name != "nt"))]
    if shutil.which(args[0]) is None:
        raise BackendError(
            f"Cannot auto-start: '{args[0]}' was not found on PATH. Install it, or set "
            "`server.autostart: false` and start your model server yourself."
        )

    echo("Starting model server (disable with --no-autostart)...")
    if os.name == "nt":
        kwargs: Dict[str, Any] = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    else:
        kwargs = {"start_new_session": True}  # Ctrl+C in rx must not kill it
    with open(log_path(), "ab") as log:
        proc = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=log,
                                stderr=subprocess.STDOUT, **kwargs)
    _started = proc
    if srv.get("stop_on_exit"):
        atexit.register(stop_server)

    timeout = float(srv.get("startup_timeout", 60))
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise BackendError(
                f"The model server exited right away (code {proc.returncode}). "
                f"See {log_path()}. A common cause is that the port is already in use."
            )
        if is_up(llm["base_url"], llm.get("api_key", "")):
            echo("Model server ready.")
            return True
        time.sleep(0.5)
    raise BackendError(f"The model server did not respond within {timeout:.0f}s. See {log_path()}.")


def stop_server() -> None:
    """Stop the server, but only if this process started it."""
    global _started
    proc, _started = _started, None
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
