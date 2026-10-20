"""Command-line interface: index, ask, chat, overview, deps, doctor, config."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import __version__
from .agent import Agent, Turn, source_label
from .config import ConfigError, dump_defaults, load_config
from .indexer import index_repo
from .overview import load_overview, render_deps_report, render_overview
from .providers import BackendError, make_provider
from .router import ROUTES
from .store import INDEX_DIRNAME, VectorStore
from .sysinfo import run_doctor
from .verify import grounding_warnings


def _fix_stdout() -> None:
    """Avoid UnicodeEncodeError on Windows consoles."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except (AttributeError, ValueError):
            pass


def _config_for(args: argparse.Namespace, root: Optional[Path]) -> Dict[str, Any]:
    explicit = Path(args.config) if getattr(args, "config", None) else None
    return load_config(root, explicit)


def _load_store(root: Path, cfg: Dict[str, Any]) -> Optional[VectorStore]:
    store = VectorStore(root / INDEX_DIRNAME)
    if not store.load():
        print(f"No index found for {root}. Run:  repoexplain index {root}", file=sys.stderr)
        return None
    if store.meta.get("embed_model") != cfg["embeddings"]["model"]:
        print(
            f"The index was built with '{store.meta.get('embed_model')}' but the config uses "
            f"'{cfg['embeddings']['model']}'. Run:  repoexplain index {root} --rebuild",
            file=sys.stderr,
        )
        return None
    return store


def _load_overview(root: Path) -> Optional[Dict[str, Any]]:
    overview = load_overview(root / INDEX_DIRNAME)
    if overview is None:
        print(f"No project overview found for {root}. Run:  repoexplain index {root}", file=sys.stderr)
    return overview


def _make_agent(root: Path, cfg: Dict[str, Any]) -> Optional[Agent]:
    store = _load_store(root, cfg)
    overview = _load_overview(root)
    if store is None or overview is None:
        return None
    return Agent(root, cfg, store, overview, make_provider(cfg["embeddings"]), make_provider(cfg["llm"]))


def _ask(agent: Agent, question: str, history: List[Turn], args: argparse.Namespace) -> str:
    result = agent.answer(question, history, forced_route=getattr(args, "route", None))
    if args.trace:
        print("\n--- how I answered ---")
        for step in result.steps:
            print(f"  * {step}")
    if args.show_context:
        print("\n--- evidence ---")
        for number, e in enumerate(result.evidence, start=1):
            print(f"[{number}] {source_label(e)}")
            if e.text:
                print(e.text)
            print()
    print()
    pieces: List[str] = []
    for piece in result.stream:
        pieces.append(piece)
        print(piece, end="", flush=True)
    answer = "".join(pieces)
    print("\n")

    if result.evidence:
        print("Sources:")
        for number, e in enumerate(result.evidence, start=1):
            print(f"  [{number}] {source_label(e)}")
    if (not result.direct) and agent.cfg["verify"]["grounding"] and not getattr(args, "no_verify", False):
        for warning in grounding_warnings(answer, result.context_text, result.known_names, len(result.evidence)):
            print(f"Heads-up: {warning}; double-check that part.")
    return answer


def cmd_index(args: argparse.Namespace) -> int:
    root = Path(args.path).resolve()
    if not root.is_dir():
        print(f"Not a folder: {root}", file=sys.stderr)
        return 2
    cfg = _config_for(args, root)
    embedder = make_provider(cfg["embeddings"])
    summarize = bool(args.summarize or cfg["summarizer"]["enabled"])
    summarizer = make_provider(cfg["summarizer"]) if summarize else None
    stats = index_repo(root, cfg, embedder, rebuild=args.rebuild, summarize=summarize, summarizer=summarizer)
    print(f"\nDone. {stats['total_chunks']} chunks indexed (~{stats['approx_tokens']:,} tokens of code) "
          f"in {root / INDEX_DIRNAME}")
    print(f"Overview: {', '.join(stats['languages']) or 'no code languages'}; {stats['dependencies']} dependencies, "
          f"{stats['technologies']} technologies, {stats['entry_points']} entry points detected.")
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    root = Path(args.repo).resolve()
    cfg = _config_for(args, root)
    if args.top_k:
        cfg["retrieval"]["top_k"] = args.top_k
    agent = _make_agent(root, cfg)
    if agent is None:
        return 2
    _ask(agent, args.question, [], args)
    return 0


def cmd_chat(args: argparse.Namespace) -> int:
    root = Path(args.repo).resolve()
    cfg = _config_for(args, root)
    agent = _make_agent(root, cfg)
    if agent is None:
        return 2
    history: List[Turn] = []
    print(f"Chatting about {root.name}. Follow-up questions are remembered. Type 'exit' to quit.")
    while True:
        try:
            question = input("\nyou> ").strip()
        except EOFError:
            print()
            return 0
        if question.lower() in {"exit", "quit", ":q"}:
            return 0
        if question:
            answer = _ask(agent, question, history, args)
            history.append(Turn(question, answer))
            del history[:-cfg["history"]["turns"]]


def cmd_overview(args: argparse.Namespace) -> int:
    root = Path(args.repo).resolve()
    overview = _load_overview(root)
    if overview is None:
        return 2
    print(render_overview(overview, max_chars=6000))
    return 0


def cmd_deps(args: argparse.Namespace) -> int:
    root = Path(args.repo).resolve()
    overview = _load_overview(root)
    if overview is None:
        return 2
    print(render_deps_report(overview))
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    return run_doctor(_config_for(args, Path(args.repo).resolve()))


def cmd_config(args: argparse.Namespace) -> int:
    if args.init:
        target = Path.cwd() / "repoexplain.yaml"
        if target.exists():
            print(f"{target} already exists; not overwriting.", file=sys.stderr)
            return 2
        target.write_text(dump_defaults(), encoding="utf-8")
        print(f"Wrote {target}. Edit it to change models, backends or chunk sizes.")
        return 0
    print(dump_defaults())
    return 0


def _add_answer_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--show-context", action="store_true", help="print the evidence given to the model")
    p.add_argument("--trace", action="store_true", help="show which route and tools were used")
    p.add_argument("--no-verify", action="store_true", help="skip the grounding check on the answer")
    p.add_argument("--route", choices=ROUTES, help="force a route instead of letting the router decide")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="repoexplain",
        description="Explain a code repository using local LLMs. No accounts, no cloud.",
    )
    parser.add_argument("--version", action="version", version=f"repoexplain {__version__}")
    parser.add_argument("--config", help="path to an extra YAML config file")
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("index", help="index (or update the index of) a repository")
    p.add_argument("path", nargs="?", default=".", help="repository folder (default: current folder)")
    p.add_argument("--rebuild", action="store_true", help="ignore the existing index and start over")
    p.add_argument("--summarize", action="store_true",
                   help="also write one-line file summaries with the small model (slower, better overviews)")
    p.set_defaults(func=cmd_index)

    p = sub.add_parser("ask", help="ask one question about an indexed repository")
    p.add_argument("question")
    p.add_argument("--repo", default=".", help="repository folder (default: current folder)")
    p.add_argument("--top-k", type=int, help="number of code chunks to retrieve")
    _add_answer_flags(p)
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("chat", help="interactive question loop with memory of recent turns")
    p.add_argument("--repo", default=".")
    _add_answer_flags(p)
    p.set_defaults(func=cmd_chat)

    p = sub.add_parser("overview", help="print the project overview (no LLM needed)")
    p.add_argument("--repo", default=".")
    p.set_defaults(func=cmd_overview)

    p = sub.add_parser("deps", help="print dependencies, languages and technologies (no LLM needed)")
    p.add_argument("--repo", default=".")
    p.set_defaults(func=cmd_deps)

    p = sub.add_parser("doctor", help="check Python, RAM and model backends")
    p.add_argument("--repo", default=".")
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("config", help="show the default config, or write one with --init")
    p.add_argument("--init", action="store_true", help="write repoexplain.yaml in the current folder")
    p.set_defaults(func=cmd_config)
    return parser


def main(argv=None) -> int:
    _fix_stdout()
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 0
    try:
        return args.func(args) or 0
    except (BackendError, ConfigError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    except BrokenPipeError:  # e.g. `repoexplain config | head`
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
