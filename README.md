# RepoExplain

Understand an unfamiliar code repository with **local** language models: what it does, where things
are implemented, what each piece means, and which dependencies and technologies it uses.
No accounts, no API keys, no cloud: your code never leaves your machine. Free and open source (MIT).

```
$ repoexplain index ./my-project
$ repoexplain ask "What does this project do?"
$ repoexplain ask "Where is the password checked?"
$ repoexplain ask "Explain handle_login in simple words"
$ repoexplain deps
```

Answers cite the files and line ranges they came from. If the evidence does not contain the answer,
the assistant says so instead of guessing.

## What it can answer

| You ask | What happens |
|---|---|
| "What does this project do?", "What are its features?", "Where does it start?" | **overview** route: answers from facts extracted at index time (README, manifests, entry points, file list) plus a few code excerpts |
| "What dependencies / technologies / languages are used?" | **deps** route: an exact report from `pyproject.toml`, `requirements.txt`, `package.json`, `go.mod`, `Cargo.toml`, `pom.xml`, `Gemfile`, `composer.json`, config files and imports. No LLM involved, so nothing is invented |
| "Where is X implemented?" | **locate** route: symbol lookup (functions and classes with line numbers) plus semantic code search |
| "Explain `some_function`" / "Explain parser.py" | **explain** route: reads the exact function or file and explains it step by step in plain words |
| Anything else about how the code works | **search** route: semantic search over code chunks |

Use `--trace` to see the route and tools used, and `--route overview|deps|locate|explain|search` to force one.

## Requirements

| | |
|---|---|
| OS | macOS 13 (Ventura)+, Windows 10+, Linux |
| Python | 3.9 or newer |
| RAM | 8 GB minimum (a 4B model at 4-bit quantisation uses about 2.5 GB) |
| Model backend | **Ollama**, or any OpenAI-compatible local server (LM Studio, llama.cpp server, ...) |

> Your model backend has its own OS requirements, which can be stricter than this tool's.
> Check the Ollama or LM Studio website for the versions they currently support.

## Install

```bash
git clone https://github.com/eight4bit/repoexplain.git && cd repoexplain
python3 -m venv .venv
source .venv/bin/activate          # Windows (PowerShell): .venv\Scripts\Activate.ps1
pip install -e .                   # -e lets you edit the code and see changes immediately
```

Using Ollama (default):

```bash
ollama pull qwen3:4b-instruct-2507-q4_K_M
ollama pull qwen3-embedding:0.6b
ollama pull qwen2.5-coder:1.5b     # only for optional file summaries
repoexplain doctor
```

## Commands

```bash
repoexplain index path/to/repo               # embeds code, builds the project overview; re-runs only redo changed files
repoexplain index path/to/repo --summarize   # also writes one-line file summaries with the small coder model
repoexplain ask "question" --repo path/to/repo [--trace] [--show-context] [--route R] [--no-verify]
repoexplain chat --repo path/to/repo         # interactive, remembers the last few turns
repoexplain overview --repo path/to/repo     # print the project overview (no LLM needed)
repoexplain deps --repo path/to/repo         # dependencies, languages, technologies (no LLM needed)
repoexplain doctor                           # check Python, RAM and models
repoexplain config --init                    # write repoexplain.yaml to customise everything
```

After upgrading from an older version, run `repoexplain index <path>` once to build the overview.
The index lives in `<repo>/.repoexplain/` and is never committed (it has its own `.gitignore`).

## How it works

```
question -> router (plan) -> tools (gather evidence) -> LLM (explain) -> grounding check
```

1. **Index time (no LLM):** scan files (respecting `.gitignore`; skipping binaries, lockfiles, symlinks and
   secret-like files), split code into chunks along function/class boundaries, embed the chunks, and build the
   **overview**: languages, dependencies, technologies, entry points (e.g. the `main()` behind a console script),
   symbols with line numbers, and the README's purpose paragraph.
2. **Router:** transparent rules decide the route (optionally the LLM decides: `routing.mode: llm`).
3. **Tools:** `project_overview`, `search_code`, `find_symbols`, `read_file` (only indexed files, so no `.env`,
   no path traversal), `list_tree`. They are deterministic; small models are unreliable at free-form tool calls.
4. **LLM:** explains the gathered evidence, citing `[1]`, `[2]`, ... The embedder is unloaded before the LLM
   runs, so only one model is in RAM at a time.
5. **Grounding check:** names in `backticks` and citation numbers in the answer are checked against the evidence;
   anything unsupported is flagged with a "Heads-up" line. It costs no extra model call.

Vector search is exact cosine similarity over a NumPy array: no database server, identical on every OS.

## Use a different backend (no Ollama)

`repoexplain config --init`, then edit `repoexplain.yaml`. For LM Studio:

```yaml
llm:
  provider: openai_compatible
  model: <model name shown in LM Studio>
  base_url: http://localhost:1234/v1
embeddings:
  provider: openai_compatible
  model: <an embedding model loaded in LM Studio>
  base_url: http://localhost:1234/v1
  query_prefix: ""        # only Qwen3-Embedding wants an instruction prefix
```

Changing the embedding model requires `repoexplain index --rebuild`. Config is read from (later wins): built-in
defaults, the user config file (`~/.config/repoexplain/config.yaml`,
`~/Library/Application Support/repoexplain/config.yaml` or `%APPDATA%\repoexplain\config.yaml`),
`<repo>/repoexplain.yaml`, and `--config FILE`.

## Tests

```bash
python -m unittest discover -s tests
```

The tests use a fake model server, so they need neither Ollama nor a network.

## Limitations

- A 4B model can still misread code or reason badly over correct evidence; the grounding check catches invented
  names, not wrong logic. Always check the cited sources.
- Technology detection relies on a built-in table of common packages plus config-file markers; unknown packages
  are listed without a description.
- Symbols and entry points are exact for Python (via `ast`) and approximate (regex) for other languages.
- Only the repository's root `.gitignore` is read.
- Not yet implemented: a LangGraph version of the agent, evaluation script, GUI, MCP server.

## License

MIT, see `LICENSE`. Models are downloaded separately and have their own licenses.
