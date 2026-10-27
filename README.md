# RepoExplain

Understand an unfamiliar code repository with **local** language models: what it does, where things
are implemented, what each piece means, and which dependencies and technologies it uses.
No accounts, no API keys, no cloud: your code never leaves your machine. Free and open source (MIT).
Works with **Ollama** or **llama.cpp** (and any other OpenAI-compatible local server).

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
| Model backend | **Ollama**, **llama.cpp** (`llama-server`), or any OpenAI-compatible local server (LM Studio, ...) |

> Your model backend has its own OS requirements, which can be stricter than this tool's.
> Check the Ollama, llama.cpp or LM Studio website for the versions they currently support.

## Install

```bash
git clone https://github.com/eight4bit/repoexplain.git && cd repoexplain
python3 -m venv .venv
source .venv/bin/activate          # Windows (PowerShell): .venv\Scripts\Activate.ps1
pip install -e .                   # -e lets you edit the code and see changes immediately
```

Then set up **one** model backend.

### Option A: Ollama (default, no configuration needed)

```bash
ollama pull qwen3:4b-instruct-2507-q4_K_M
ollama pull qwen3-embedding:0.6b
ollama pull qwen2.5-coder:1.5b     # only for optional file summaries
repoexplain doctor
```

### Option B: llama.cpp

Install llama.cpp (for example `brew install llama.cpp` on macOS; see the llama.cpp project for
Windows and Linux). You need a recent build with *router mode*, which serves several models from one
server: `llama-server --help` should list `--models-preset`.

1. Create a models file, for example `~/models.ini`. Models are downloaded from Hugging Face the first
   time they are used:

   ```ini
   version = 1

   [*]
   c = 4096
   np = 1
   fa = on

   [qwen3]
   hf-repo = unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_K_M

   [qwen3-embedding]
   hf-repo = Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0
   embeddings = true
   pooling = last

   [qwen2.5-coder]
   hf-repo = Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF:Q4_K_M
   ```

   The preset syntax is new and may change between llama.cpp versions, so check its documentation if
   the server rejects the file.

2. Point RepoExplain at it. Put this in your user config file (see [Configuration](#configuration)), or in
   `repoexplain.yaml` (create one with `repoexplain config --init`):

   ```yaml
   llm:
     provider: openai_compatible
     model: qwen3
     base_url: http://localhost:8080/v1
   embeddings:
     provider: openai_compatible
     model: qwen3-embedding
     base_url: http://localhost:8080/v1
   summarizer:               # only needed with `index --summarize`
     provider: openai_compatible
     model: qwen2.5-coder
     base_url: http://localhost:8080/v1
   server:
     command: llama-server --models-preset ~/models.ini --models-max 2 --port 8080
   ```

   The model names are the section names from `models.ini`. Context size is set by `c = 4096` in that file;
   the `num_ctx` setting only applies to Ollama.

3. Run `repoexplain index .` and `repoexplain ask "..."`. With `server.command` set, RepoExplain starts
   the server for you (see below). Without it, start the server yourself first:

   ```bash
   llama-server --models-preset ~/models.ini --models-max 2 --port 8080
   ```

> `repoexplain doctor` only checks the backend and never starts it, so start the server first (or run
> `index` / `ask` once) before running `doctor`.

### Automatic server start

When a local OpenAI-compatible server is configured and nothing is answering, RepoExplain can launch it
for you, so `index`, `ask` and `chat` work without a separate step (like Ollama's background service).

It starts a server only when **all** of these are true:

- `server.command` is set and `server.autostart` is not `false`
- the `llm` backend is `openai_compatible` and its `base_url` points to localhost
- nothing is already answering at that address

It never touches a server that is already running, never starts anything for a remote URL, and only stops a
server that it started itself. Server output goes to `repoexplain-server.log` in your temp folder.

```yaml
server:
  autostart: true          # false turns it off permanently
  command: llama-server --models-preset ~/models.ini --models-max 2 --port 8080
  stop_on_exit: false      # true stops the server when the command finishes (models reload every time)
  startup_timeout: 60      # seconds to wait for the server to answer
```

To turn it off: set `server.autostart: false`, pass `--no-autostart` for one run, or set the environment
variable `REPOEXPLAIN_NO_AUTOSTART=1`.

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

`index`, `ask` and `chat` also accept `--no-autostart` (see above).

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
4. **LLM:** explains the gathered evidence, citing `[1]`, `[2]`, ... With Ollama the embedder is unloaded
   before the LLM runs, so only one model is in RAM at a time. With llama.cpp, `--models-max` limits how many
   models stay loaded.
5. **Grounding check:** names in `backticks` and citation numbers in the answer are checked against the evidence;
   anything unsupported is flagged with a "Heads-up" line. It costs no extra model call.

Vector search is exact cosine similarity over a NumPy array: no database server, identical on every OS.

## Configuration

`repoexplain config --init` writes a `repoexplain.yaml` with every setting, for example to use LM Studio:

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
- Automatic server start only supports local OpenAI-compatible servers, and only watches the `llm` backend's address.
- Not yet implemented: a LangGraph version of the agent, evaluation script, GUI, MCP server.

## License

MIT, see `LICENSE`. Models are downloaded separately and have their own licenses.
