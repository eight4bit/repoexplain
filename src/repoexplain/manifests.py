"""Read dependency manifests and infer the technologies a project uses.

Supported: pyproject.toml, requirements*.txt, package.json, go.mod, Cargo.toml,
pom.xml, Gemfile, composer.json. All parsing is deterministic and offline.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

try:  # Python 3.11+
    import tomllib  # type: ignore
except ImportError:  # pragma: no cover
    try:
        import tomli as tomllib  # type: ignore
    except ImportError:
        tomllib = None  # type: ignore

MANIFEST_NAMES = {"pyproject.toml", "package.json", "go.mod", "cargo.toml", "pom.xml", "gemfile", "composer.json"}


def is_manifest(rel: str) -> bool:
    if rel.count("/") > 2:  # ignore deeply nested (vendored) manifests
        return False
    name = rel.rsplit("/", 1)[-1].lower()
    return name in MANIFEST_NAMES or (name.startswith("requirements") and name.endswith(".txt"))


def _dep(name: str, spec: str, group: str, ecosystem: str, source: str) -> Dict[str, str]:
    return {"name": name, "spec": (spec or "").strip(), "group": group, "ecosystem": ecosystem, "source": source}


def _empty() -> Dict[str, Any]:
    return {"deps": [], "meta": {}, "scripts": [], "entries": []}


# --------------------------------------------------------------- Python
_REQ_RX = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)\s*(?:\[[^\]]*\])?\s*(.*)$")


def _parse_requirement(line: str) -> Optional[Tuple[str, str]]:
    line = line.split("#", 1)[0].strip()
    if not line or line.startswith(("-", "git+", "http://", "https://", ".", "/")):
        return None
    line = line.split(";", 1)[0].strip()
    match = _REQ_RX.match(line)
    return (match.group(1), match.group(2).strip()) if match else None


def parse_requirements(text: str, source: str) -> Dict[str, Any]:
    out = _empty()
    group = "dev" if re.search(r"dev|test", source.lower()) else "runtime"
    for line in text.splitlines():
        parsed = _parse_requirement(line)
        if parsed:
            out["deps"].append(_dep(parsed[0], parsed[1], group, "python", source))
    return out


def parse_pyproject(text: str, source: str) -> Dict[str, Any]:
    out = _empty()
    if tomllib is None:
        return out
    data = tomllib.loads(text)
    proj = data.get("project", {}) or {}
    for key in ("name", "version", "description", "requires-python"):
        if proj.get(key):
            out["meta"][key] = proj[key]
    for req in proj.get("dependencies", []) or []:
        parsed = _parse_requirement(req)
        if parsed:
            out["deps"].append(_dep(parsed[0], parsed[1], "runtime", "python", source))
    for extra, reqs in (proj.get("optional-dependencies", {}) or {}).items():
        for req in reqs:
            parsed = _parse_requirement(req)
            if parsed:
                out["deps"].append(_dep(parsed[0], parsed[1], f"optional:{extra}", "python", source))
    for req in (data.get("build-system", {}) or {}).get("requires", []) or []:
        parsed = _parse_requirement(req)
        if parsed:
            out["deps"].append(_dep(parsed[0], parsed[1], "build", "python", source))
    for name, target in (proj.get("scripts", {}) or {}).items():
        out["scripts"].append({"name": name, "command": str(target), "source": source, "kind": "console_script"})

    tool = data.get("tool", {}) or {}
    poetry = tool.get("poetry", {}) or {}
    if poetry:
        for key in ("name", "version", "description"):
            if poetry.get(key) and key not in out["meta"]:
                out["meta"][key] = poetry[key]

        def add_poetry(table: Dict[str, Any], group: str) -> None:
            for name, spec in (table or {}).items():
                if name.lower() == "python":
                    continue
                version = spec if isinstance(spec, str) else (spec.get("version", "") if isinstance(spec, dict) else "")
                out["deps"].append(_dep(name, version, group, "python", source))

        add_poetry(poetry.get("dependencies", {}), "runtime")
        add_poetry(poetry.get("dev-dependencies", {}), "dev")
        for gname, gdata in (poetry.get("group", {}) or {}).items():
            add_poetry((gdata or {}).get("dependencies", {}), "dev" if gname in ("dev", "test") else gname)
        for name, target in (poetry.get("scripts", {}) or {}).items():
            out["scripts"].append({"name": name, "command": str(target), "source": source, "kind": "console_script"})
    tools = [t for t in tool if t in ("black", "ruff", "mypy", "pytest", "isort", "flake8", "pylint",
                                      "setuptools", "hatch", "poetry", "flit", "pdm", "maturin")]
    if tools:
        out["meta"]["tools"] = tools
    return out


# ----------------------------------------------------------- JavaScript
def parse_package_json(text: str, source: str) -> Dict[str, Any]:
    out = _empty()
    data = json.loads(text)
    for key in ("name", "version", "description", "main"):
        if data.get(key):
            out["meta"][key] = data[key]
    for field, group in (("dependencies", "runtime"), ("devDependencies", "dev"),
                         ("peerDependencies", "peer"), ("optionalDependencies", "optional")):
        for name, spec in (data.get(field, {}) or {}).items():
            out["deps"].append(_dep(name, str(spec), group, "npm", source))
    for name, command in (data.get("scripts", {}) or {}).items():
        out["scripts"].append({"name": name, "command": str(command), "source": source, "kind": "npm_script"})
    if data.get("main"):
        out["entries"].append({"text": f"`main` field in {source} points to {data['main']}", "file": data["main"]})
    bin_field = data.get("bin")
    if isinstance(bin_field, str):
        out["entries"].append({"text": f"`bin` in {source} exposes the command `{data.get('name', '?')}` -> {bin_field}",
                               "file": bin_field})
    elif isinstance(bin_field, dict):
        for cmd, target in bin_field.items():
            out["entries"].append({"text": f"`bin` in {source} exposes the command `{cmd}` -> {target}", "file": target})
    return out


# ------------------------------------------------------------------ Go
def parse_go_mod(text: str, source: str) -> Dict[str, Any]:
    out = _empty()
    match = re.search(r"^module\s+(\S+)", text, re.M)
    if match:
        out["meta"]["name"] = match.group(1)
    match = re.search(r"^go\s+(\S+)", text, re.M)
    if match:
        out["meta"]["go"] = match.group(1)
    in_block = False
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("require ("):
            in_block = True
            continue
        if in_block and s == ")":
            in_block = False
            continue
        candidate = None
        if in_block:
            candidate = s
        elif s.startswith("require "):
            candidate = s[len("require "):].strip()
        if candidate and "// indirect" not in candidate:
            parts = candidate.split()
            if len(parts) >= 2:
                out["deps"].append(_dep(parts[0], parts[1], "runtime", "go", source))
    return out


# ---------------------------------------------------------------- Rust
def parse_cargo(text: str, source: str) -> Dict[str, Any]:
    out = _empty()
    if tomllib is None:
        return out
    data = tomllib.loads(text)
    pkg = data.get("package", {}) or {}
    for key in ("name", "version", "description"):
        if pkg.get(key):
            out["meta"][key] = pkg[key]
    for table, group in (("dependencies", "runtime"), ("dev-dependencies", "dev"), ("build-dependencies", "build")):
        for name, spec in (data.get(table, {}) or {}).items():
            version = spec if isinstance(spec, str) else (spec.get("version", "") if isinstance(spec, dict) else "")
            out["deps"].append(_dep(name, version, group, "rust", source))
    for binary in data.get("bin", []) or []:
        out["entries"].append({"text": f"Cargo binary `{binary.get('name', '?')}` ({binary.get('path', 'src/main.rs')})",
                               "file": binary.get("path", "src/main.rs")})
    return out


# ---------------------------------------------------------------- Java
def parse_pom(text: str, source: str) -> Dict[str, Any]:
    out = _empty()
    for block in re.findall(r"<dependency>(.*?)</dependency>", text, re.S):
        group = re.search(r"<groupId>(.*?)</groupId>", block, re.S)
        artifact = re.search(r"<artifactId>(.*?)</artifactId>", block, re.S)
        version = re.search(r"<version>(.*?)</version>", block, re.S)
        scope = re.search(r"<scope>(.*?)</scope>", block, re.S)
        if artifact:
            name = f"{group.group(1).strip()}:{artifact.group(1).strip()}" if group else artifact.group(1).strip()
            kind = "test" if scope and scope.group(1).strip() == "test" else "runtime"
            out["deps"].append(_dep(name, version.group(1).strip() if version else "", kind, "maven", source))
    return out


# ---------------------------------------------------------------- Ruby
def parse_gemfile(text: str, source: str) -> Dict[str, Any]:
    out = _empty()
    for match in re.finditer(r"^\s*gem\s+['\"]([^'\"]+)['\"](?:\s*,\s*['\"]([^'\"]+)['\"])?", text, re.M):
        out["deps"].append(_dep(match.group(1), match.group(2) or "", "runtime", "ruby", source))
    return out


# ---------------------------------------------------------------- PHP
def parse_composer(text: str, source: str) -> Dict[str, Any]:
    out = _empty()
    data = json.loads(text)
    for key in ("name", "description"):
        if data.get(key):
            out["meta"][key] = data[key]
    for field, group in (("require", "runtime"), ("require-dev", "dev")):
        for name, spec in (data.get(field, {}) or {}).items():
            if name != "php" and not name.startswith("ext-"):
                out["deps"].append(_dep(name, str(spec), group, "php", source))
    return out


def parse_manifest(rel: str, text: str) -> Dict[str, Any]:
    name = rel.rsplit("/", 1)[-1].lower()
    if name == "pyproject.toml":
        return parse_pyproject(text, rel)
    if name.startswith("requirements") and name.endswith(".txt"):
        return parse_requirements(text, rel)
    if name == "package.json":
        return parse_package_json(text, rel)
    if name == "go.mod":
        return parse_go_mod(text, rel)
    if name == "cargo.toml":
        return parse_cargo(text, rel)
    if name == "pom.xml":
        return parse_pom(text, rel)
    if name == "gemfile":
        return parse_gemfile(text, rel)
    if name == "composer.json":
        return parse_composer(text, rel)
    return _empty()


# ------------------------------------------------------------ technology
TECH: Dict[str, Tuple[str, str]] = {
    # Python
    "numpy": ("NumPy", "numerical computing"), "pandas": ("pandas", "data analysis"),
    "scipy": ("SciPy", "scientific computing"), "scikit-learn": ("scikit-learn", "machine learning"),
    "sklearn": ("scikit-learn", "machine learning"), "torch": ("PyTorch", "deep learning"),
    "pytorch": ("PyTorch", "deep learning"), "tensorflow": ("TensorFlow", "deep learning"),
    "keras": ("Keras", "deep learning"), "transformers": ("Hugging Face Transformers", "AI / LLM"),
    "openai": ("OpenAI SDK", "AI / LLM"), "anthropic": ("Anthropic SDK", "AI / LLM"),
    "langchain": ("LangChain", "AI / LLM framework"), "langgraph": ("LangGraph", "AI / LLM framework"),
    "llama-index": ("LlamaIndex", "AI / LLM framework"), "crewai": ("CrewAI", "AI / LLM framework"),
    "autogen": ("AutoGen", "AI / LLM framework"), "chromadb": ("ChromaDB", "vector database"),
    "faiss-cpu": ("FAISS", "vector search"), "faiss-gpu": ("FAISS", "vector search"),
    "sentence-transformers": ("Sentence Transformers", "embeddings"), "ollama": ("Ollama client", "AI / LLM"),
    "fastapi": ("FastAPI", "web framework"), "flask": ("Flask", "web framework"),
    "django": ("Django", "web framework"), "streamlit": ("Streamlit", "web UI"),
    "gradio": ("Gradio", "web UI"), "uvicorn": ("Uvicorn", "web server"), "gunicorn": ("Gunicorn", "web server"),
    "requests": ("requests", "HTTP client"), "httpx": ("HTTPX", "HTTP client"), "aiohttp": ("aiohttp", "HTTP client / server"),
    "pydantic": ("Pydantic", "data validation"), "sqlalchemy": ("SQLAlchemy", "database ORM"),
    "alembic": ("Alembic", "database migrations"), "psycopg2": ("psycopg2", "PostgreSQL driver"),
    "psycopg": ("psycopg", "PostgreSQL driver"), "pymongo": ("PyMongo", "MongoDB driver"),
    "redis": ("Redis client", "cache / database"), "celery": ("Celery", "task queue"),
    "pytest": ("pytest", "testing"), "black": ("Black", "code formatting"), "ruff": ("Ruff", "linting"),
    "flake8": ("Flake8", "linting"), "mypy": ("mypy", "type checking"), "pyyaml": ("PyYAML", "YAML parsing"),
    "yaml": ("PyYAML", "YAML parsing"), "click": ("Click", "command-line interface"),
    "typer": ("Typer", "command-line interface"), "rich": ("Rich", "terminal output"),
    "pathspec": ("pathspec", ".gitignore-style matching"), "tomli": ("tomli", "TOML parsing"),
    "beautifulsoup4": ("Beautiful Soup", "HTML parsing"), "bs4": ("Beautiful Soup", "HTML parsing"),
    "selenium": ("Selenium", "browser automation"), "playwright": ("Playwright", "browser automation"),
    "matplotlib": ("Matplotlib", "plotting"), "seaborn": ("Seaborn", "plotting"), "plotly": ("Plotly", "plotting"),
    "opencv-python": ("OpenCV", "computer vision"), "cv2": ("OpenCV", "computer vision"),
    "pillow": ("Pillow", "image processing"), "boto3": ("AWS SDK (boto3)", "cloud"),
    "setuptools": ("setuptools", "packaging"), "wheel": ("wheel", "packaging"), "poetry-core": ("Poetry", "packaging"),
    "hatchling": ("Hatch", "packaging"), "flit-core": ("Flit", "packaging"),
    # JavaScript / TypeScript
    "react": ("React", "UI library"), "react-dom": ("React DOM", "UI library"), "next": ("Next.js", "web framework"),
    "vue": ("Vue", "UI framework"), "nuxt": ("Nuxt", "web framework"), "svelte": ("Svelte", "UI framework"),
    "express": ("Express", "web framework"), "koa": ("Koa", "web framework"), "fastify": ("Fastify", "web framework"),
    "typescript": ("TypeScript", "language / tooling"), "vite": ("Vite", "build tool"),
    "webpack": ("webpack", "build tool"), "jest": ("Jest", "testing"), "vitest": ("Vitest", "testing"),
    "mocha": ("Mocha", "testing"), "prettier": ("Prettier", "code formatting"), "tailwindcss": ("Tailwind CSS", "styling"),
    "axios": ("Axios", "HTTP client"), "redux": ("Redux", "state management"), "mongoose": ("Mongoose", "MongoDB ODM"),
    "sequelize": ("Sequelize", "database ORM"), "socket.io": ("Socket.IO", "real-time communication"),
    "graphql": ("GraphQL", "API"), "electron": ("Electron", "desktop apps"), "three": ("three.js", "3D graphics"),
    "d3": ("D3", "data visualisation"), "lodash": ("Lodash", "utilities"), "dotenv": ("dotenv", "configuration"),
    "jsonwebtoken": ("jsonwebtoken", "authentication"), "zod": ("Zod", "data validation"),
    # Go / Rust / Java / others
    "gin": ("Gin", "web framework"), "mux": ("gorilla/mux", "HTTP router"), "cobra": ("Cobra", "command-line interface"),
    "gorm": ("GORM", "database ORM"), "tokio": ("Tokio", "async runtime"), "serde": ("Serde", "serialisation"),
    "clap": ("clap", "command-line interface"), "actix-web": ("Actix Web", "web framework"),
    "axum": ("Axum", "web framework"), "rocket": ("Rocket", "web framework"), "reqwest": ("reqwest", "HTTP client"),
    "junit": ("JUnit", "testing"), "hibernate-core": ("Hibernate", "database ORM"), "lombok": ("Lombok", "code generation"),
    "rails": ("Ruby on Rails", "web framework"), "rspec": ("RSpec", "testing"), "laravel/framework": ("Laravel", "web framework"),
}
PREFIX_TECH = [
    ("@angular/", "Angular", "web framework"), ("@nestjs/", "NestJS", "web framework"),
    ("spring-boot", "Spring Boot", "web framework"), ("langchain", "LangChain", "AI / LLM framework"),
    ("@types/", "TypeScript type definitions", "tooling"), ("@babel/", "Babel", "build tool"),
    ("eslint", "ESLint", "linting"), ("@testing-library/", "Testing Library", "testing"),
    ("@aws-sdk/", "AWS SDK", "cloud"), ("@prisma/", "Prisma", "database ORM"),
    ("django-", "Django ecosystem", "web framework"), ("flask-", "Flask ecosystem", "web framework"),
    ("pytest-", "pytest plugins", "testing"), ("opentelemetry", "OpenTelemetry", "observability"),
]


def classify_dependency(name: str, ecosystem: str) -> Optional[Tuple[str, str]]:
    key = name.lower().replace("_", "-")
    if ecosystem == "go":
        key = key.rsplit("/", 1)[-1]
    elif ecosystem == "maven":
        key = key.split(":")[-1]
    if key in TECH:
        return TECH[key]
    for prefix, label, category in PREFIX_TECH:
        if key.startswith(prefix):
            return label, category
    return None


def detect_markers(files: List[str]) -> List[Dict[str, str]]:
    found: Dict[str, Dict[str, str]] = {}

    def add(label: str, category: str, evidence: str) -> None:
        found.setdefault(label, {"label": label, "category": category, "evidence": evidence})

    for rel in sorted(files):
        low = rel.lower()
        name = low.rsplit("/", 1)[-1]
        if name == "dockerfile" or name.startswith("dockerfile."):
            add("Docker", "containerisation", rel)
        elif name in ("docker-compose.yml", "docker-compose.yaml", "compose.yml", "compose.yaml"):
            add("Docker Compose", "containerisation", rel)
        elif low.startswith(".github/workflows/"):
            add("GitHub Actions", "CI/CD", rel)
        elif name == ".gitlab-ci.yml":
            add("GitLab CI", "CI/CD", rel)
        elif name == "jenkinsfile":
            add("Jenkins", "CI/CD", rel)
        elif name == "makefile":
            add("Make", "build tool", rel)
        elif name == "tsconfig.json":
            add("TypeScript", "language / tooling", rel)
        elif name.startswith("vite.config."):
            add("Vite", "build tool", rel)
        elif name.startswith("webpack.config."):
            add("webpack", "build tool", rel)
        elif name.startswith("next.config."):
            add("Next.js", "web framework", rel)
        elif name.startswith("tailwind.config."):
            add("Tailwind CSS", "styling", rel)
        elif name.startswith(".eslintrc"):
            add("ESLint", "linting", rel)
        elif name.startswith(".prettierrc"):
            add("Prettier", "code formatting", rel)
        elif name in ("pytest.ini", "tox.ini"):
            add("pytest" if name == "pytest.ini" else "tox", "testing", rel)
        elif name == "manage.py":
            add("Django", "web framework", rel)
        elif name == "alembic.ini":
            add("Alembic", "database migrations", rel)
        elif name == "procfile":
            add("Heroku (Procfile)", "deployment", rel)
        elif name == "vercel.json":
            add("Vercel", "deployment", rel)
        elif name == "netlify.toml":
            add("Netlify", "deployment", rel)
        elif name == ".pre-commit-config.yaml":
            add("pre-commit", "code quality", rel)
        elif name.endswith(".tf"):
            add("Terraform", "infrastructure as code", rel)
    return list(found.values())
