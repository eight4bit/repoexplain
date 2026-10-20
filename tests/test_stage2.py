import copy
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fakes import FakeProvider
from repoexplain.agent import Agent, Turn
from repoexplain.analyzer import analyze_file, is_stdlib
from repoexplain.config import DEFAULTS
from repoexplain.indexer import index_repo
from repoexplain.manifests import (classify_dependency, parse_cargo, parse_go_mod, parse_package_json,
                                   parse_pyproject, parse_requirements)
from repoexplain.overview import load_overview, readme_purpose, render_deps_report, render_overview
from repoexplain.router import classify, extract_targets
from repoexplain.store import INDEX_DIRNAME, VectorStore
from repoexplain.tools import RepoTools
from repoexplain.verify import grounding_warnings

PYPROJECT = '''
[build-system]
requires = ["setuptools>=61"]
[project]
name = "demoapp"
version = "1.2.3"
description = "A demo web service"
requires-python = ">=3.9"
dependencies = ["requests>=2.28", "fastapi", "numpy; python_version>'3.8'"]
[project.optional-dependencies]
dev = ["pytest>=7"]
[project.scripts]
demoapp = "demoapp.cli:main"
[tool.black]
line-length = 100
'''

CLI = '''"""Command line entry for the demo app."""
import argparse
import requests


def build_parser():
    return argparse.ArgumentParser()


def handle_login(user, password):
    """Check the password for this user."""
    return password == "x"


def main():
    parser = build_parser()
    parser.parse_args()


if __name__ == "__main__":
    main()
'''

README = '''# Demo App

[![badge](x)](y)

Demo App is a tiny web service that checks logins and reports their status.

## Install
'''


def make_repo(root: Path):
    (root / "pyproject.toml").write_text(PYPROJECT)
    (root / "README.md").write_text(README)
    (root / "Dockerfile").write_text("FROM python:3.11\n")
    pkg = root / "src" / "demoapp"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text('"""Demo package."""\n')
    (pkg / "cli.py").write_text(CLI)
    (pkg / "util.py").write_text("def helper(x):\n    return x * 2\n")
    (root / "tests").mkdir()
    (root / "tests" / "test_cli.py").write_text("def test_x():\n    assert True\n")
    (root / ".env").write_text("SECRET=1\n")


def cfg():
    c = copy.deepcopy(DEFAULTS)
    c["embeddings"]["query_prefix"] = ""
    return c


class ParserTests(unittest.TestCase):
    def test_pyproject(self):
        out = parse_pyproject(PYPROJECT, "pyproject.toml")
        names = {(d["name"], d["group"]) for d in out["deps"]}
        self.assertIn(("requests", "runtime"), names)
        self.assertIn(("numpy", "runtime"), names)
        self.assertIn(("pytest", "optional:dev"), names)
        self.assertIn(("setuptools", "build"), names)
        self.assertEqual(out["scripts"][0]["command"], "demoapp.cli:main")
        self.assertEqual(out["meta"]["name"], "demoapp")
        self.assertIn("black", out["meta"]["tools"])

    def test_requirements(self):
        out = parse_requirements("# c\nflask==2.0\n-r other.txt\nnumpy>=1.2 ; python_version>'3'\ngit+https://x\n", "requirements.txt")
        self.assertEqual([d["name"] for d in out["deps"]], ["flask", "numpy"])

    def test_package_json(self):
        text = '{"name":"app","main":"src/index.js","dependencies":{"react":"^18"},"devDependencies":{"jest":"1"},"scripts":{"start":"node src/index.js"},"bin":{"app":"bin/cli.js"}}'
        out = parse_package_json(text, "package.json")
        self.assertEqual({(d["name"], d["group"]) for d in out["deps"]}, {("react", "runtime"), ("jest", "dev")})
        self.assertEqual(out["scripts"][0]["name"], "start")
        self.assertEqual(len(out["entries"]), 2)

    def test_go_and_cargo(self):
        go = parse_go_mod("module example.com/x\n\ngo 1.21\n\nrequire (\n\tgithub.com/gin-gonic/gin v1.9.0\n\tgolang.org/x/y v0.1.0 // indirect\n)\n", "go.mod")
        self.assertEqual([d["name"] for d in go["deps"]], ["github.com/gin-gonic/gin"])
        cargo = parse_cargo('[package]\nname="c"\n[dependencies]\ntokio={version="1"}\nserde="1"\n', "Cargo.toml")
        self.assertEqual({d["name"] for d in cargo["deps"]}, {"tokio", "serde"})

    def test_classify(self):
        self.assertEqual(classify_dependency("FastAPI", "python")[0], "FastAPI")
        self.assertEqual(classify_dependency("github.com/gin-gonic/gin", "go")[0], "Gin")
        self.assertEqual(classify_dependency("@angular/core", "npm")[0], "Angular")
        self.assertIsNone(classify_dependency("totally-unknown-lib", "python"))


class AnalyzerTests(unittest.TestCase):
    def test_python_file(self):
        info = analyze_file("src/demoapp/cli.py", CLI)
        names = {s["name"] for s in info["symbols"]}
        self.assertTrue({"build_parser", "handle_login", "main"} <= names)
        self.assertEqual(info["doc"], "Command line entry for the demo app.")
        self.assertIn("requests", info["imports"])
        self.assertTrue(info["entry"])
        main = next(s for s in info["symbols"] if s["name"] == "main")
        self.assertIsNotNone(main["end"])

    def test_class_methods(self):
        info = analyze_file("a.py", "class A:\n    def go(self):\n        pass\n")
        self.assertIn("A.go", {s["name"] for s in info["symbols"]})

    def test_other_languages(self):
        js = analyze_file("server.js", "// Express server\nconst app = express();\nfunction start() {}\nconst run = async () => {};\napp.listen(3000);\n")
        self.assertEqual({s["name"] for s in js["symbols"]}, {"start", "run"})
        self.assertTrue(js["entry"])
        go = analyze_file("main.go", "package main\n\nfunc main() {}\nfunc (s *S) Do() {}\n")
        self.assertEqual({s["name"] for s in go["symbols"]}, {"main", "Do"})
        self.assertTrue(go["entry"])

    def test_stdlib(self):
        self.assertTrue(is_stdlib("os"))
        self.assertFalse(is_stdlib("requests_not_stdlib_xyz"))


class RouterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        make_repo(cls.root)
        index_repo(cls.root, cfg(), FakeProvider(), log=lambda m: None)
        cls.overview = load_overview(cls.root / INDEX_DIRNAME)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def route(self, q):
        return classify(q, self.overview).route

    def test_routes(self):
        cases = {
            "What dependencies does it use?": "deps",
            "Which technologies and frameworks are used?": "deps",
            "What language is this written in?": "deps",
            "What can this project do? List its features.": "overview",
            "Which function runs first when I type demoapp?": "overview",
            "What does this project do?": "overview",
            "Where is the password checked?": "locate",
            "Where is handle_login defined?": "locate",
            "Explain handle_login": "explain",
            "Explain cli.py": "explain",
            "How does the parser get built?": "search",
            "How does the tool decide which files to skip?": "search",
            "Which file handles the login check?": "locate",
        }
        for question, expected in cases.items():
            self.assertEqual(self.route(question), expected, question)

    def test_targets(self):
        paths, symbols = extract_targets("what does `handle_login` do in cli.py", self.overview)
        self.assertEqual(symbols, ["handle_login"])
        self.assertEqual(paths, ["src/demoapp/cli.py"])

    def test_plain_words_are_not_symbols(self):
        _, symbols = extract_targets("how does the parser work with helper words", self.overview)
        self.assertEqual(symbols, [])


class OverviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        make_repo(self.root)
        self.fake = FakeProvider()
        self.cfg = cfg()
        self.stats = index_repo(self.root, self.cfg, self.fake, log=lambda m: None)
        self.overview = load_overview(self.root / INDEX_DIRNAME)

    def tearDown(self):
        self.tmp.cleanup()

    def test_facts(self):
        p = self.overview["project"]
        self.assertEqual(p["name"], "demoapp")
        self.assertEqual(p["version"], "1.2.3")
        self.assertTrue(p["purpose"].startswith("Demo App is a tiny web service"))
        self.assertEqual(p["languages"][0]["language"], "Python")
        labels = {t["label"] for t in p["technologies"]}
        self.assertTrue({"FastAPI", "NumPy", "Docker", "black"} <= labels)
        entry = p["entry_points"][0]
        self.assertEqual(entry["file"], "src/demoapp/cli.py")
        self.assertIsNotNone(entry["line"])
        self.assertIn("`demoapp` command runs `main()`", entry["text"])
        third = {i["name"] for i in p["third_party_imports"]}
        self.assertIn("requests", third)
        self.assertNotIn("argparse", third)
        self.assertNotIn(".env", self.overview["files"])
        self.assertFalse(any("tests/" in e["text"] for e in p["entry_points"]))

    def test_renderers(self):
        text = render_overview(self.overview, 3200)
        self.assertIn("ENTRY POINTS", text)
        self.assertIn("src/demoapp/cli.py", text)
        self.assertLessEqual(len(render_overview(self.overview, 600)), 700)
        deps = render_deps_report(self.overview)
        self.assertIn("requests >=2.28 - requests: HTTP client", deps)
        self.assertIn("Docker", deps)

    def test_readme_purpose(self):
        self.assertEqual(readme_purpose("# T\n\n```\ncode\n```\n\nFirst *real* line\nsecond line\n\nNext"),
                         "First real line second line")

    def test_tools_are_safe(self):
        tools = RepoTools(self.root, self.overview)
        self.assertIsNone(tools.read_file(".env"))
        self.assertIsNone(tools.read_file("../outside.txt"))
        self.assertIsNone(tools.read_file("../../etc/passwd"))
        got = tools.read_file("src/demoapp/cli.py", 1, 3)
        self.assertEqual(got["end"], 3)
        self.assertIn("Command line entry", got["text"])
        found = tools.find_symbols(["handle_login"])
        self.assertEqual(found[0]["path"], "src/demoapp/cli.py")
        self.assertIn("src/", tools.list_tree(depth=1))

    def make_agent(self):
        store = VectorStore(self.root / INDEX_DIRNAME)
        store.load()
        return Agent(self.root, self.cfg, store, self.overview, self.fake, self.fake)

    def test_deps_route_needs_no_llm(self):
        agent = self.make_agent()
        before = self.fake.chat_calls
        result = agent.answer("What dependencies are used?")
        text = "".join(result.stream)
        self.assertEqual(result.route, "deps")
        self.assertTrue(result.direct)
        self.assertEqual(self.fake.chat_calls, before)
        self.assertIn("fastapi", text)

    def test_overview_route_contains_overview(self):
        agent = self.make_agent()
        result = agent.answer("Which function runs first when I type demoapp?")
        "".join(result.stream)
        self.assertEqual(result.route, "overview")
        self.assertIn("runs `main()`", result.evidence[0].text)
        self.assertIn("runs `main()`", self.fake.last_messages[-1]["content"])

    def test_explain_route_reads_the_symbol(self):
        agent = self.make_agent()
        result = agent.answer("Explain handle_login")
        "".join(result.stream)
        self.assertEqual(result.route, "explain")
        self.assertIn("password", result.evidence[0].text)
        self.assertTrue(any("read_file" in s for s in result.steps))

    def test_locate_route(self):
        agent = self.make_agent()
        result = agent.answer("Where is handle_login defined?")
        "".join(result.stream)
        self.assertEqual(result.route, "locate")
        self.assertIn("src/demoapp/cli.py:", result.evidence[0].text)

    def test_forced_route_and_bad_route(self):
        agent = self.make_agent()
        result = agent.answer("anything", forced_route="search")
        self.assertEqual(result.route, "search")
        with self.assertRaises(ValueError):
            agent.answer("anything", forced_route="nope")

    def test_chat_history_is_sent(self):
        agent = self.make_agent()
        history = [Turn("Where is handle_login defined?", "In cli.py")]
        "".join(agent.answer("and what does it return?", history).stream)
        roles = [m["role"] for m in self.fake.last_messages]
        self.assertEqual(roles, ["system", "user", "assistant", "user"])

    def test_not_found_without_llm(self):
        agent = self.make_agent()
        agent.store.reset({})
        before = self.fake.chat_calls
        text = "".join(agent.answer("something unrelated", forced_route="search").stream)
        self.assertEqual(text, "I couldn't find that in the indexed code.")
        self.assertEqual(self.fake.chat_calls, before)

    def test_summaries_and_cache(self):
        c = cfg()
        c["summarizer"]["enabled"] = True
        llm = FakeProvider()
        llm.reply = ["Handles ", "the command line."]
        stats = index_repo(self.root, c, self.fake, log=lambda m: None, summarize=True, summarizer=llm)
        self.assertGreaterEqual(stats["summaries"], 1)
        calls = llm.chat_calls
        ov = load_overview(self.root / INDEX_DIRNAME)
        self.assertEqual(ov["files"]["src/demoapp/cli.py"]["summary"], "Handles the command line.")
        index_repo(self.root, c, self.fake, log=lambda m: None, summarize=True, summarizer=llm)
        self.assertEqual(llm.chat_calls, calls)  # cached, no new model calls


class VerifyTests(unittest.TestCase):
    def test_flags_unknown_names_and_bad_citations(self):
        context = "def real_function(): pass"
        warnings = grounding_warnings("Use `real_function()` and `ghost_function` as in [1] and [5].",
                                      context, {"real_function"}, n_sources=2)
        self.assertEqual(len(warnings), 2)
        self.assertTrue(any("ghost_function" in w for w in warnings))
        self.assertTrue(any("[5]" in w for w in warnings))

    def test_clean_answer(self):
        self.assertEqual(grounding_warnings("See `a.b` and `--flag` and `x y`.", "a.b", set(), 1), [])


if __name__ == "__main__":
    unittest.main()
