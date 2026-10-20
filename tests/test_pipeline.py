import copy
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fakes import FakeProvider
from repoexplain.config import DEFAULTS
from repoexplain.indexer import index_repo
from repoexplain.rag import NOT_FOUND, answer_stream, retrieve
from repoexplain.scanner import looks_secret, scan_repo
from repoexplain.store import INDEX_DIRNAME, VectorStore


def make_repo(root: Path):
    (root / "node_modules").mkdir()
    (root / "node_modules" / "junk.js").write_text("module.exports = 1;\n")
    (root / "src").mkdir()
    (root / "src" / "auth.py").write_text(
        "def login(user, password):\n    # check the password hash for this user\n    return password == 'x'\n")
    (root / "src" / "shapes.py").write_text(
        "def circle_area(radius):\n    return 3.14 * radius * radius\n")
    (root / ".env").write_text("API_KEY=supersecret\n")
    (root / ".env.example").write_text("API_KEY=\n")
    (root / "logo.png").write_bytes(b"\x89PNG\x00\x00binary")
    (root / "ignored.txt").write_text("should be ignored by gitignore\n")
    (root / ".gitignore").write_text("ignored.txt\n")


def cfg():
    c = copy.deepcopy(DEFAULTS)
    c["embeddings"]["query_prefix"] = ""
    c["indexing"]["chunk_chars"] = 400
    return c


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        make_repo(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_scanner_skips_unsafe_and_noise(self):
        result = scan_repo(self.root)
        self.assertIn("src/auth.py", result.files)
        self.assertIn(".env.example", result.files)
        for bad in (".env", "logo.png", "ignored.txt", "node_modules/junk.js"):
            self.assertNotIn(bad, result.files)
        self.assertTrue(looks_secret("server.PEM"))
        self.assertFalse(looks_secret(".env.example"))

    def test_index_and_retrieve(self):
        fake, c = FakeProvider(), cfg()
        stats = index_repo(self.root, c, fake, log=lambda m: None)
        self.assertGreaterEqual(stats["total_chunks"], 2)
        store = VectorStore(self.root / INDEX_DIRNAME)
        self.assertTrue(store.load())
        self.assertTrue((self.root / INDEX_DIRNAME / ".gitignore").is_file())
        hits = retrieve("where is the user password checked in login", store, fake, c)
        self.assertEqual(hits[0][1]["path"], "src/auth.py")
        self.assertIn((c["embeddings"]["model"], "embed"), fake.unloaded)
        self.assertEqual("".join(answer_stream("q", hits, fake, c)), "answer text")

    def test_no_hits_returns_not_found_without_llm(self):
        fake, c = FakeProvider(), cfg()
        self.assertEqual("".join(answer_stream("q", [], fake, c)), NOT_FOUND)

    def test_incremental_reindex(self):
        fake, c = FakeProvider(), cfg()
        index_repo(self.root, c, fake, log=lambda m: None)

        fake.texts_embedded = 0
        stats = index_repo(self.root, c, fake, log=lambda m: None)
        self.assertEqual(fake.texts_embedded, 0)
        self.assertEqual(stats["changed"], 0)

        (self.root / "src" / "shapes.py").write_text("def square_area(side):\n    return side * side\n")
        (self.root / "src" / "auth.py").unlink()
        stats = index_repo(self.root, c, fake, log=lambda m: None)
        self.assertEqual(stats["changed"], 1)
        self.assertEqual(stats["removed"], 1)
        store = VectorStore(self.root / INDEX_DIRNAME)
        store.load()
        paths = {ch["path"] for ch in store.chunks}
        self.assertNotIn("src/auth.py", paths)
        self.assertTrue(any("square_area" in ch["text"] for ch in store.chunks))
        self.assertFalse(any("circle_area" in ch["text"] for ch in store.chunks))

    def test_changed_embedding_model_forces_rebuild(self):
        fake, c = FakeProvider(), cfg()
        index_repo(self.root, c, fake, log=lambda m: None)
        c2 = cfg()
        c2["embeddings"]["model"] = "other-model"
        fake.texts_embedded = 0
        index_repo(self.root, c2, fake, log=lambda m: None)
        self.assertGreater(fake.texts_embedded, 0)


if __name__ == "__main__":
    unittest.main()
