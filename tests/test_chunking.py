import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from repoexplain.chunking import chunk_file

PY = '''import os


def alpha(x):
    """first"""
    return x + 1


def beta(y):
    return y * 2


class Big:
    def one(self):
        return "one" * 3

    def two(self):
        return "two" * 3
'''


class ChunkingTests(unittest.TestCase):
    def test_python_functions_stay_whole(self):
        chunks = chunk_file("m.py", PY, max_chars=80, overlap_lines=1)
        joined = "\n".join(c.text for c in chunks)
        for fragment in ("def alpha", "def beta", "def one", "def two"):
            self.assertIn(fragment, joined)
        for c in chunks:
            self.assertLessEqual(len(c.text), 80 * 2)
            self.assertLessEqual(c.start, c.end)

    def test_line_numbers_match_text(self):
        lines = PY.split("\n")
        for c in chunk_file("m.py", PY, max_chars=80, overlap_lines=1):
            self.assertEqual("\n".join(lines[c.start - 1:c.end]), c.text)

    def test_every_line_is_covered(self):
        covered = set()
        for c in chunk_file("m.py", PY, max_chars=80, overlap_lines=1):
            covered.update(range(c.start, c.end + 1))
        non_blank = {i for i, l in enumerate(PY.split("\n"), 1) if l.strip()}
        self.assertTrue(non_blank <= covered)

    def test_generic_file_and_long_block_split(self):
        text = "\n".join(f"line number {i} with some padding text" for i in range(100))
        chunks = chunk_file("notes.txt", text, max_chars=300, overlap_lines=2)
        self.assertGreater(len(chunks), 5)
        self.assertTrue(all(len(c.text) <= 600 for c in chunks))

    def test_syntax_error_falls_back(self):
        chunks = chunk_file("broken.py", "def (:\n    pass\n", max_chars=100, overlap_lines=1)
        self.assertEqual(len(chunks), 1)

    def test_empty_file(self):
        self.assertEqual(chunk_file("e.py", "", 100, 1), [])


if __name__ == "__main__":
    unittest.main()
