"""A deterministic stand-in for a model server, so tests need no Ollama."""
import re
import zlib

DIM = 128


class FakeProvider:
    def __init__(self):
        self.texts_embedded = 0
        self.unloaded = []
        self.chat_calls = 0
        self.last_messages = None
        self.reply = ["answer ", "text"]

    def embed(self, model, texts):
        self.texts_embedded += len(texts)
        out = []
        for text in texts:
            vec = [0.0] * DIM
            for word in re.findall(r"[a-z]+", text.lower().replace("_", " ")):
                vec[zlib.crc32(word.encode()) % DIM] += 1.0
            out.append(vec)
        return out

    def unload(self, model, kind="chat"):
        self.unloaded.append((model, kind))

    def chat_stream(self, model, messages, options):
        self.chat_calls += 1
        self.last_messages = messages
        for piece in self.reply:
            yield piece
