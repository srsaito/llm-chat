"""Persistent cross-session memory for the llm-chat agent.

A deliberately tiny, framework-free RAG store:
  * Each past Q&A exchange is embedded (via Ollama's embeddinggemma) and appended
    as one JSON line to ~/.llm-chat/memory.jsonl.
  * At startup every line is loaded into RAM. Retrieval is a brute-force cosine
    similarity (vectors are stored L2-normalized, so cosine == dot product).
  * At the expected scale (hundreds–thousands of notes) brute force is sub-50ms,
    so there is no need for numpy, a vector DB, or any extra dependency.

Everything here is plain stdlib plus the `ollama` Client that agent.py already
holds; the file is human-readable (`cat`/`grep` your own memories).
"""
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

from ollama import Client

# --- configuration (env-overridable) ----------------------------------------
MEMORY_PATH = Path(
    os.environ.get("LLM_MEMORY", "~/.llm-chat/memory.jsonl")
).expanduser()
EMBED_MODEL = os.environ.get("LLM_EMBED_MODEL", "embeddinggemma")

TOP_K = 3          # auto-inject: how many notes to surface per turn
# Tuned empirically against embeddinggemma: relevant exchanges score ~0.45-0.57,
# unrelated noise stays <=0.12, so 0.35 sits comfortably in the gap.
MIN_SIM = 0.35     # auto-inject: similarity floor
RECALL_MIN_SIM = 0.20  # recall_memory tool: looser floor for deliberate lookups
RECALL_K = 5           # recall_memory tool: more results than auto-inject
EMBED_DOC_CHARS = 1000   # how much of the answer feeds the embedding


# --- embedding helper -------------------------------------------------------
def _embed(client: Client, text: str, *, is_query: bool) -> list[float]:
    """Embed text with EmbeddingGemma's task prefixes, then L2-normalize.

    Raw Ollama does not add EmbeddingGemma's required prompt prefixes, so we do
    it here; without them retrieval quality measurably degrades.
    """
    if is_query:
        prompt = f"task: search result | query: {text}"
    else:
        prompt = f"title: none | text: {text}"
    resp = client.embed(model=EMBED_MODEL, input=[prompt])
    vec = list(resp["embeddings"][0])
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


# --- the store --------------------------------------------------------------
class MemoryStore:
    """Append-only JSONL of embedded Q&A exchanges with cosine retrieval."""

    def __init__(self, client: Client, path: Path = MEMORY_PATH) -> None:
        self.client = client
        self.path = path
        self.items: list[dict] = []
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        kept, skipped = 0, 0
        with self.path.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    skipped += 1
                    continue
                # Vectors from a different embedding model live in an
                # incompatible space; ignore them rather than mix.
                if rec.get("emb_model") != EMBED_MODEL:
                    skipped += 1
                    continue
                self.items.append(rec)
                kept += 1
        if skipped:
            print(
                f"  → [memory] loaded {kept} notes, skipped {skipped} "
                f"(corrupt or different embedding model)",
                file=sys.stderr,
            )

    def add(self, question: str, answer: str, session: str, model: str) -> None:
        """Embed and append one exchange. Append-only => crash-safe."""
        composite = f"{question}\n{answer[:EMBED_DOC_CHARS]}"
        vec = _embed(self.client, composite, is_query=False)
        rec = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "session": session,
            "q": question,
            "a": answer,
            "model": model,
            "emb_model": EMBED_MODEL,
            "v": vec,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        self.items.append(rec)

    def search(
        self,
        query: str,
        *,
        exclude_session: str | None = None,
        k: int = TOP_K,
        min_sim: float = MIN_SIM,
    ) -> list[dict]:
        """Return up to k records most similar to query, each with a 'sim' key."""
        if not self.items:
            return []
        qv = _embed(self.client, query, is_query=True)
        scored = []
        for rec in self.items:
            if exclude_session and rec.get("session") == exclude_session:
                continue
            sim = _dot(qv, rec["v"])
            if sim >= min_sim:
                scored.append((sim, rec))
        scored.sort(key=lambda t: t[0], reverse=True)
        out = []
        for sim, rec in scored[:k]:
            hit = dict(rec)
            hit["sim"] = sim
            out.append(hit)
        return out

    def wipe(self) -> None:
        """Delete the store on disk and in RAM."""
        self.items.clear()
        if self.path.exists():
            self.path.unlink()
