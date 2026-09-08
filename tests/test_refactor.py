"""B1 RED — the ollama app must consume llm-chat-core, not a local copy.

After the refactor:
  * memory + the tool surface are imported from ``llm_chat_core`` (single
    source of truth shared with the vLLM app), and
  * an ``make_ollama_embed(client)`` factory adapts Ollama's ``client.embed``
    to the engine-agnostic ``embed_fn`` seam expected by the core MemoryStore.
"""
import agent


class _FakeClient:
    """Minimal stand-in for ollama.Client."""

    def __init__(self):
        self.calls = []

    def embed(self, model, input):
        self.calls.append((model, input))
        return {"embeddings": [[0.1, 0.2, 0.3]]}


def test_make_ollama_embed_adapts_client():
    fake = _FakeClient()
    embed = agent.make_ollama_embed(fake)
    vec = embed("already-prefixed text")
    # Adapter returns the raw vector (core applies prefixes + normalization).
    assert vec == [0.1, 0.2, 0.3]
    # It must pass the text as a single-element batch to the embed model.
    assert fake.calls[0][1] == ["already-prefixed text"]


def test_memorystore_comes_from_core():
    assert agent.MemoryStore.__module__ == "llm_chat_core.memory"


def test_web_search_comes_from_core():
    assert agent.web_search.__module__ == "llm_chat_core.tools"


def test_tool_schemas_come_from_core():
    assert agent.WEB_SEARCH_TOOL is __import__(
        "llm_chat_core.tools", fromlist=["WEB_SEARCH_TOOL"]
    ).WEB_SEARCH_TOOL
    assert agent.RECALL_MEMORY_TOOL["function"]["name"] == "recall_memory"
