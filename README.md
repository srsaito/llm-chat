# llm-chat

A minimal, framework-free agentic chat over **Ollama**: any tool-capable local
model + live **web search**, with **conversational memory** and **persistent
cross-session memory (RAG)** — in ~2 small files, no LangChain/LlamaIndex, no
MCP, no Docker.

## What it does

- **Native Ollama tool-calling.** The model is given a `web_search` tool (and,
  when memory is on, a `recall_memory` tool). It decides when to call them; we
  execute and feed results back. You can watch every decision on stderr
  (`→ [tool] …`, `→ [memory] …`, `→ [model] …`).
- **Web search** via DuckDuckGo (`ddgs`) — no API key.
- **Conversational memory** — follow-up questions in the REPL see the running
  thread ("where do I live?" after you mentioned it).
- **Persistent memory (RAG)** — each Q&A is embedded and saved; relevant past
  exchanges are recalled in future sessions, both automatically and on demand.
- **Multiple models** — switch the chat model at runtime; only tool-capable
  models are offered.

## Prerequisites

- A running Ollama server (this project assumes it's reachable at
  `http://127.0.0.1:11434`; override with `OLLAMA_HOST`).
- A tool-capable chat model, e.g. `ollama pull gemma4:31b` (default) or
  `ollama pull qwen3.5:9b`.
- An embedding model for memory: **`ollama pull embeddinggemma`**.
  (Without it, the app prints a notice and runs with memory disabled.)
- [`uv`](https://docs.astral.sh/uv/) for dependency management. `uv run` auto-syncs the environment from `pyproject.toml` / `uv.lock`, so there's no venv to manage by hand. (First run does the sync automatically; `uv sync` does it explicitly.)

## Usage

```bash
# one-shot
uv run agent.py "what's the latest news about <topic>?"

# interactive REPL
uv run agent.py

# disable persistent memory for this run
uv run agent.py --no-memory
```

### REPL commands
| Command | Effect |
|---|---|
| `change model` (or `/model`) | Numbered menu of installed **tool-capable** models; pick one for the rest of the session. Conversation history is kept across the switch. |
| `/clear` | Reset the current conversation (keeps persistent memory). |
| `/forget` | Erase **all** persistent memory (asks to confirm). |
| `Ctrl-C` / `Ctrl-D` | Quit. |

## How memory works

**In-session:** the REPL owns one `messages` list. After each answer the turn is
pruned to just `[question, answer]` — the bulky tool/search dumps are dropped —
and history is capped at `MAX_HISTORY_TURNS` (8) pairs, so context stays small
even in long sessions.

**Cross-session (RAG):** `memory.py` keeps an append-only JSONL of embedded Q&A
exchanges and retrieves by brute-force cosine similarity (vectors stored
L2-normalized; pure Python, no numpy/vector-DB — fast enough at this scale and
`cat`/`grep`-able).

- **Hybrid retrieval:** before each question, the top-k (3) most similar past
  notes above a similarity floor are **auto-injected** as context (excluding the
  current session). The model can also call the **`recall_memory`** tool for a
  deliberate, lower-threshold lookup of anything not already shown.
- **Embeddings:** `embeddinggemma` (768-dim) via Ollama, with EmbeddingGemma's
  required query/document task prefixes applied in `memory.py`. The embedding
  model is independent of the chat model, so switching chat models never
  invalidates the store.
- **Tuning:** the auto-inject floor `MIN_SIM` is `0.35` (relevant exchanges
  score ~0.45–0.57, noise stays ≤0.12). Watch the `→ [memory] recalled …` sim
  scores on stderr to retune.

## Configuration (env vars)

| Var | Default | Purpose |
|---|---|---|
| `LLM_MODEL` | `gemma4:31b` | Starting chat model |
| `LLM_EMBED_MODEL` | `embeddinggemma` | Embedding model for memory |
| `LLM_MEMORY` | `~/.llm-chat/memory.jsonl` | Persistent memory file |
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | Ollama server |

## Files

- `agent.py` — the tool loop, model switcher, REPL, CLI.
- `memory.py` — embeddings + JSONL store + cosine retrieval.
- `~/.llm-chat/memory.jsonl` — your persistent memories (created on first use).

## Notes

- Large dense models are memory-bandwidth-bound (e.g. `gemma4:31b` ≈ 11 tok/s on
  a GB10), so answers take ~20–40s depending on how much searching happens;
  `qwen3.5:9b` is much faster if you want snappier responses (`change model`).
- The model is told today's date in its system prompt — without that grounding,
  models past their training cutoff over-search and fail to converge.
