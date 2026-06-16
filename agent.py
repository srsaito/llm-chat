#!/usr/bin/env python3
"""Minimal agentic web-search chat: any tool-capable Ollama model + DuckDuckGo.

This demonstrates native Ollama tool-calling (no MCP, no framework):
  1. We register a `web_search` function with a JSON schema.
  2. The model decides when to call it and emits a structured tool call.
  3. We execute the search and feed the results back.
  4. The model answers using the results.

On top of that it adds:
  * Conversational memory  — follow-up questions see the running thread.
  * Persistent memory (RAG) — past exchanges are embedded and recalled across
    sessions (see memory.py); surfaced both automatically and via a
    `recall_memory` tool.
  * Model switching        — type `change model` to pick any installed
    tool-capable model for the rest of the session.

Usage:
  ./.venv/bin/python agent.py "what's the latest news about <topic>?"
  ./.venv/bin/python agent.py            # interactive REPL
  ./.venv/bin/python agent.py --no-memory ...

REPL commands: change model (or /model), /clear, /forget, Ctrl-C to quit.
Env vars: LLM_MODEL, LLM_EMBED_MODEL, LLM_MEMORY, OLLAMA_HOST.
"""
import argparse
import json
import os
import sys
import uuid
from datetime import date

from ddgs import DDGS
from ollama import Client

import memory as memory_mod
from memory import MemoryStore

MODEL = os.environ.get("LLM_MODEL", "gemma4:31b")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
MAX_TOOL_ROUNDS = 6      # safety cap so a confused model can't loop forever
MAX_HISTORY_TURNS = 8    # user/assistant pairs kept in the live context
NUM_CTX = 16384          # context window; ample on this hardware


# --- the web_search tool ----------------------------------------------------
def web_search(query: str, max_results: int = 5) -> str:
    """Run a DuckDuckGo search and return formatted text results."""
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=max_results))
    if not results:
        return "No results found."
    blocks = []
    for i, r in enumerate(results, 1):
        blocks.append(
            f"{i}. {r.get('title', '')}\n"
            f"   URL: {r.get('href', '')}\n"
            f"   {r.get('body', '')}"
        )
    return "\n\n".join(blocks)


WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the web for current, real-time, or uncertain information. "
            "Use this for recent events, news, prices, or any fact you are not "
            "confident about from memory."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query."},
                "max_results": {
                    "type": "integer",
                    "description": "How many results to return (default 5).",
                },
            },
            "required": ["query"],
        },
    },
}

RECALL_MEMORY_TOOL = {
    "type": "function",
    "function": {
        "name": "recall_memory",
        "description": (
            "Search your long-term memory of previous conversations with this "
            "user. Relevant memories are usually provided to you automatically; "
            "only call this if you need to look up something that was NOT already "
            "shown to you (e.g. an older detail, or with different search terms)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to look for in past conversations.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "How many memories to return (default 5).",
                },
            },
            "required": ["query"],
        },
    },
}

DISPATCH = {"web_search": web_search}

SYSTEM_PROMPT = (
    "You are a helpful assistant with access to a web_search tool. "
    f"Today's date is {date.today():%A, %B %d, %Y}. Trust this date and the "
    "search results over your own training knowledge, which is older and may be "
    "out of date. When a question needs current or uncertain information, call "
    "web_search (usually ONE search is enough). Do not repeat the same or "
    "similar searches, and do not search for today's date — it is given above. "
    "Once you have results that address the question, STOP searching and answer, "
    "citing the source URLs you used. If the results are imperfect, answer with "
    "what you have rather than searching again. "
    "You may also be shown notes from previous conversations with this user; "
    "they may be stale, so prefer fresh web_search results for anything "
    "time-sensitive. Relevant notes are provided automatically — only call "
    "recall_memory if you need something that isn't already shown."
)


def _format_memories(hits: list[dict], answer_chars: int) -> str:
    lines = []
    for h in hits:
        day = h.get("ts", "")[:10]
        lines.append(f"- ({day}) Q: {h['q']} → A: {h['a'][:answer_chars]}")
    return "\n".join(lines)


def answer(
    client: Client,
    question: str,
    messages: list,
    model: str,
    *,
    memory: MemoryStore | None = None,
    session_id: str = "",
    tools: list | None = None,
) -> str | None:
    """Run one question through the tool loop on the shared `messages` list.

    Returns the final answer text, or None if the model failed to converge.
    On success the turn is pruned down to [question, answer] so the live
    context stays small; the bulky search dumps are discarded.
    """
    if tools is None:
        tools = [WEB_SEARCH_TOOL]
    turn_start = len(messages)

    # Hybrid retrieval, part 1: auto-inject relevant past notes (other sessions).
    if memory is not None:
        hits = memory.search(question, exclude_session=session_id)
        if hits:
            sims = ", ".join(f"{h['sim']:.2f}" for h in hits)
            print(
                f"  → [memory] recalled {len(hits)} note(s) (sim {sims})",
                file=sys.stderr,
            )
            block = (
                "Notes from previous conversations with this user "
                "(may be outdated):\n" + _format_memories(hits, 300)
            )
            messages.append({"role": "system", "content": block})

    user_msg = {"role": "user", "content": question}
    messages.append(user_msg)

    for _ in range(MAX_TOOL_ROUNDS):
        resp = client.chat(
            model=model, messages=messages, tools=tools,
            options={"num_ctx": NUM_CTX},
        )
        msg = resp["message"]
        messages.append(msg)

        tool_calls = msg.get("tool_calls")
        if not tool_calls:
            content = msg.get("content") or "(no content)"
            print("\n" + content)
            # Prune this turn to just the Q&A; drop notes block + tool dumps.
            messages[turn_start:] = [user_msg, msg]
            # Cap history (keep system at index 0).
            if len(messages) > 1 + 2 * MAX_HISTORY_TURNS:
                messages[1:] = messages[1:][-2 * MAX_HISTORY_TURNS:]
            return content

        for call in tool_calls:
            name = call["function"]["name"]
            args = call["function"]["arguments"]
            if isinstance(args, str):
                args = json.loads(args)
            print(f"  → [tool] {name}({args})", file=sys.stderr)
            try:
                if name == "recall_memory":
                    if memory is None:
                        result = "Memory is disabled."
                    else:
                        recalled = memory.search(
                            args["query"],
                            k=args.get("max_results", memory_mod.RECALL_K),
                            min_sim=memory_mod.RECALL_MIN_SIM,
                        )
                        print(
                            f"  → [memory] recall_memory matched "
                            f"{len(recalled)} note(s)",
                            file=sys.stderr,
                        )
                        result = (
                            _format_memories(recalled, 500)
                            if recalled else "No relevant memories found."
                        )
                else:
                    result = DISPATCH[name](**args)
            except Exception as e:  # surface tool errors back to the model
                result = f"Tool error: {e}"
            messages.append(
                {"role": "tool", "tool_name": name, "content": result}
            )

    print("\n(stopped: exceeded tool-call rounds)")
    messages[turn_start:] = []  # don't poison history with a failed turn
    return None


# --- model switching --------------------------------------------------------
def _models_by_tool_support(client: Client, cache: dict) -> tuple[list, list]:
    """Return (tool_capable, excluded) installed model names."""
    capable, excluded = [], []
    for m in client.list()["models"]:
        name = m["model"]
        if name not in cache:
            try:
                cache[name] = client.show(name).get("capabilities", []) or []
            except Exception:
                cache[name] = []
        (capable if "tools" in cache[name] else excluded).append(name)
    return sorted(capable), sorted(excluded)


def choose_model(client: Client, current: str, cache: dict) -> str:
    """Show a numbered menu of tool-capable models; return the chosen one."""
    capable, excluded = _models_by_tool_support(client, cache)
    if not capable:
        print("  (no tool-capable models installed)", file=sys.stderr)
        return current
    print("\nTool-capable models:")
    for i, name in enumerate(capable, 1):
        marker = "  * (current)" if name == current else ""
        print(f"  {i}. {name}{marker}")
    if excluded:
        print(f"  (excluded — no tool support: {', '.join(excluded)})")
    try:
        choice = input("pick a model #> ").strip()
    except (KeyboardInterrupt, EOFError):
        print()
        return current
    if choice.isdigit() and 1 <= int(choice) <= len(capable):
        new = capable[int(choice) - 1]
        if new != current:
            print(f"  → [model] switched to {new}", file=sys.stderr)
        return new
    print("  (no change)", file=sys.stderr)
    return current


# --- entrypoint -------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="*", help="one-shot question")
    parser.add_argument(
        "--no-memory", action="store_true",
        help="disable persistent cross-session memory",
    )
    args = parser.parse_args()

    client = Client(host=OLLAMA_HOST)
    session_id = uuid.uuid4().hex
    model = MODEL

    mem: MemoryStore | None = None
    if not args.no_memory:
        try:
            mem = MemoryStore(client)
            memory_mod._embed(client, "probe", is_query=True)  # fail fast
        except Exception as e:
            print(
                f"  → [memory] disabled ({e}); "
                f"run: ollama pull {memory_mod.EMBED_MODEL}",
                file=sys.stderr,
            )
            mem = None

    tools = [WEB_SEARCH_TOOL] + ([RECALL_MEMORY_TOOL] if mem else [])
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # One-shot mode.
    if args.question:
        q = " ".join(args.question)
        result = answer(
            client, q, messages, model,
            memory=mem, session_id=session_id, tools=tools,
        )
        if mem is not None and result:
            mem.add(q, result, session_id, model)
        return

    # Interactive REPL.
    mem_state = "on" if mem else "off"
    print(f"llm-chat  (model={model}, memory={mem_state}, host={OLLAMA_HOST})")
    print("Commands: 'change model', /clear, /forget — or Ctrl-C to quit.\n")
    caps_cache: dict = {}
    try:
        while True:
            q = input("ask> ").strip()
            if not q:
                continue
            low = q.lower()
            if low in ("change model", "/model"):
                model = choose_model(client, model, caps_cache)
                print()
                continue
            if low == "/clear":
                del messages[1:]
                print("  (conversation cleared)\n")
                continue
            if low == "/forget":
                if mem is None:
                    print("  (memory is disabled)\n")
                    continue
                confirm = input("  erase ALL persistent memory? [y/N] ").strip()
                if confirm.lower() == "y":
                    mem.wipe()
                    print("  (memory erased)\n")
                else:
                    print("  (kept)\n")
                continue
            result = answer(
                client, q, messages, model,
                memory=mem, session_id=session_id, tools=tools,
            )
            if mem is not None and result:
                mem.add(q, result, session_id, model)
            print()
    except (KeyboardInterrupt, EOFError):
        print("\nbye")


if __name__ == "__main__":
    main()
