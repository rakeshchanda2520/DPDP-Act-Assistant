"""
Local LLM access via Ollama. Shared by index.py and ask.py.

Stdlib only — Ollama speaks plain HTTP+JSON, so a dependency would buy nothing.

Everything is overridable from the environment, so switching to a bigger model
is a shell variable rather than an edit:

    OLLAMA_HOST   default http://127.0.0.1:11434
    DPDP_MODEL    default qwen2.5:3b-instruct
    DPDP_NUM_CTX  default 8192
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
MODEL = os.environ.get("DPDP_MODEL", "qwen2.5:3b-instruct")
NUM_CTX = int(os.environ.get("DPDP_NUM_CTX", "8192"))

# Models known to be too small to answer legal questions reliably. Not a
# blocker — a warning, because they are what most laptops actually have.
SMALL = ("1b", "1.5b", "2b", "3b", "4b")


def _post(path: str, payload: dict, timeout: int) -> dict:
    request = urllib.request.Request(
        HOST + path, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def models() -> list[str]:
    try:
        with urllib.request.urlopen(HOST + "/api/tags", timeout=10) as response:
            return [m["name"] for m in json.loads(response.read()).get("models", [])]
    except OSError:
        return []


def check(model: str = MODEL) -> str | None:
    """Return an error string if the model can't be used, else None."""
    available = models()
    if not available:
        return (f"no Ollama server at {HOST}. Start it with `ollama serve`, "
                f"or set OLLAMA_HOST.")
    if model not in available and f"{model}:latest" not in available:
        return (f"model '{model}' not installed. Have: {', '.join(available)}\n"
                f"  pull it with:  ollama pull {model}")
    return None


def chat(prompt: str, system: str | None = None, schema: dict | None = None,
         model: str = MODEL, num_ctx: int = NUM_CTX, temperature: float = 0.2,
         timeout: int = 600) -> str | dict:
    """One turn. With `schema`, Ollama constrains the output to it and the
    parsed object comes back; without it, you get the text."""
    messages = ([{"role": "system", "content": system}] if system else []) + \
               [{"role": "user", "content": prompt}]
    payload = {
        "model": model, "messages": messages, "stream": False,
        "options": {"num_ctx": num_ctx, "temperature": temperature},
    }
    if schema:
        payload["format"] = schema

    try:
        content = _post("/api/chat", payload, timeout)["message"]["content"]
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"ollama {e.code}: {e.read().decode('utf-8', 'replace')[:300]}")
    except OSError as e:
        raise RuntimeError(f"cannot reach ollama at {HOST}: {e}")

    if not schema:
        return content
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"model returned invalid JSON despite a schema: {e}\n{content[:300]}")


def chat_stream(prompt: str, system: str | None = None, model: str = MODEL,
                num_ctx: int = NUM_CTX, temperature: float = 0.2,
                timeout: int = 600):
    """Yield answer fragments as the model produces them.

    A local 3B model takes 30-60s to finish a statutory answer. Waiting for the
    whole thing feels broken; streaming the first token in a second or two does
    not. Ollama emits one JSON object per line, so this is a plain line loop —
    no SSE parsing, no dependency.
    """
    messages = ([{"role": "system", "content": system}] if system else []) + \
               [{"role": "user", "content": prompt}]
    payload = {"model": model, "messages": messages, "stream": True,
               "options": {"num_ctx": num_ctx, "temperature": temperature}}
    request = urllib.request.Request(
        HOST + "/api/chat", data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"})

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for raw in response:
                if not raw.strip():
                    continue
                chunk = json.loads(raw)
                if fragment := chunk.get("message", {}).get("content"):
                    yield fragment
                if chunk.get("done"):
                    return
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"ollama {e.code}: {e.read().decode('utf-8', 'replace')[:300]}")
    except OSError as e:
        raise RuntimeError(f"cannot reach ollama at {HOST}: {e}")


def warn_if_small(model: str = MODEL) -> None:
    if any(tag in model.lower() for tag in SMALL):
        print(f"note: {model} is a small model. It is fine for indexing, but for\n"
              f"      answering legal questions a 7B+ model is materially better:\n"
              f"        ollama pull qwen2.5:7b-instruct\n"
              f"        set DPDP_MODEL=qwen2.5:7b-instruct\n", file=sys.stderr)
