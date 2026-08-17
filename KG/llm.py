"""
Pluggable LLM access. Shared by index.py, ask.py, api.py.

Two providers, same five-function surface (`chat`, `chat_stream`, `check`,
`models`, `warn_if_small`) so callers never know which one is behind them:

    OllamaProvider   local, via Ollama's plain HTTP+JSON API. No key needed.
                      Default, so this project still runs out of the box.
    ClaudeProvider    hosted, via the official `anthropic` SDK. Materially
                      better at the actual task (reading 10k characters of
                      statute and answering precisely) and ~10x faster.

Select with DPDP_PROVIDER. Everything else is overridable from the
environment, so switching providers or models is a shell variable, not a
code change:

    DPDP_PROVIDER      "ollama" (default) or "claude"
    OLLAMA_HOST        default http://127.0.0.1:11434
    DPDP_MODEL         default qwen2.5:3b-instruct (ollama) / claude-sonnet-5 (claude)
    DPDP_NUM_CTX       default 8192 (ollama only)
    ANTHROPIC_API_KEY  required only when DPDP_PROVIDER=claude
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from abc import ABC, abstractmethod

PROVIDER = os.environ.get("DPDP_PROVIDER", "ollama").lower()

_DEFAULT_MODEL = {"ollama": "qwen2.5:3b-instruct", "claude": "claude-sonnet-5"}
MODEL = os.environ.get("DPDP_MODEL", _DEFAULT_MODEL.get(PROVIDER, _DEFAULT_MODEL["ollama"]))
NUM_CTX = int(os.environ.get("DPDP_NUM_CTX", "8192"))

# Models known to be too small to answer legal questions reliably. Ollama-only
# — not a blocker, a warning, because they are what most laptops actually have.
SMALL = ("1b", "1.5b", "2b", "3b", "4b")


class Provider(ABC):
    @abstractmethod
    def models(self) -> list[str]: ...

    @abstractmethod
    def check(self, model: str) -> str | None:
        """Return an error string if `model` can't be used right now, else None."""

    @abstractmethod
    def chat(self, prompt: str, system: str | None, schema: dict | None,
              model: str, num_ctx: int, temperature: float, timeout: int) -> str | dict: ...

    @abstractmethod
    def chat_stream(self, prompt: str, system: str | None, model: str,
                     num_ctx: int, temperature: float, timeout: int): ...


# --------------------------------------------------------------------------- #
# Ollama — local, stdlib-only HTTP client
# --------------------------------------------------------------------------- #

class OllamaProvider(Provider):
    def __init__(self):
        self.host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")

    def _post(self, path: str, payload: dict, timeout: int) -> dict:
        request = urllib.request.Request(
            self.host + path, data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())

    def models(self) -> list[str]:
        try:
            with urllib.request.urlopen(self.host + "/api/tags", timeout=10) as response:
                return [m["name"] for m in json.loads(response.read()).get("models", [])]
        except OSError:
            return []

    def check(self, model: str) -> str | None:
        available = self.models()
        if not available:
            return (f"no Ollama server at {self.host}. Start it with `ollama serve`, "
                    f"or set OLLAMA_HOST.")
        if model not in available and f"{model}:latest" not in available:
            return (f"model '{model}' not installed. Have: {', '.join(available)}\n"
                    f"  pull it with:  ollama pull {model}")
        return None

    def chat(self, prompt, system, schema, model, num_ctx, temperature, timeout):
        messages = ([{"role": "system", "content": system}] if system else []) + \
                   [{"role": "user", "content": prompt}]
        payload = {
            "model": model, "messages": messages, "stream": False,
            "options": {"num_ctx": num_ctx, "temperature": temperature},
        }
        if schema:
            payload["format"] = schema
        try:
            content = self._post("/api/chat", payload, timeout)["message"]["content"]
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"ollama {e.code}: {e.read().decode('utf-8', 'replace')[:300]}")
        except OSError as e:
            raise RuntimeError(f"cannot reach ollama at {self.host}: {e}")

        if not schema:
            return content
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"model returned invalid JSON despite a schema: {e}\n{content[:300]}")

    def chat_stream(self, prompt, system, model, num_ctx, temperature, timeout):
        messages = ([{"role": "system", "content": system}] if system else []) + \
                   [{"role": "user", "content": prompt}]
        payload = {"model": model, "messages": messages, "stream": True,
                   "options": {"num_ctx": num_ctx, "temperature": temperature}}
        request = urllib.request.Request(
            self.host + "/api/chat", data=json.dumps(payload).encode("utf-8"),
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
            raise RuntimeError(f"cannot reach ollama at {self.host}: {e}")


# --------------------------------------------------------------------------- #
# Claude — hosted, via the official SDK
# --------------------------------------------------------------------------- #

class ClaudeProvider(Provider):
    """The `anthropic` package is imported lazily, inside methods, not at
    module scope — so this file still imports (and Ollama still works) on a
    machine that never installed it or never set an API key."""

    MAX_TOKENS = 4096

    def __init__(self):
        self._client = None

    def _get_client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        return self._client

    def models(self) -> list[str]:
        return [MODEL]  # no local pull step; whatever DPDP_MODEL names is "installed"

    def check(self, model: str) -> str | None:
        if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
            return ("ANTHROPIC_API_KEY is not set. Export it to use Claude, or "
                    "set DPDP_PROVIDER=ollama to run fully local instead.")
        return None

    def chat(self, prompt, system, schema, model, num_ctx, temperature, timeout):
        import anthropic
        client = self._get_client()
        kwargs = dict(model=model, max_tokens=self.MAX_TOKENS, temperature=temperature,
                      messages=[{"role": "user", "content": prompt}])
        if system:
            kwargs["system"] = system
        if schema:
            # Force a single tool call shaped exactly like `schema`, so the
            # reply is structured JSON rather than prose that might contain it.
            kwargs["tools"] = [{"name": "emit", "description": "Return the result.",
                                "input_schema": schema}]
            kwargs["tool_choice"] = {"type": "tool", "name": "emit"}
        try:
            response = client.with_options(timeout=float(timeout)).messages.create(**kwargs)
        except anthropic.APIError as e:
            raise RuntimeError(f"claude: {e}")

        if schema:
            for block in response.content:
                if block.type == "tool_use":
                    return block.input
            raise RuntimeError("model did not return the requested structured output")
        return "".join(b.text for b in response.content if b.type == "text")

    def chat_stream(self, prompt, system, model, num_ctx, temperature, timeout):
        import anthropic
        client = self._get_client()
        kwargs = dict(model=model, max_tokens=self.MAX_TOKENS, temperature=temperature,
                      messages=[{"role": "user", "content": prompt}])
        if system:
            kwargs["system"] = system
        try:
            with client.with_options(timeout=float(timeout)).messages.stream(**kwargs) as stream:
                yield from stream.text_stream
        except anthropic.APIConnectionError as e:
            raise RuntimeError(f"cannot reach the Claude API: {e}")
        except anthropic.APIError as e:
            raise RuntimeError(f"claude: {e}")


_PROVIDERS: dict[str, Provider] = {}


def _provider() -> Provider:
    if PROVIDER not in _PROVIDERS:
        if PROVIDER == "claude":
            _PROVIDERS[PROVIDER] = ClaudeProvider()
        elif PROVIDER == "ollama":
            _PROVIDERS[PROVIDER] = OllamaProvider()
        else:
            raise RuntimeError(f"unknown DPDP_PROVIDER '{PROVIDER}' (want 'ollama' or 'claude')")
    return _PROVIDERS[PROVIDER]


def models() -> list[str]:
    return _provider().models()


def check(model: str = None) -> str | None:
    """Return an error string if `model` (default: the module's MODEL, which
    callers may reassign at runtime — see ask.py's --model flag) can't be
    used right now, else None."""
    return _provider().check(model or MODEL)


def chat(prompt: str, system: str | None = None, schema: dict | None = None,
         model: str = None, num_ctx: int = None, temperature: float = 0.2,
         timeout: int = 600) -> str | dict:
    """One turn. With `schema`, the parsed object comes back; without it, the text."""
    return _provider().chat(prompt, system, schema, model or MODEL,
                            num_ctx or NUM_CTX, temperature, timeout)


def chat_stream(prompt: str, system: str | None = None, model: str = None,
                num_ctx: int = None, temperature: float = 0.2, timeout: int = 600):
    """Yield answer fragments as the model produces them."""
    yield from _provider().chat_stream(prompt, system, model or MODEL,
                                       num_ctx or NUM_CTX, temperature, timeout)


def warn_if_small(model: str = None) -> None:
    if PROVIDER == "ollama" and any(tag in (model or MODEL).lower() for tag in SMALL):
        print(f"note: {model or MODEL} is a small model. It is fine for indexing, but for\n"
              f"      answering legal questions a hosted model is materially better:\n"
              f"        set DPDP_PROVIDER=claude and export ANTHROPIC_API_KEY\n"
              f"      or, staying local:\n"
              f"        ollama pull qwen2.5:7b-instruct && set DPDP_MODEL=qwen2.5:7b-instruct\n",
              file=sys.stderr)
