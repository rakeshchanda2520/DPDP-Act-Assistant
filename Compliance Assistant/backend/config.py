"""
Configuration, validated once at import.

Every setting is read here and nowhere else. Two rules this enforces:

1. **Fail at startup, not at first request.** A missing NEO4J_PASSWORD should
   stop the process immediately with a clear message, not surface as a 500 to
   whoever asks the first question.
2. **Secrets never leave this module.** `public_settings()` is the only thing
   any route may return, and it is a hand-written allow-list. A blanket
   `dict(os.environ)` or a `__repr__` of a settings object is how credentials
   end up in a health endpoint or a log line.
"""
from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
FRONTEND_DIR = BASE_DIR / "frontend"
# Outside data/: data/ is rebuilt by kg_build, and an audit trail that a
# rebuild can delete is not an audit trail.
LOG_DIR = BASE_DIR / "logs"


def _env(name: str, default: str | None = None, *, required: bool = False) -> str:
    value = os.environ.get(name, default if default is not None else "")
    if required and not value:
        raise RuntimeError(
            f"{name} is not set. Copy .env.example to .env and fill it in.")
    return value


def _flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except ValueError:
        raise RuntimeError(f"{name} must be an integer, got {os.environ[name]!r}")


def load_dotenv(path: Path = BASE_DIR / ".env") -> None:
    """Minimal .env reader — no dependency for something this small.

    Deliberately does NOT overwrite variables already in the environment:
    a value injected by the container or CI must win over a stale file left
    on a developer's disk.
    """
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


load_dotenv()

# --- Neo4j: the graph is the source of truth at runtime -------------------- #
NEO4J_URI = _env("NEO4J_URI", required=True)
NEO4J_USER = _env("NEO4J_USER") or _env("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = _env("NEO4J_PASSWORD", required=True)
NEO4J_DATABASE = _env("NEO4J_DATABASE", "neo4j")

# --- Language model -------------------------------------------------------- #
PROVIDER = _env("DPDP_PROVIDER", "ollama").lower()
OLLAMA_HOST = _env("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
_DEFAULT_MODEL = {"ollama": "qwen2.5:3b-instruct", "claude": "claude-sonnet-5"}
MODEL = _env("DPDP_MODEL") or _DEFAULT_MODEL.get(PROVIDER, _DEFAULT_MODEL["ollama"])
NUM_CTX = _int("DPDP_NUM_CTX", 16384)
LLM_TIMEOUT = _int("DPDP_LLM_TIMEOUT", 600)

# --- Retrieval ------------------------------------------------------------- #
MAX_CONTEXT_CHARS = _int("DPDP_MAX_CONTEXT_CHARS", 10000)
# Below this BM25 score the corpus contains nothing resembling an answer.
# Calibrated against out-of-scope probes: clearly unrelated questions score
# 7-12, genuine ones 22-85. It catches the obvious end only — questions from
# adjacent legal domains (GDPR, HIPAA) share enough vocabulary to score high,
# so this is a first-line filter, not a domain classifier.
ABSTAIN_THRESHOLD = float(_env("DPDP_ABSTAIN_THRESHOLD", "15.0"))

# --- Tracing (optional) ----------------------------------------------------- #
# Off unless BOTH keys are set — no partial/broken state where a public key
# exists but tracing silently never authenticates. Never required: unlike
# NEO4J_*, an install with nothing set here just doesn't trace.
LANGFUSE_PUBLIC_KEY = _env("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = _env("LANGFUSE_SECRET_KEY")
LANGFUSE_HOST = _env("LANGFUSE_HOST", "https://cloud.langfuse.com")
TRACING_ENABLED = bool(LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY)

# --- HTTP ------------------------------------------------------------------ #
# Same-origin by default: the frontend is served by this app, so no
# cross-origin access is needed and none is granted. Set explicitly (comma
# separated) only if a separate origin must call the API. "*" is refused
# below rather than silently honoured.
_origins = _env("DPDP_CORS_ORIGINS", "")
CORS_ORIGINS = [o.strip() for o in _origins.split(",") if o.strip()]
if "*" in CORS_ORIGINS:
    raise RuntimeError(
        "DPDP_CORS_ORIGINS=* is refused. This API answers from a private "
        "corpus and writes an audit log; list real origins explicitly.")

# Surfaces internal error detail in HTTP responses. Off in production so a
# stack trace or a connection string can never reach a browser.
DEBUG = _flag("DPDP_DEBUG", False)


def public_settings() -> dict:
    """The ONLY settings any HTTP response may include.

    An allow-list, not a filter: adding a config value above must not
    silently make it public. Nothing here is a credential, a host, or a path.
    """
    return {
        "provider": PROVIDER,
        "model": MODEL,
        "abstain_threshold": ABSTAIN_THRESHOLD,
        "max_context_chars": MAX_CONTEXT_CHARS,
        "tracing_enabled": TRACING_ENABLED,
    }
