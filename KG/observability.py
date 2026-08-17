"""
Optional Langfuse tracing.

Enabled only when LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are both set in
the environment — the same shape as llm.py's ClaudeProvider (checked
explicitly before use, imported lazily) and hybrid.py's DPDP_HYBRID gate:
real behavior when configured, zero dependency or behavior cost when not.
Nothing in ask.py or api.py should import `langfuse` directly — go through
this module so the rest of the codebase never has to know whether tracing is
on.

Every request becomes one root trace with three nested child observations —
`retrieve` (as_type="retriever"), `generate` (as_type="generation"), and
`verify_citations` (as_type="evaluator") — deliberately mirroring the three
stages api.py already treats as distinct (the SSE event stages, and
audit_log.jsonl's own record shape), so a trace in the Langfuse UI and a
line in the local audit log describe the same request the same way.

Run:  export LANGFUSE_PUBLIC_KEY=pk-lf-...
      export LANGFUSE_SECRET_KEY=sk-lf-...
      # LANGFUSE_BASE_URL defaults to https://cloud.langfuse.com;
      # point it at a self-hosted instance instead if you run one.
"""
from __future__ import annotations

import os
from contextlib import contextmanager

ENABLED = bool(os.environ.get("LANGFUSE_PUBLIC_KEY")) and bool(os.environ.get("LANGFUSE_SECRET_KEY"))

_client = None


def _get_client():
    """Imported lazily so a process that never configures Langfuse never
    pays for importing it (and its opentelemetry dependency chain) — same
    reasoning as llm.py's `import anthropic` inside ClaudeProvider's methods."""
    global _client
    if _client is None:
        from langfuse import get_client
        _client = get_client()
    return _client


def check() -> str | None:
    """Return an error string if Langfuse is configured but unreachable or
    misauthenticated, else None. Mirrors llm.check()'s contract, so a caller
    that already knows how to surface `llm.check()`'s result can handle this
    one identically."""
    if not ENABLED:
        return None
    try:
        if not _get_client().auth_check():
            return "Langfuse credentials are set but authentication failed"
    except Exception as e:
        return f"cannot reach Langfuse: {e}"
    return None


@contextmanager
def trace(name: str, **fields):
    """Root span for one request. A no-op (yields None) when disabled, so
    callers can unconditionally write `with observability.trace(...) as t:`
    and just guard any direct use of `t` behind `if t:` — never behind
    `if observability.ENABLED:` scattered through the caller."""
    if not ENABLED:
        yield None
        return
    with _get_client().start_as_current_observation(name=name, as_type="span", **fields) as span:
        yield span


@contextmanager
def step(name: str, as_type: str = "span", **fields):
    """A child observation nested under whichever trace/step is currently
    open (Langfuse's context manager nests via contextvars, matching how
    these are always used here — inside an open `trace(...)` block). Also a
    no-op when disabled.

    as_type: "span" (default), "generation", "retriever", "evaluator",
    "embedding", "agent", "tool", "chain", "guardrail" — see Langfuse's
    ObservationTypeLiteral. Picking the specific type ("retriever" for the
    BM25/hybrid retrieval step, "generation" for the LLM call, "evaluator"
    for citation verification) is not cosmetic: Langfuse renders each type
    differently in the UI and a "generation" observation is what unlocks
    token-usage/cost tracking if usage_details are ever passed to update().
    """
    if not ENABLED:
        yield None
        return
    with _get_client().start_as_current_observation(name=name, as_type=as_type, **fields) as obs:
        yield obs


def flush() -> None:
    """Langfuse batches events and sends them asynchronously in the
    background. A short-lived process (the ask.py CLI) MUST call this before
    exit or buffered events are silently dropped when the process ends. A
    long-lived process (the uvicorn server) doesn't strictly need this per
    request — the background flush thread would eventually send everything
    — but calling it keeps trace latency predictable rather than deferred,
    and the cost is negligible at this request volume."""
    if ENABLED and _client is not None:
        _client.flush()
