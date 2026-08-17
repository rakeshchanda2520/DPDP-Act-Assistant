"""
HTTP layer.

`POST /api/chat` streams over SSE in stages so the wait is legible instead of
blank:

    retrieval   which provisions were found, before the model is called
    token       answer fragments as they are produced
    citations   every citation, checked against the graph
    done        timings and the build id
    abstain     sent instead of an answer when the corpus does not cover it
    error       a safe, human-readable failure

Security posture of this module:
  * No internal detail in responses. Exceptions are logged with a request id;
    the client receives that id and a generic message unless DEBUG is on.
  * Request bodies are bounded by Pydantic before any work is done.
  * The frontend is served from a fixed file, never a client-supplied path.
  * CORS is same-origin unless origins are configured explicitly.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, field_validator
from sse_starlette.sse import EventSourceResponse

from . import audit, citations, config, graph_store, llm, observability, retrieval
from .indexing import load_chunks
from .prompt import SYSTEM_PROMPT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
log = logging.getLogger("compliance")

STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load everything once, and fail loudly here rather than on request one."""
    graph = graph_store.load_graph()
    chunks = load_chunks(config.DATA_DIR / "chunks.json")
    vocab = yaml.safe_load((config.DATA_DIR / "vocab.yaml").read_text(encoding="utf-8"))

    STATE["graph"] = graph
    STATE["retriever"] = retrieval.Retriever(chunks, graph, vocab)
    STATE["audit"] = audit.AuditLog(config.LOG_DIR)
    STATE["chunk_count"] = len(chunks)

    log.info("ready: %d provisions, %d chunks, build %s, model %s/%s",
             len(graph.provisions), len(chunks), graph.build_id,
             config.PROVIDER, config.MODEL)
    if llm.is_small_model():
        log.warning("%s is a small model; it has misread statute in this "
                    "corpus. Prefer a larger model for real use.", config.MODEL)
    yield
    STATE.clear()


app = FastAPI(title="DPDP Compliance Assistant", version="2.0",
              lifespan=lifespan,
              # No interactive docs in production: they enumerate the API
              # surface for anyone who finds the port.
              docs_url="/docs" if config.DEBUG else None,
              redoc_url=None,
              openapi_url="/openapi.json" if config.DEBUG else None)

if config.CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.CORS_ORIGINS,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )


@app.middleware("http")
async def security_headers(request: Request, call_next):
    """Headers that cost nothing and close real classes of attack.

    The CSP is strict because the frontend needs nothing else: no inline
    event handlers, no remote scripts, no framing. `connect-src 'self'` means
    a script injected into the page cannot exfiltrate an answer to another
    host. Google Fonts is the one external origin, and it is style/font only.
    """
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    response.headers.setdefault(
        "Content-Security-Policy",
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "font-src 'self' https://fonts.gstatic.com; "
        "img-src 'self' data:; "
        "connect-src 'self'; "
        "frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
    return response


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception) -> JSONResponse:
    """Nothing internal reaches the client. The id ties the user's report to
    the stack trace in the log."""
    request_id = audit.AuditLog.new_request_id()
    log.exception("unhandled error [%s] on %s", request_id, request.url.path)
    detail = str(exc) if config.DEBUG else "internal error"
    return JSONResponse(status_code=500,
                        content={"error": detail, "request_id": request_id})


class Question(BaseModel):
    # Bounded before any work happens: an unbounded question becomes an
    # unbounded prompt, which is both a cost and a context-overflow problem.
    question: str = Field(min_length=2, max_length=800)
    k: int = Field(default=6, ge=1, le=15)

    @field_validator("question")
    @classmethod
    def clean(cls, value: str) -> str:
        # Control characters can corrupt the SSE framing and the audit log.
        text = "".join(ch for ch in value if ch == "\n" or ch >= " ").strip()
        if not text:
            raise ValueError("question is empty")
        return text


@app.get("/api/health")
def health() -> dict:
    graph = STATE.get("graph")
    llm_error = llm.check()
    return {
        "ok": llm_error is None and graph is not None,
        "detail": llm_error or "ready",
        "provisions": len(graph.provisions) if graph else 0,
        "chunks": STATE.get("chunk_count", 0),
        "build_id": graph.build_id if graph else "",
        "tracing_detail": observability.check() or (
            "ready" if config.TRACING_ENABLED else "not configured"),
        **config.public_settings(),
    }


@app.get("/api/provision/{node_id}")
def provision(node_id: str) -> dict:
    """Verbatim text of one provision — what a citation click opens.

    `node_id` is used only as a dictionary key against provisions loaded from
    Neo4j, so an unknown or hostile value can only ever miss and 404.
    """
    graph: graph_store.Graph = STATE["graph"]
    found = graph.provisions.get(node_id)
    if found is None:
        raise HTTPException(status_code=404, detail="no such provision")
    return {"id": found.id, "label": found.label, "kind": found.kind,
            "headnote": found.headnote, "text": found.text,
            "penalty": found.penalty, "page": found.page}


def _sse(event: str, payload: dict) -> dict:
    return {"event": event, "data": json.dumps(payload, ensure_ascii=False)}


@app.post("/api/chat")
async def chat(q: Question) -> EventSourceResponse:
    graph: graph_store.Graph = STATE["graph"]
    retriever: retrieval.Retriever = STATE["retriever"]
    trail: audit.AuditLog = STATE["audit"]
    request_id = trail.new_request_id()
    started = time.perf_counter()

    def elapsed_ms() -> int:
        return int((time.perf_counter() - started) * 1000)

    async def stream():
        base = {"request_id": request_id, "question": q.question,
                "build_id": graph.build_id}

        # One Langfuse trace per request, with a child observation per stage
        # — the same three stages as the SSE events above and the fields
        # audit.py writes, so a trace in the Langfuse UI, the network tab,
        # and a line in the local audit log describe one request the same
        # way. A no-op everywhere tracing isn't configured (observability.py)
        # — this function's control flow is identical whether it's on or off.
        with observability.trace(
                "compliance.answer", input=q.question,
                metadata={"request_id": request_id, "k": q.k}) as root:

            # 1. Retrieve. Sent immediately: it is fast, and showing the
            #    evidence before the argument is the whole trust model.
            with observability.step("retrieve", as_type="retriever",
                                    input=q.question) as retr_span:
                results, trace = await asyncio.to_thread(
                    retriever.retrieve, q.question, q.k)
                if retr_span:
                    retr_span.update(
                        output=[r.chunk.node_id for r in results],
                        metadata={"vocab_hits": trace.vocab_hits,
                                 "intents": trace.intents})

            if not results:
                trail.write({**base, "outcome": "no_results"})
                if root:
                    root.update(output="no_results", level="WARNING")
                observability.flush()
                yield _sse("abstain", {
                    "message": "Nothing in this Act matches that question.",
                    "reason": "no provision scored above zero"})
                return

            yield _sse("retrieval", {
                "elapsed_ms": elapsed_ms(),
                "build_id": graph.build_id,
                "vocabulary": trace.vocab_hits,
                "intents": trace.intents,
                "provisions": [{
                    "id": r.chunk.node_id, "label": r.chunk.label, "kind": r.chunk.kind,
                    "headnote": r.chunk.headnote, "hop": r.hop,
                    "score": r.score, "via": r.via,
                } for r in results],
            })

            # 2. Abstain before spending a generation call on an out-of-scope
            #    question — deterministic, not left to the model's judgement.
            if reason := retrieval.should_abstain(results, config.ABSTAIN_THRESHOLD):
                trail.write({**base, "outcome": "abstained", "reason": reason})
                if root:
                    root.update(output=f"abstained: {reason}", level="WARNING")
                observability.flush()
                yield _sse("abstain", {
                    "message": "This doesn't look like something the Digital Personal "
                               "Data Protection Act, 2023 covers. The closest match was "
                               "too weak to answer from — try rephrasing, or this may "
                               "be outside the Act's scope.",
                    "reason": reason})
                return

            if error := llm.check():
                trail.write({**base, "outcome": "llm_unavailable", "error": error})
                if root:
                    root.update(output=f"llm_unavailable: {error}", level="ERROR")
                observability.flush()
                yield _sse("error", {"message": error})
                return

            # 3. Generate. The blocking provider call runs on a worker thread
            #    and feeds this coroutine through a queue, so the event loop
            #    keeps serving other requests while one answer streams.
            context = retrieval.build_context(results, config.MAX_CONTEXT_CHARS)
            prompt = f"{context}\n\nQuestion: {q.question}"
            queue: asyncio.Queue = asyncio.Queue()
            loop = asyncio.get_running_loop()

            def produce() -> None:
                try:
                    for fragment in llm.stream(prompt, SYSTEM_PROMPT):
                        loop.call_soon_threadsafe(queue.put_nowait, ("token", fragment))
                except llm.LLMError as exc:
                    loop.call_soon_threadsafe(queue.put_nowait, ("error", str(exc)))
                except Exception:                    # noqa: BLE001
                    log.exception("generation failed [%s]", request_id)
                    loop.call_soon_threadsafe(
                        queue.put_nowait, ("error", "the answer could not be generated"))
                finally:
                    loop.call_soon_threadsafe(queue.put_nowait, ("eof", None))

            with observability.step("generate", as_type="generation",
                                    model=config.MODEL, input=prompt) as gen_span:
                loop.run_in_executor(None, produce)

                parts: list[str] = []
                while True:
                    kind, payload = await queue.get()
                    if kind == "eof":
                        break
                    if kind == "error":
                        trail.write({**base, "outcome": "generation_error",
                                    "error": payload})
                        if gen_span:
                            gen_span.update(level="ERROR", status_message=payload)
                        if root:
                            root.update(output=f"generation_error: {payload}",
                                       level="ERROR")
                        observability.flush()
                        yield _sse("error", {"message": payload})
                        return
                    parts.append(payload)
                    yield _sse("token", {"t": payload})

                answer = "".join(parts)
                if gen_span:
                    gen_span.update(output=answer)

            # 4. Check what it cited, and render amounts from the graph.
            with observability.step("verify_citations", as_type="evaluator",
                                    input=answer) as verify_span:
                retrieved_ids = {r.chunk.node_id for r in results}
                checked = citations.check(answer, retrieved_ids, graph)
                penalties = citations.penalty_facts(results, graph)
                if verify_span:
                    verify_span.update(
                        output=[{"id": c.id, "status": c.status} for c in checked])

            yield _sse("citations", {
                "citations": [c.to_dict() for c in checked],
                "penalties": penalties})
            yield _sse("done", {
                "elapsed_ms": elapsed_ms(),
                "model": config.MODEL,
                "provider": config.PROVIDER,
                "build_id": graph.build_id,
                "context_chars": len(prompt)})

            if root:
                root.update(output=answer, metadata={
                    "citation_statuses": [c.status for c in checked],
                    "elapsed_ms": elapsed_ms()})
            observability.flush()

            trail.write({
                **base,
                "outcome": "answered",
                "model": config.MODEL,
                "provider": config.PROVIDER,
                "elapsed_ms": elapsed_ms(),
                "context_chars": len(prompt),
                "retrieved": [{"id": r.chunk.node_id, "hop": r.hop, "score": r.score}
                             for r in results],
                "answer": answer,
                "citations": [{"id": c.id, "status": c.status} for c in checked],
            })

    return EventSourceResponse(stream())


@app.get("/")
def index() -> FileResponse:
    """Serves one fixed file. No path is ever taken from the request, so
    directory traversal is not reachable here."""
    page: Path = config.FRONTEND_DIR / "index.html"
    if not page.is_file():
        raise HTTPException(status_code=500, detail="frontend not installed")
    return FileResponse(page)
