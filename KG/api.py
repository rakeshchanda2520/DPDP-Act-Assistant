"""
FastAPI backend for the DPDP assistant.

Streams over SSE in four stages, so the user sees work happening instead of a
spinner:

    retrieval  the provisions found, and how — sent before the model is called
    token      answer fragments as they are generated
    citations  every § the model cited, CHECKED against the graph
    done       timings

The citations stage is the point of the whole thing. A language model can write
"§8(5)" whether or not §8(5) says what it claims — so every citation is resolved
against the verbatim graph and labelled:

    verified        the provision exists AND was in the retrieved context
    out_of_context  the provision exists but was NOT retrieved — the model
                    recalled it from training. Treat with suspicion.
    unresolved      no such provision. The model invented it.

That three-way split is the accuracy proof: the user can click any citation and
read the Act's own words.

Run:  uvicorn api:app --reload --port 8000
      http://127.0.0.1:8000
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import sys
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

import ask
import llm

ROOT = Path(__file__).parent
WEB = ROOT / "web"
# Deliberately OUTSIDE out/: that directory is disposable and gets wiped on a
# clean rebuild, and an audit trail must survive that — same reasoning as
# plain_language.json living at the repo root instead of under out/.
LOGS = ROOT / "logs"

# "§8(5)", "§ 8 (5)(a)", "section 8(5)", "Schedule entry 2", "Schedule 2".
RE_CITATION = re.compile(
    r"(?:§\s*|\bsections?\s+)(\d{1,2})((?:\s*\(\s*[0-9a-zA-Z]{1,3}\s*\))*)"
    r"|\bSchedule\s+(?:entry\s+)?(\d)\b", re.IGNORECASE)
RE_PART = re.compile(r"\(\s*([0-9a-zA-Z]{1,3})\s*\)")

# Below this BM25 score, retrieval found essentially nothing — a genuinely
# unrelated question ("how do I bake a chocolate cake?") scores 7-12 here.
#
# Calibrated against 8 out-of-scope probes and 6 real DPDP questions (see
# eval_answers.py / answer_eval.yaml's `abstain: true` cases). Honest result:
# this threshold only catches the CLEARLY unrelated end — income tax filing
# (10.6), "capital of France" (11.9), cryptocurrency legality (7.8). It does
# NOT catch adjacent-domain questions that share real legal vocabulary with
# the Act: "GDPR penalty" scored 80.9, "HIPAA" 33.8, "RBI KYC cycle" 24.5 —
# all inside or above the range of genuine in-scope questions (22.7-85.2).
# That gap is real and is exactly the failure mode TARGET.md's hybrid-
# retrieval phase is meant to close with an actual out-of-domain signal, not
# a lexical score. Keep this threshold low and treat it as a first-line
# filter for the obvious cases, not a complete classifier.
ABSTAIN_THRESHOLD = 15.0

app = FastAPI(title="DPDP Act 2023 assistant", version="1.0")

STATE: dict = {}


def compute_build_id(graph: dict) -> str:
    """A content hash of the graph, not a version number someone has to
    remember to bump. Stamped on every response and every audit record, so
    "which answers are now stale?" is answerable the moment the Act (or the
    Rules, later) is amended and the graph is rebuilt — the hash changes,
    and every record before it is visibly from the old text."""
    payload = json.dumps(graph, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


@app.on_event("startup")
def startup() -> None:
    """Load the index once. BM25 is rebuilt per query (142 docs, milliseconds),
    but the JSON parsing is not free and does not belong in the request path."""
    index, graph, vocab = ask.load()
    LOGS.mkdir(exist_ok=True)
    STATE.update(
        index=index, graph=graph, vocab=vocab,
        tree=json.loads((ROOT / "out" / "dpdp_tree.json").read_text(encoding="utf-8")),
        nodes={n["id"]: n for n in graph["nodes"]},
        build_id=compute_build_id(graph),
    )
    print(f"loaded {len(index['chunks'])} chunks, {len(graph['nodes'])} nodes; "
          f"model {llm.MODEL}; build {STATE['build_id']}")


def should_abstain(results: list[dict]) -> str | None:
    """Return a reason to abstain, or None to proceed to generation.

    The only prior line of defence against an out-of-scope question was the
    model's own judgement — and the whole reason this project distrusts a
    small model's judgement on in-scope questions is exactly why it should
    not be trusted alone on the harder question of "should I answer at all".
    A BM25 score threshold is deterministic, auditable, and tunable against
    eval data the same way every other retrieval decision in this system is.
    """
    top = max((r["score"] for r in results if r["hop"] == 0), default=0.0)
    if top < ABSTAIN_THRESHOLD:
        return (f"the closest match scored {top:.1f}, well below what a real "
                f"provision of this Act scores for an in-scope question")
    return None


def audit_log(record: dict) -> None:
    """One line per request. Append-only, never rewritten — an audit trail
    that could be silently edited after the fact is not an audit trail."""
    line = json.dumps(record, ensure_ascii=False)
    with open(LOGS / "audit_log.jsonl", "a", encoding="utf-8") as f:
        f.write(line + "\n")


class Question(BaseModel):
    question: str = Field(min_length=2, max_length=800)
    k: int = Field(default=6, ge=1, le=15)
    parent_doc: bool = False
    hybrid: bool | None = None  # None = defer to DPDP_HYBRID env var


# --------------------------------------------------------------------------- #
# Citation checking
# --------------------------------------------------------------------------- #

def citation_id(section: str, parts: str) -> str:
    return "-".join(["s", section] + RE_PART.findall(parts or ""))


def label_for(node_id: str) -> str:
    if node_id.startswith("pen-"):
        return f"Schedule entry {node_id[4:]}"
    bits = node_id.split("-")[1:]
    return f"§{bits[0]}" + "".join(f"({b})" for b in bits[1:])


def check_citations(answer: str, retrieved_ids: set[str]) -> list[dict]:
    """Resolve every citation in the answer against the verbatim graph."""
    tree, nodes = STATE["tree"], STATE["nodes"]
    seen: dict[str, dict] = {}

    for match in RE_CITATION.finditer(answer):
        section, parts, schedule = match.groups()
        node_id = f"pen-{schedule}" if schedule else citation_id(section, parts)
        if node_id in seen:
            continue

        source = tree.get(node_id) or nodes.get(node_id)
        if source is None:
            # Fall back to the parent: a model that writes §8(5)(z) is still
            # pointing at §8(5), and saying so is more useful than "invented".
            parent = "-".join(node_id.split("-")[:-1])
            if len(node_id.split("-")) > 2 and (tree.get(parent) or nodes.get(parent)):
                seen[node_id] = {
                    "id": node_id, "label": label_for(node_id), "status": "unresolved",
                    "text": "", "note": f"no such provision; nearest is {label_for(parent)}"}
            else:
                seen[node_id] = {
                    "id": node_id, "label": label_for(node_id), "status": "unresolved",
                    "text": "", "note": "no such provision in this Act"}
            continue

        text = source.get("text") or source.get("penalty") or ""
        if node_id.startswith("pen-"):
            text = f"{source.get('text', '')}  —  {source.get('penalty', '')}"
        in_context = node_id in retrieved_ids or any(
            r.startswith(node_id + "-") or node_id.startswith(r + "-") for r in retrieved_ids)
        seen[node_id] = {
            "id": node_id,
            "label": label_for(node_id),
            "headnote": source.get("headnote", ""),
            "status": "verified" if in_context else "out_of_context",
            "text": text.strip(),
            "note": "" if in_context else
                    "this provision exists but was not retrieved for this question",
        }

    order = {"unresolved": 0, "out_of_context": 1, "verified": 2}
    return sorted(seen.values(), key=lambda c: (order[c["status"]], c["id"]))


def penalty_facts(results: list[dict]) -> list[dict]:
    """Penalty amounts straight from the graph. Small models misread the
    Schedule; these values are never routed through the model."""
    nodes = STATE["nodes"]
    duty_of: dict[str, list[str]] = {}
    for e in STATE["graph"]["links"]:
        if e["type"] == "PENALISED_BY":
            duty_of.setdefault(e["target"], []).append(e["source"])
    return [{
        "entry": r["label"],
        "amount": nodes.get(r["node_id"], {}).get("penalty", ""),
        "applies_to": [label_for(d) for d in sorted(duty_of.get(r["node_id"], []))],
    } for r in results if r["kind"] == "Penalty"]


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #

@app.get("/api/health")
def health() -> dict:
    error = llm.check()
    return {"ok": error is None, "model": llm.MODEL, "provider": llm.PROVIDER,
            "detail": error or "ready",
            "chunks": len(STATE["index"]["chunks"]),
            "nodes": len(STATE["graph"]["nodes"]),
            "build_id": STATE["build_id"]}


@app.get("/api/provision/{node_id}")
def provision(node_id: str) -> dict:
    """Verbatim text of one provision — what a citation click opens."""
    source = STATE["tree"].get(node_id) or STATE["nodes"].get(node_id)
    if source is None:
        raise HTTPException(404, f"no provision {node_id}")
    return {"id": node_id, "label": label_for(node_id),
            "headnote": source.get("headnote", ""),
            "page": source.get("page", 0),
            "text": source.get("text", ""), "penalty": source.get("penalty", "")}


@app.post("/api/chat")
async def chat(q: Question) -> EventSourceResponse:
    started = time.time()
    request_id = uuid.uuid4().hex[:12]

    async def stream():
        # 1. Retrieve. Send it immediately — it is instant, and showing which
        #    provisions were found makes the wait legible instead of blank.
        results, trace = await asyncio.to_thread(
            ask.retrieve, STATE["index"], STATE["graph"], STATE["vocab"], q.question, q.k,
            q.parent_doc, q.hybrid)
        if not results:
            audit_log({"request_id": request_id, "ts": time.time(), "question": q.question,
                      "build_id": STATE["build_id"], "outcome": "no_results"})
            yield {"event": "error",
                   "data": json.dumps({"message":
                                       "Nothing in this Act matches that question."})}
            return

        retrieved_ids = {r["node_id"] for r in results}
        yield {"event": "retrieval", "data": json.dumps({
            "vocabulary": trace["vocab_hits"],
            "intents": trace["intents"],
            "elapsed_ms": int((time.time() - started) * 1000),
            "build_id": STATE["build_id"],
            "provisions": [{
                "id": r["node_id"], "label": r["label"], "kind": r["kind"],
                "headnote": r.get("headnote", ""), "hop": r["hop"],
                "score": r["score"], "via": r.get("via", ""),
            } for r in results],
        })}

        # 1a. Abstain before spending a generation call on a question this Act
        # plainly does not cover — see should_abstain's docstring for why this
        # is a threshold, not a judgement call left to the model.
        if reason := should_abstain(results):
            audit_log({"request_id": request_id, "ts": time.time(), "question": q.question,
                      "build_id": STATE["build_id"], "outcome": "abstained", "reason": reason,
                      "top_score": max((r["score"] for r in results if r["hop"] == 0), default=0.0)})
            yield {"event": "abstain", "data": json.dumps({
                "message": "This doesn't look like something the Digital Personal Data "
                          "Protection Act, 2023 covers — the closest match in the Act "
                          "was too weak to answer from. Try rephrasing, or this may be "
                          "outside this Act's scope entirely.",
                "reason": reason,
            })}
            return

        if error := llm.check():
            audit_log({"request_id": request_id, "ts": time.time(), "question": q.question,
                      "build_id": STATE["build_id"], "outcome": "llm_unavailable", "error": error})
            yield {"event": "error", "data": json.dumps({"message": error})}
            return

        # 2. Stream the answer.
        prompt = f"{ask.build_context(results)}\n\nQuestion: {q.question}"
        chunks: list[str] = []
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def produce() -> None:
            try:
                for fragment in llm.chat_stream(prompt, system=ask.SYSTEM,
                                                num_ctx=ask.ANSWER_NUM_CTX,
                                                temperature=0.1):
                    loop.call_soon_threadsafe(queue.put_nowait, ("token", fragment))
            except RuntimeError as e:
                loop.call_soon_threadsafe(queue.put_nowait, ("error", str(e)))
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, ("eof", None))

        asyncio.get_running_loop().run_in_executor(None, produce)

        while True:
            kind, payload = await queue.get()
            if kind == "eof":
                break
            if kind == "error":
                audit_log({"request_id": request_id, "ts": time.time(), "question": q.question,
                          "build_id": STATE["build_id"], "outcome": "generation_error",
                          "error": payload})
                yield {"event": "error", "data": json.dumps({"message": payload})}
                return
            chunks.append(payload)
            yield {"event": "token", "data": json.dumps({"t": payload})}

        # 3. Check what it cited, and hand back the amounts from the graph.
        answer = "".join(chunks)
        citations = check_citations(answer, retrieved_ids)
        elapsed_ms = int((time.time() - started) * 1000)
        yield {"event": "citations", "data": json.dumps({
            "citations": citations,
            "penalties": penalty_facts(results),
        })}
        yield {"event": "done", "data": json.dumps({
            "elapsed_ms": elapsed_ms,
            "model": llm.MODEL,
            "provider": llm.PROVIDER,
            "build_id": STATE["build_id"],
            "context_chars": len(prompt),
        })}

        # Everything needed to reconstruct and investigate this answer later:
        # what was retrieved, what was sent to the model, what it said, and
        # how each citation resolved. This is the record a "your tool told me
        # X yesterday, was that right?" complaint gets investigated against.
        audit_log({
            "request_id": request_id, "ts": time.time(), "question": q.question,
            "build_id": STATE["build_id"], "outcome": "answered",
            "model": llm.MODEL, "provider": llm.PROVIDER,
            "elapsed_ms": elapsed_ms, "context_chars": len(prompt),
            "retrieved": [{"id": r["node_id"], "hop": r["hop"], "score": r["score"]}
                         for r in results],
            "answer": answer,
            "citations": [{"id": c["id"], "status": c["status"]} for c in citations],
        })

    return EventSourceResponse(stream())


@app.get("/")
def index_page() -> FileResponse:
    return FileResponse(WEB / "index.html")
