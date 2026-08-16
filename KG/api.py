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
import json
import re
import sys
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

import ask
import llm

ROOT = Path(__file__).parent
WEB = ROOT / "web"

# "§8(5)", "§ 8 (5)(a)", "section 8(5)", "Schedule entry 2", "Schedule 2".
RE_CITATION = re.compile(
    r"(?:§\s*|\bsections?\s+)(\d{1,2})((?:\s*\(\s*[0-9a-zA-Z]{1,3}\s*\))*)"
    r"|\bSchedule\s+(?:entry\s+)?(\d)\b", re.IGNORECASE)
RE_PART = re.compile(r"\(\s*([0-9a-zA-Z]{1,3})\s*\)")

app = FastAPI(title="DPDP Act 2023 assistant", version="1.0")

STATE: dict = {}


@app.on_event("startup")
def startup() -> None:
    """Load the index once. BM25 is rebuilt per query (142 docs, milliseconds),
    but the JSON parsing is not free and does not belong in the request path."""
    index, graph, vocab = ask.load()
    STATE.update(
        index=index, graph=graph, vocab=vocab,
        tree=json.loads((ROOT / "out" / "dpdp_tree.json").read_text(encoding="utf-8")),
        nodes={n["id"]: n for n in graph["nodes"]},
    )
    print(f"loaded {len(index['chunks'])} chunks, {len(graph['nodes'])} nodes; "
          f"model {llm.MODEL}")


class Question(BaseModel):
    question: str = Field(min_length=2, max_length=800)
    k: int = Field(default=6, ge=1, le=15)


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
    return {"ok": llm.check() is None, "model": llm.MODEL,
            "detail": llm.check() or "ready",
            "chunks": len(STATE["index"]["chunks"]),
            "nodes": len(STATE["graph"]["nodes"])}


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

    async def stream():
        # 1. Retrieve. Send it immediately — it is instant, and showing which
        #    provisions were found makes the wait legible instead of blank.
        results, trace = await asyncio.to_thread(
            ask.retrieve, STATE["index"], STATE["graph"], STATE["vocab"], q.question, q.k)
        if not results:
            yield {"event": "error",
                   "data": json.dumps({"message":
                                       "Nothing in this Act matches that question."})}
            return

        retrieved_ids = {r["node_id"] for r in results}
        yield {"event": "retrieval", "data": json.dumps({
            "vocabulary": trace["vocab_hits"],
            "intents": trace["intents"],
            "elapsed_ms": int((time.time() - started) * 1000),
            "provisions": [{
                "id": r["node_id"], "label": r["label"], "kind": r["kind"],
                "headnote": r.get("headnote", ""), "hop": r["hop"],
                "score": r["score"], "via": r.get("via", ""),
            } for r in results],
        })}

        if error := llm.check():
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
                yield {"event": "error", "data": json.dumps({"message": payload})}
                return
            chunks.append(payload)
            yield {"event": "token", "data": json.dumps({"t": payload})}

        # 3. Check what it cited, and hand back the amounts from the graph.
        answer = "".join(chunks)
        yield {"event": "citations", "data": json.dumps({
            "citations": check_citations(answer, retrieved_ids),
            "penalties": penalty_facts(results),
        })}
        yield {"event": "done", "data": json.dumps({
            "elapsed_ms": int((time.time() - started) * 1000),
            "model": llm.MODEL,
            "context_chars": len(prompt),
        })}

    return EventSourceResponse(stream())


@app.get("/")
def index_page() -> FileResponse:
    return FileResponse(WEB / "index.html")
