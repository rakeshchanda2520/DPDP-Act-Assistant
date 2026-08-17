"""
Dense retrieval: embeddings + Reciprocal Rank Fusion with BM25, plus a
cross-encoder reranker over the fused top-k.

FLOW.md's own "Stage 6 — the embeddings question" lays out exactly why this
project runs BM25-only by default (exact tokens like "§8(5)" and "250 crore"
are the whole game in law; embeddings blur precisely those distinctions) and
exactly what the concrete upgrade path looks like when that stops being
enough — this module is that upgrade path, implemented as specified there:
RRF fusion (never score-blending — BM25 and cosine scores aren't on
comparable scales), BM25 always kept in the mix (never dense-only), and gated
on eval.yaml / answer_eval.yaml rather than shipped on vibes.

Two local models, both loaded from the local Hugging Face cache with no
network call at query time:

    sentence-transformers/all-MiniLM-L6-v2      embeddings, ~80MB, CPU-fast
    cross-encoder/ms-marco-MiniLM-L-6-v2        reranker

`transformers`' TensorFlow auto-probe is broken on at least one dev machine's
DLL setup (unrelated import chain — this module never touches TensorFlow) and
crashes just importing `sentence_transformers`. USE_TF=0 must be set before
that import to skip the probe; done here at module load, before the import.

Off by default (DPDP_HYBRID=1 to enable) — this pulls in torch + transformers,
a much heavier dependency chain than the rest of this project, and FLOW.md's
BM25-only reasoning is correct until proven otherwise on THIS eval set, not
assumed better because it's newer. Turn it on to measure; see eval.yaml's
test_retrieval_eval_set with DPDP_HYBRID=1 for the actual before/after.
"""
from __future__ import annotations

import os

os.environ.setdefault("USE_TF", "0")
# Both models are expected to already be cached locally. Without this,
# sentence-transformers still makes an HTTP call per load to check for cache
# updates — on a machine with no/slow internet that's a multi-second-to-hung
# stall on every single startup, silently contradicting this module's own
# "no network call at query time" claim. Set DPDP_HF_ONLINE=1 to allow that
# revalidation check (e.g. the first time a new model name is used).
if os.environ.get("DPDP_HF_ONLINE", "") not in ("1", "true"):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

from functools import lru_cache

import numpy as np

EMBED_MODEL = os.environ.get("DPDP_EMBED_MODEL",
                             "sentence-transformers/all-MiniLM-L6-v2")
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RRF_K = 60          # the standard RRF constant from the original paper; not
                    # tuned against this corpus, just the well-known default
RERANK_POOL = 20    # candidates handed to the cross-encoder — cheap at this
                    # size, and TARGET.md's whole point is reranking only the
                    # fused shortlist, not all 142 chunks every query


@lru_cache(maxsize=1)
def _embedder():
    """Load EMBED_MODEL.

    Raw encoder checkpoints (`law-ai/InLegalBERT`) need no wrapper here:
    sentence-transformers detects the missing pooling head and composes
    Transformer + mean pooling itself, logging "Creating a new one with mean
    pooling". Verified — it loads at dim 768 with pooling_mode='mean', which
    is the right default for a BERT-family encoder. An explicit
    models.Transformer + models.Pooling composition was written for this and
    then deleted as dead code.
    """
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBED_MODEL)


@lru_cache(maxsize=1)
def _reranker():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(RERANK_MODEL)


@lru_cache(maxsize=4)
def chunk_embeddings(docs: tuple[str, ...]) -> np.ndarray:
    """Cached on the exact tuple of documents, so a changed index (a rebuild)
    naturally invalidates the cache instead of silently serving stale vectors.
    Encoding 142 short chunks on CPU is sub-second; this cache exists so a
    long-running server doesn't redo it on every single request."""
    return _embedder().encode(list(docs), normalize_embeddings=True, show_progress_bar=False)


def dense_ranks(query: str, doc_embeddings: np.ndarray) -> list[int]:
    """Cosine similarity ranking — a dot product, since embeddings are
    pre-normalised. Returns document indices, best match first."""
    q = _embedder().encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
    scores = doc_embeddings @ q
    return list(np.argsort(-scores))


def rrf_fuse(*rank_lists: list[int], k: int = RRF_K) -> dict[int, float]:
    """Reciprocal Rank Fusion across any number of ranked lists:
    RRF(d) = sum(1 / (k + rank)) over every list d appears in.

    Rank-based, not score-based, on purpose: BM25 scores and cosine
    similarities live on different, incomparable scales, and blending them
    numerically needs a normalisation constant that is really just a knob
    tuned until the eval passes. RRF sidesteps the whole problem — only
    ordinal rank position matters, so nothing needs normalising."""
    fused: dict[int, float] = {}
    for ranks in rank_lists:
        for rank, i in enumerate(ranks):
            fused[i] = fused.get(i, 0.0) + 1.0 / (k + rank + 1)
    return fused


def rerank(query: str, candidates: list[tuple[int, str]]) -> list[int]:
    """Cross-encoder reranking of the fused shortlist. A cross-encoder reads
    the query and the candidate TOGETHER through one model pass, rather than
    comparing two independently-computed vectors — markedly more accurate at
    final ordering than BM25 or embeddings alone. Only affordable because it
    runs on ~20 candidates, not the full 142-chunk corpus."""
    if not candidates:
        return []
    pairs = [(query, text) for _, text in candidates]
    scores = _reranker().predict(pairs)
    order = np.argsort(-np.asarray(scores))
    return [candidates[i][0] for i in order]
