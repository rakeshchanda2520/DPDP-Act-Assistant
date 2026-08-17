"""
Optional NLI citation-content verification.

`api.check_citations()` answers "does this provision exist, and was it
retrieved?" — it does NOT answer "is what the model said about it actually
true of its text." That gap is exactly the project's canonical bug: asked
whether a dead customer's family can access the account, a 3B model answered
"family can step in" and cited §14. §14 exists, §14 was retrieved, so the
citation was labelled `verified` — while the claim built on it was wrong
(§14 gives the right to *nominate any other individual*, not a general
family right).

This module closes that gap with natural-language inference: each sentence
of the answer is checked against the verbatim text of the provisions it
cites, and a sentence the provisions do not entail is reported as
`unsupported`. That catches the whole *class* of misstatement automatically,
rather than needing a hand-written `must_not_say` regression case in
answer_eval.yaml for every new instance of it.

Off by default (DPDP_ENTAILMENT=1 to enable), same discipline as hybrid.py:
it adds a ~400MB model and per-sentence inference to the request path, so it
is opt-in and measured rather than assumed better.

    DPDP_ENTAILMENT=1        enable
    DPDP_ENTAILMENT_MODEL    default cross-encoder/nli-deberta-v3-base
    DPDP_ENTAILMENT_MIN      default 0.5 — entailment probability below which
                             a sentence is reported unsupported
"""
from __future__ import annotations

import os
import re

os.environ.setdefault("USE_TF", "0")
if os.environ.get("DPDP_HF_ONLINE", "") not in ("1", "true"):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

from functools import lru_cache

ENABLED = os.environ.get("DPDP_ENTAILMENT", "") not in ("", "0", "false")
MODEL = os.environ.get("DPDP_ENTAILMENT_MODEL", "cross-encoder/nli-deberta-v3-base")
MIN_ENTAILMENT = float(os.environ.get("DPDP_ENTAILMENT_MIN", "0.5"))

# Sentences shorter than this are things like "Short answer:" or "Why:" —
# section headers from the answer format, not claims to verify.
MIN_CLAIM_CHARS = 25

# The structural scaffolding of the answer format defined in ask.SYSTEM.
# These are not legal claims and must not be entailment-checked — doing so
# would flag every well-formed answer.
RE_SECTION_HEADER = re.compile(
    r"^\s*(Short answer|Why|The law says|What to do|Penalty)\s*:", re.IGNORECASE)


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(MODEL)


def check() -> str | None:
    """Return an error string if entailment checking is on but unusable."""
    if not ENABLED:
        return None
    try:
        _model()
    except Exception as e:
        return (f"entailment model '{MODEL}' unavailable: {e}\n"
                f"  set DPDP_HF_ONLINE=1 to download it, or DPDP_ENTAILMENT=0 to disable")
    return None


def split_claims(answer: str) -> list[str]:
    """Split an answer into checkable claim sentences.

    Deliberately drops the answer format's own section headers and anything
    too short to be a claim — the goal is to check what the answer *asserts
    about the law*, not to grade its formatting."""
    claims = []
    for raw in re.split(r"(?<=[.!?])\s+|\n+", answer):
        line = RE_SECTION_HEADER.sub("", raw).strip()
        if len(line) >= MIN_CLAIM_CHARS:
            claims.append(line)
    return claims


def entailment_scores(premise: str, claims: list[str]) -> list[float]:
    """P(entailment) for each claim given the premise, as a plain float.

    The model emits three logits per pair — contradiction / entailment /
    neutral — so this softmaxes and returns the entailment column. Label
    order is read from the model's own config rather than hard-coded: it is
    NOT alphabetical and NOT consistent across NLI checkpoints, and guessing
    it wrong silently inverts the whole check.
    """
    if not claims:
        return []
    import numpy as np

    model = _model()
    logits = np.asarray(model.predict([(premise, c) for c in claims]))
    if logits.ndim == 1:                      # single pair -> 1-D
        logits = logits.reshape(1, -1)

    label_of = {i: str(l).lower() for i, l in model.model.config.id2label.items()}
    entail_idx = next((i for i, l in label_of.items() if "entail" in l), None)
    if entail_idx is None:                    # not a 3-way NLI head
        raise RuntimeError(f"model {MODEL} has no entailment label: {label_of}")

    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp / exp.sum(axis=1, keepdims=True)
    return [float(p) for p in probs[:, entail_idx]]


def verify(answer: str, citation_texts: list[str]) -> list[dict]:
    """Check each claim in `answer` against the verbatim provisions cited.

    A claim is supported if ANY cited provision entails it — an answer that
    correctly draws on three provisions should not be penalised because
    claim 2 is grounded in provision 3 rather than provision 1. Returns one
    record per claim; callers decide how to present them.
    """
    claims = split_claims(answer)
    premises = [t for t in citation_texts if t and t.strip()]
    if not claims or not premises:
        return []

    best = [0.0] * len(claims)
    for premise in premises:
        for i, score in enumerate(entailment_scores(premise, claims)):
            best[i] = max(best[i], score)

    return [{"claim": c, "entailment": round(s, 3),
             "status": "supported" if s >= MIN_ENTAILMENT else "unsupported"}
            for c, s in zip(claims, best)]
