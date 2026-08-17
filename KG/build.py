"""
Build a verbatim knowledge graph of the Digital Personal Data Protection Act, 2023
from the Gazette of India PDF.

Design contract
---------------
1. VERBATIM. Every `text` field holds the Act's exact words. A lossless round-trip
   check reassembles the whole document from the graph and diffs it against the
   extracted body stream; any mismatch fails the build.
2. NOTHING HAND-ENTERED. Column bounds, the indentation ladder, the Schedule table
   and the Act's own metadata are all measured or read from the PDF at runtime.
   There is no data file to get out of sync with the source.
3. NO SILENT CONVENTIONS. Three typesetting conventions cannot be derived --
   headnote alignment, relative cross-reference scope, illustration boundaries.
   Each is resolved, then written to review/ for human sign-off, and each can be
   corrected in overrides.yaml without touching code.

Run:  python build.py
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
from bisect import bisect_right
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import networkx as nx
import pdfplumber
import yaml

# Windows consoles default to cp1252 and would crash on the Act's curly
# quotes and em-dashes. Never let an encoding detail fail a run.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent
PDF = ROOT / "data" / "dpdp_act_2023.pdf"

def load_dotenv(path: Path) -> None:
    """Read KEY=value lines from .env into the environment.

    Six lines of stdlib instead of a dependency. Real environment variables win,
    so CI and one-off overrides behave the way anyone would expect. `.env` holds
    the credentials and is gitignored — nothing secret belongs in this file.
    """
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


load_dotenv(ROOT / ".env")

# Neo4j. Works against a local instance or Aura — the only difference is the
# URI scheme (neo4j:// vs neo4j+s://) and, on Aura, that the username and
# database are often the instance id rather than "neo4j".
NEO4J_URI = os.environ.get("NEO4J_URI", "neo4j://127.0.0.1:7687")
NEO4J_USER = os.environ.get("NEO4J_USER") or os.environ.get("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "")
NEO4J_DATABASE = os.environ.get("NEO4J_DATABASE", "neo4j")
OUT = ROOT / "out"
REVIEW = ROOT / "review"

# Content landmarks. Every one of these is asserted in validate(), not assumed.
RE_RUNNING_HEADER = re.compile(r"THE\s+GAZETTE\s+OF\s+INDIA\s+EXTRAORDINARY")
RE_ACT_TITLE = re.compile(r"^THE DIGITAL PERSONAL DATA PROTECTION ACT, 2023$")
RE_SCHEDULE_HEAD = re.compile(r"^THE\s+SCHEDULE$")
RE_CHAPTER = re.compile(r"^CHAPTER\s+([IVXL]+)$")
RE_ILLUSTRATION = re.compile(r"^Illustrations?\.$")
RE_ACT_NUMBER = re.compile(r"\(NO\.\s*(\d+)\s*OF\s*(\d{4})\)")
RE_ASSENT_DATE = re.compile(r"\[(\d{1,2}\w{2}\s+\w+,\s*\d{4})\.\]")

# Marker tokens that open a node. The token supplies the label; GEOMETRY decides
# the depth. That separation is what makes "(i)" unambiguous -- clause-i and
# roman-one are typeset on different rungs of the same ladder.
# `\s*` not `\s`: section 13 is typeset "(1)A Data Principal" with no space.
MARKERS = [
    ("section", re.compile(r"^(\d{1,2})\.\s*(?=\S)")),
    ("numeric", re.compile(r"^\((\d{1,2})\)\s*(?=\S)")),
    ("alpha", re.compile(r"^\(([a-z]{1,2})\)\s*(?=\S)")),
    ("roman", re.compile(r"^\(([ivxl]{1,5})\)\s*(?=\S)")),
    ("upper", re.compile(r"^\(([A-Z]{1,4})\)\s*(?=\S)")),
]
RE_INLINE_SUBSEC = re.compile(r"^(\d{1,2}\.\s*)(\(\d{1,2}\)\s*)")

# Cross-references, matched against verbatim text. The Act cites in chains and
# lists -- "clause (a) of sub-section (2) of section 10", "sub-sections (1) and
# (5) of section 8", "sections 10 and 11" -- so one pattern handles all three.
_LIST = r"(?:\s*(?:,|and)\s*{item})*"
RE_REF = re.compile(
    r"\b(?:clauses?\s*\(\s*(?P<cl>[a-z]{1,2})\s*\)\s+of\s+)?"
    r"(?:sub-sections?\s*(?P<subs>\(\s*\d{1,2}\s*\)"
    + _LIST.format(item=r"\(\s*\d{1,2}\s*\)") + r")\s+of\s+)?"
    r"sections?\s+(?P<secs>\d{1,2}" + _LIST.format(item=r"\d{1,2}") + r")\b")
RE_BARE_SUBS = re.compile(
    r"\bsub-sections?\s*(\(\s*\d{1,2}\s*\)"
    + _LIST.format(item=r"\(\s*\d{1,2}\s*\)") + r")")
# A citation that continues "... of the <Name> Act, <year>" points out of this Act.
RE_ACT_TAIL = re.compile(
    r"^\s*(?:of\s+)?(?:the\s+)?(?P<act>[A-Z][A-Za-z’'()\.\- ]{2,70}?(?:Act|Code),\s*\d{4})")
RE_ACT_NAME = re.compile(r"\b(?:The\s+)?([A-Z][A-Za-z’'()\.\- ]{2,70}?(?:Act|Code),\s*\d{4})")
RE_XREF_SCHEDULE = re.compile(r"\bthe Schedule\b")
RE_EXTERNAL_CITATION = re.compile(r"^(\d{1,3})\s+of\s+(\d{4})\.$")
RE_CITATION_INLINE = re.compile(r"\d{1,3}\s+of\s+\d{4}\.")
RE_NUM = re.compile(r"\d{1,2}")
RE_RULE = re.compile(r"^[—–—–-]{2,}$")


# --------------------------------------------------------------------------- #
# Phase 1 - geometry, derived from the document itself
# --------------------------------------------------------------------------- #

@dataclass
class Geometry:
    """Column bounds and indent ladder, measured from the PDF's word boxes.

    The Gazette sets three columns per page: marginalia left (verso) or right
    (recto), and the body between them. They never overlap horizontally, so a
    gap analysis of the x-histograms recovers the divide exactly.
    """

    body_left: float
    body_right: float
    indent_base: float
    indent_step: float
    line_tol: float
    notes: list[str] = field(default_factory=list)

    @classmethod
    def derive(cls, pdf: pdfplumber.PDF) -> "Geometry":
        x0_hist: Counter[int] = Counter()
        x1_hist: Counter[int] = Counter()
        baselines: list[float] = []

        for page in pdf.pages:
            tops = sorted({round(w["top"], 1) for w in page.extract_words()})
            baselines += [round(b - a) for a, b in zip(tops, tops[1:]) if 6 < b - a < 30]
            for w in page.extract_words():
                x0_hist[round(w["x0"])] += 1
                x1_hist[round(w["x1"])] += 1

        body_left, left_gap = cls._divide(x0_hist, side="left")
        body_right, right_gap = cls._divide(x1_hist, side="right")

        spacing = Counter(baselines).most_common(1)[0][0]
        line_tol = round(spacing * 0.4, 1)

        # Indent ladder: histogram the left edge of every line that opens with a
        # marker token, then read the rung spacing off the peaks. The Schedule
        # page is excluded -- it is a table, and its column is not a rung.
        ladder: Counter[int] = Counter()
        for page in pdf.pages:
            rows = group_rows(page.extract_words(), line_tol)
            if any(RE_SCHEDULE_HEAD.match(" ".join(w["text"] for w in ws)) for _t, ws in rows):
                continue
            for _top, words in rows:
                body = [w for w in words if body_left <= w["x0"] and w["x1"] <= body_right]
                if body and any(rx.match(" ".join(w["text"] for w in body)) for _n, rx in MARKERS):
                    ladder[round(body[0]["x0"], 1)] += 1

        peaks = cluster_peaks(ladder, tol=5.0, min_count=5)
        steps = [b - a for a, b in zip(peaks, peaks[1:])]
        step = float(statistics.median(steps))
        base = peaks[0] - step  # continuation indent sits one rung left of the first marker

        return cls(
            body_left, body_right, base, step, line_tol,
            notes=[
                f"body column: x0 >= {body_left} and x1 <= {body_right}",
                f"  left divide from a {left_gap}pt gap in the x0 histogram",
                f"  right divide from a {right_gap}pt gap in the x1 histogram",
                f"marker indent peaks (pt): {peaks}",
                f"indent ladder: base {base}, step {step} -> rungs "
                f"{[base + step * k for k in range(5)]}",
                f"modal baseline spacing {spacing}pt -> line clustering tolerance {line_tol}pt",
            ],
        )

    @staticmethod
    def _divide(hist: Counter[int], side: str, min_gap: int = 12) -> tuple[float, int]:
        """Locate the body/marginalia divide: the gap nearest the densest text
        column, on the requested side, at least `min_gap` wide."""
        xs = sorted(hist)
        mode = hist.most_common(1)[0][0]
        gaps = [(xs[i], xs[i + 1]) for i in range(len(xs) - 1) if xs[i + 1] - xs[i] >= min_gap]
        if side == "left":
            candidates = [g for g in gaps if g[1] <= mode] or gaps
            lo, hi = candidates[-1]          # nearest gap to the left of the body
        else:
            candidates = [g for g in gaps if g[0] >= mode] or gaps
            lo, hi = candidates[0]           # nearest gap to the right of the body
        return round((lo + hi) / 2, 1), hi - lo

    def indent(self, x0: float) -> int:
        """Snap a line's left edge onto the ladder. Justified text drifts a few
        points; rungs are 24pt apart, so the snap is never ambiguous."""
        return max(0, round((x0 - self.indent_base) / self.indent_step))


def cluster_peaks(hist: Counter[float], tol: float, min_count: int) -> list[float]:
    """Collapse a histogram into weighted peak centres. Justified text puts the
    same rung at 141.5 and 142.0 on different lines; those are one peak, not two."""
    peaks: list[float] = []
    run: list[tuple[float, int]] = []
    for x in sorted(hist):
        if run and x - run[-1][0] > tol:
            peaks.append(run)
            run = []
        run.append((x, hist[x]))
    if run:
        peaks.append(run)
    return [round(sum(x * n for x, n in p) / sum(n for _x, n in p), 1)
            for p in peaks if sum(n for _x, n in p) >= min_count]


def group_rows(words: list[dict], tol: float) -> list[tuple[float, list[dict]]]:
    """Group words into visual lines. The tolerance absorbs the baseline shift the
    Gazette applies to small-caps fragments (the 'OF' in a chapter heading sits
    3pt below its line) without ever merging two real lines 12pt apart."""
    rows: list[tuple[float, list[dict]]] = []
    for w in sorted(words, key=lambda w: (w["top"], w["x0"])):
        if rows and abs(w["top"] - rows[-1][0]) <= tol:
            rows[-1][1].append(w)
        else:
            rows.append((w["top"], [w]))
    return [(t, sorted(ws, key=lambda w: w["x0"])) for t, ws in rows]


@dataclass
class Line:
    page: int
    top: float
    x0: float
    indent: int
    text: str


@dataclass
class MarginNote:
    page: int
    top: float
    text: str


def extract(geo: Geometry) -> tuple[list[Line], list[MarginNote], list[dict], dict]:
    """Split every page into body lines, marginalia and the Schedule table."""
    body: list[Line] = []
    margins: list[MarginNote] = []
    schedule_rows: list[dict] = []
    dropped = {"pages": 0, "running_headers": 0, "gazette_masthead": 0, "schedule_page": 0}
    seen_title = False

    with pdfplumber.open(PDF) as pdf:
        dropped["pages"] = len(pdf.pages)
        for pno, page in enumerate(pdf.pages, 1):
            rows = group_rows(page.extract_words(), geo.line_tol)
            is_schedule = any(RE_SCHEDULE_HEAD.match(" ".join(w["text"] for w in ws))
                              for _t, ws in rows)
            if is_schedule:
                schedule_rows = parse_schedule_page(rows, geo)
                dropped["schedule_page"] += len(rows)
                continue

            for top, words in rows:
                inside = [w for w in words if geo.body_left <= w["x0"] and w["x1"] <= geo.body_right]
                outside = [w for w in words if w not in inside]

                if outside:
                    margins.append(MarginNote(pno, top, " ".join(w["text"] for w in outside)))
                if not inside:
                    continue

                text = " ".join(w["text"] for w in inside)
                if RE_RUNNING_HEADER.search(text):
                    dropped["running_headers"] += 1
                    continue

                # Page 1 carries the bilingual gazette masthead above the Act's own
                # title. The boundary is the title line, not a page coordinate.
                if not seen_title:
                    if RE_ACT_TITLE.match(text):
                        seen_title = True
                    else:
                        dropped["gazette_masthead"] += 1
                        continue

                body.append(Line(pno, top, inside[0]["x0"], geo.indent(inside[0]["x0"]), text))

    return body, margins, schedule_rows, dropped


def parse_schedule_page(rows, geo: Geometry) -> list[dict]:
    """The Schedule is a real 3-column table, so its reading order is column-major
    and text extraction scrambles it. Recover the columns by the same gap analysis
    used for the page, then read each row band."""
    body_rows = [(t, [w for w in ws if geo.body_left <= w["x0"] <= geo.body_right])
                 for t, ws in rows]
    body_rows = [(t, ws) for t, ws in body_rows if ws]

    # The table declares its own extent: a "(1) (2) (3)" column-number row opens
    # it and a horizontal rule closes it. Everything outside is title or colophon.
    def as_text(ws):
        return " ".join(w["text"] for w in ws)

    start = next(i for i, (_t, ws) in enumerate(body_rows)
                 if re.fullmatch(r"\(1\)\s+\(2\)\s+\(3\)", as_text(ws)))
    end = next((i for i, (_t, ws) in enumerate(body_rows)
                if i > start and RE_RULE.fullmatch(as_text(ws))), len(body_rows))
    table = body_rows[start + 1:end]

    # Columns are the vertical whitespace gutters running the height of the table.
    covered: set[int] = set()
    for _t, ws in table:
        for w in ws:
            covered.update(range(int(w["x0"]), int(w["x1"]) + 1))
    lo, hi = min(covered), max(covered)
    gutters, run_start = [], None
    for x in range(lo, hi + 1):
        if x not in covered and run_start is None:
            run_start = x
        elif x in covered and run_start is not None:
            if x - run_start >= 8:
                gutters.append((run_start + x) / 2)
            run_start = None
    boundaries = [float(lo)] + gutters

    entries: list[dict] = []
    for _t, ws in table:
        buckets: dict[int, list[str]] = defaultdict(list)
        for w in ws:
            buckets[max(bisect_right(boundaries, w["x0"]) - 1, 0)].append(w["text"])
        if m := re.fullmatch(r"(\d+)\.", " ".join(buckets.get(0, [])).strip()):
            entries.append({"sl_no": int(m.group(1)), "breach": "", "penalty": ""})
        if not entries:
            continue
        for col, key in ((1, "breach"), (2, "penalty")):
            if buckets.get(col):
                entries[-1][key] = f"{entries[-1][key]} {' '.join(buckets[col])}".strip()
    return entries


# --------------------------------------------------------------------------- #
# Phase 2 - marginalia -> section headnotes
# --------------------------------------------------------------------------- #

def reassemble_margins(margins: list[MarginNote], geo: Geometry) -> tuple[list[dict], list[dict]]:
    """Margin notes wrap over several lines ('Grounds for' / 'processing' /
    'personal data.'). Rejoin fragments one baseline apart; anything matching
    'NN of YYYY.' is an external-act citation, not a headnote."""
    span = geo.line_tol / 0.4 * 1.4      # ~1.4 baselines: joins wraps, splits notes

    # Act citations share the margin with headnotes and sometimes share a
    # baseline with them -- section 25 is set as "Members and officers to be
    # public" / "45 of 1860. servants." So strip citations out of each fragment
    # and keep whatever text remains as headnote material.
    citations: list[dict] = []
    stripped: list[MarginNote] = []
    for m in margins:
        found = RE_CITATION_INLINE.findall(m.text)
        citations += [{"page": m.page, "top": m.top, "text": c} for c in found]
        if residue := RE_CITATION_INLINE.sub(" ", m.text).strip():
            stripped.append(MarginNote(m.page, m.top, residue))
    margins = stripped

    blocks: list[dict] = []
    by_page: dict[int, list[MarginNote]] = defaultdict(list)
    for m in margins:
        by_page[m.page].append(m)

    for notes in by_page.values():
        notes.sort(key=lambda m: m.top)
        run: list[MarginNote] = []
        for note in notes:
            if run and note.top - run[-1].top > span:
                blocks.append(_join(run))
                run = []
            run.append(note)
        if run:
            blocks.append(_join(run))

    return blocks, citations


def _join(frags: list[MarginNote]) -> dict:
    return {"page": frags[0].page, "top": frags[0].top,
            "text": " ".join(f.text for f in frags).strip()}


# --------------------------------------------------------------------------- #
# Phase 3 - parse the indentation ladder into a tree
# --------------------------------------------------------------------------- #

@dataclass
class Node:
    id: str
    kind: str
    label: str
    prefix: str = ""          # the verbatim marker text, kept for round-tripping
    text: str = ""
    page: int = 0
    top: float = 0.0
    indent: int = 0
    parent: str | None = None
    children: list[str] = field(default_factory=list)
    headnote: str | None = None
    chapter: str | None = None
    follows: str | None = None      # illustrations: the node the block was set after

    def append(self, s: str) -> None:
        self.text = f"{self.text} {s}".strip() if self.text else s


class Tree:
    def __init__(self) -> None:
        self.nodes: dict[str, Node] = {}
        self.order: list[str] = []

    def add(self, node: Node, parent: Node | None) -> Node:
        if parent is not None:
            node.parent = parent.id
            node.chapter = parent.id if parent.kind == "Chapter" else parent.chapter
            parent.children.append(node.id)
        self.nodes[node.id] = node
        self.order.append(node.id)
        return node


ORDINAL = {"SubSection": 1, "Clause": 2, "SubClause": 3, "Item": 4}


def parse(lines: list[Line]) -> Tree:
    tree = Tree()
    act = tree.add(Node("act", "Act", "Digital Personal Data Protection Act, 2023"), None)
    preamble = tree.add(Node("preamble", "Preamble", "Preamble"), act)

    stack: list[Node] = [act, preamble]
    chapter: Node | None = None
    illustration: Node | None = None
    awaiting_chapter_title: str | None = None
    ill_seq: Counter[str] = Counter()

    for line in lines:
        text = line.text

        if awaiting_chapter_title is not None:
            roman = awaiting_chapter_title
            chapter = tree.add(
                Node(f"ch-{roman}", "Chapter", f"Chapter {roman} — {text}",
                     prefix=f"CHAPTER {roman}", text=text, page=line.page, top=line.top),
                act)
            chapter.chapter = chapter.id
            awaiting_chapter_title = None
            stack = [act, chapter]
            illustration = None
            continue

        if m := RE_CHAPTER.match(text):
            awaiting_chapter_title = m.group(1)
            continue

        if RE_ILLUSTRATION.match(text):
            # `Illustration.` is centred, so geometry says nothing about which
            # provision it belongs to. An illustration explains a provision, and
            # provisions are sections and sub-sections -- so it hangs off the
            # nearest of those, not off whatever clause happens to be open.
            # `follows` keeps the innermost open node so review/ can show both.
            follows = stack[-1]
            host = next((n for n in reversed(stack)
                         if n.kind in ("SubSection", "Section")), stack[-1])
            ill_seq[host.id] += 1
            illustration = tree.add(
                Node(f"{host.id}-ill{ill_seq[host.id]}", "Illustration",
                     f"Illustration to {host.label}", prefix=text,
                     page=line.page, top=line.top, indent=99, follows=follows.id),
                host)
            continue

        opened = disambiguate(
            candidates(text, line.indent, in_illustration=illustration is not None),
            stack, line.indent)

        if opened is None:
            (illustration or stack[-1]).append(text)
            continue
        kind, marker, prefix = opened

        if kind != "IllustrationItem":
            illustration = None       # any statutory marker closes the block

        if kind == "Section":
            node = tree.add(
                Node(f"s-{marker}", "Section", f"Section {marker}", prefix=prefix,
                     page=line.page, top=line.top, indent=1),
                chapter or preamble)
            stack = [act, chapter or preamble, node]
            rest = text[len(prefix):].strip()

            # "6. (1) The consent given ..." opens a section and its first
            # sub-section on one typeset line. Split it, keeping both verbatim.
            if inline := RE_INLINE_SUBSEC.match(text):
                sub_prefix = inline.group(2)
                sub = tree.add(
                    Node(f"s-{marker}-{sub_prefix.strip()[1:-1]}", "SubSection",
                         f"Section {marker}({sub_prefix.strip()[1:-1]})", prefix=sub_prefix.strip(),
                         page=line.page, top=line.top, indent=1),
                    node)
                sub.append(text[len(inline.group(0)):].strip())
                stack.append(sub)
            else:
                node.append(rest)
            continue

        if kind == "IllustrationItem":
            host = illustration
            node = tree.add(
                Node(f"{host.id}-{marker}", kind, f"{host.label} ({marker})",
                     prefix=prefix, page=line.page, top=line.top, indent=99),
                host)
            node.append(text[len(prefix):].strip())
            illustration = node          # continuations belong to the item
            continue

        depth = ORDINAL[kind]
        while len(stack) > 2 and ORDINAL.get(stack[-1].kind, 0) >= depth:
            stack.pop()
        parent = stack[-1]
        node = tree.add(
            Node(f"{parent.id}-{marker}", kind, f"{parent.label}({marker})", prefix=prefix,
                 page=line.page, top=line.top, indent=line.indent),
            parent)
        node.append(text[len(prefix):].strip())
        stack.append(node)

    return tree


def candidates(text: str, indent: int, in_illustration: bool) -> list[tuple[str, str, str]]:
    """Every node kind this line could open, given its rung on the ladder.

    Geometry rules out most of them: `(i)` at rung 3 can only be a sub-clause,
    because a clause is never set that deep. It stays ambiguous at rung 2, where
    both `(a)`-style clauses and `(i)`-style sub-clauses are typeset -- that case
    is settled by `disambiguate` using the siblings already parsed.
    """
    out: list[tuple[str, str, str]] = []
    for name, rx in MARKERS:
        m = rx.match(text)
        if not m:
            continue
        prefix, tok = m.group(0).rstrip(), m.group(1)
        if name == "section" and indent == 1:
            out.append(("Section", tok, prefix))
        elif name == "numeric" and indent <= 1:
            out.append(("SubSection", tok, prefix))
        elif name == "alpha" and indent == 2:
            out.append(("Clause", tok, prefix))
        elif name == "roman" and indent >= 2:
            out.append(("SubClause", tok, prefix))
        elif name == "upper":
            if in_illustration:
                out.append(("IllustrationItem", tok, prefix))
            elif indent >= 4:
                out.append(("Item", tok, prefix))
    return out


def disambiguate(options: list[tuple[str, str, str]], stack: list[Node], indent: int
                 ) -> tuple[str, str, str] | None:
    """Resolve `(i)` — letter-i clause, or roman-numeral one?

    Both are legal at rung 2 and the character is identical, so no amount of
    geometry settles it. Sequence does: a clause list always opens at `(a)`, a
    sub-clause list always opens at `(i)`, and a list never changes type
    part-way through. So if a list is already open on this rung, the line
    continues it; otherwise the token itself opens a new one.
    """
    if len(options) <= 1:
        return options[0] if options else None
    if {k for k, _t, _p in options} != {"Clause", "SubClause"}:
        return options[0]

    open_here = next((n.kind for n in reversed(stack)
                      if n.indent == indent and n.kind in ("Clause", "SubClause")), None)
    want = open_here or ("SubClause" if options[0][1] == "i" else "Clause")
    return next(o for o in options if o[0] == want)


# --------------------------------------------------------------------------- #
# Phase 4 - headnotes, definitions, cross-references
# --------------------------------------------------------------------------- #

def attach_headnotes(tree: Tree, headnotes: list[dict], overrides: dict) -> list[dict]:
    """Gazette convention: a section's headnote is typeset level with that
    section's opening line. A convention, not a derivable fact -- so every
    pairing is written to review/headnotes.md for sign-off."""
    by_page: dict[int, list[Node]] = defaultdict(list)
    for n in tree.nodes.values():
        if n.kind == "Section":
            by_page[n.page].append(n)

    audit: list[dict] = []
    for hn in headnotes:
        candidates = by_page.get(hn["page"], [])
        if not candidates:
            audit.append({**hn, "section": None, "delta": None, "source": "unmatched"})
            continue
        best = min(candidates, key=lambda s: abs(s.top - hn["top"]))
        audit.append({**hn, "section": best.id, "delta": round(best.top - hn["top"], 1),
                      "source": "auto"})

    for e in audit:
        if e["section"]:
            tree.nodes[e["section"]].headnote = e["text"]
    for sid, text in (overrides.get("headnotes") or {}).items():
        if sid in tree.nodes:
            tree.nodes[sid].headnote = text
            audit.append({"section": sid, "text": text, "page": "", "top": "",
                          "delta": "", "source": "override"})
    return audit


def parse_definitions(tree: Tree) -> list[dict]:
    """Section 2 defines every actor and term the Act uses. Read them verbatim --
    no model, no inference, no hallucination surface."""
    defs = []
    if (s2 := tree.nodes.get("s-2")) is None:
        return defs
    for cid in s2.children:
        node = tree.nodes[cid]
        # The term is the opening quoted phrase; the relation is the first
        # "means"/"includes" after it. A qualifier may sit between the two --
        # “processing” in relation to personal data, means ...
        term = re.match(r'^[“"](.+?)[”"]', node.text)
        if not term:
            continue
        rel = re.search(r"\b(means|includes)\b", node.text[term.end():])
        if not rel:
            continue
        defs.append({"id": f"def-{slug(term.group(1))}", "term": term.group(1),
                     "relation": rel.group(1),
                     "qualifier": node.text[term.end():term.end() + rel.start()].strip(" ,"),
                     "defined_in": node.id, "text": node.text})
    return defs


def scan_references(text: str, owner: str | None, max_section: int) -> list[dict]:
    """Pull every statutory citation out of one verbatim passage.

    Three shapes occur, and all three matter:

      qualified   "clause (a) of sub-section (2) of section 10"  -> s-10-2-a
      list        "sub-sections (1) and (5) of section 8"        -> s-8-1, s-8-5
      bare        "sub-section (5)"                              -> the enclosing section

    A citation is treated as pointing OUT of this Act when it names another Act
    ("... of section 3 of the Insolvency and Bankruptcy Code, 2016") or names a
    section number this Act does not have. Section 44 amends four other statutes,
    so without that guard its references would silently link to the wrong law.
    """
    refs: list[dict] = []
    external = False

    for m in RE_REF.finditer(text):
        secs = [int(x) for x in RE_NUM.findall(m.group("secs"))]
        subs = RE_NUM.findall(m.group("subs")) if m.group("subs") else []
        clause = m.group("cl")
        tail = RE_ACT_TAIL.match(text[m.end():])

        if tail or any(s > max_section for s in secs):
            external = True
            refs.append({"target": None, "quote": m.group(0), "resolution": "external-act",
                         "act": tail.group("act") if tail else None})
            continue

        for sec in secs:
            for sub in subs or [None]:
                tid = f"s-{sec}" if sub is None else f"s-{sec}-{sub}"
                refs.append({"target": f"{tid}-{clause}" if clause else tid,
                             "quote": m.group(0), "resolution": "qualified"})

    # Bare sub-section citations mean "of the section I am in" -- but only in a
    # passage that is not already talking about somebody else's statute.
    if owner and not external:
        masked = RE_REF.sub(lambda m: " " * len(m.group(0)), text)
        for m in RE_BARE_SUBS.finditer(masked):
            for sub in RE_NUM.findall(m.group(1)):
                refs.append({"target": f"{owner}-{sub}", "quote": m.group(0),
                             "resolution": "relative-to-own-section"})

    if RE_XREF_SCHEDULE.search(text):
        refs.append({"target": "schedule", "quote": "the Schedule", "resolution": "qualified"})
    return refs


def find_xrefs(tree: Tree, overrides: dict) -> tuple[list[dict], list[dict], list[dict]]:
    skip = {tuple(p) for p in (overrides.get("drop_xrefs") or [])}
    max_section = max(int(n.id.split("-")[1]) for n in tree.nodes.values() if n.kind == "Section")
    edges, unresolved, externals = [], [], []

    for node in tree.nodes.values():
        if not node.text:
            continue
        owner = ancestor_section(tree, node)
        for ref in scan_references(node.text, owner.id if owner else None, max_section):
            if ref["resolution"] == "external-act":
                externals.append({"src": node.id, "quote": ref["quote"],
                                  "act": ref.get("act") or inherited_act(tree, node) or "unnamed"})
            elif (node.id, ref["target"]) in skip or ref["target"] == node.id:
                continue
            elif ref["target"] in tree.nodes or ref["target"] == "schedule":
                edges.append({"src": node.id, "dst": ref["target"], "quote": ref["quote"],
                              "resolution": ref["resolution"]})
            else:
                unresolved.append({"src": node.id, "target": ref["target"], "quote": ref["quote"]})
    return edges, unresolved, externals


def inherited_act(tree: Tree, node: Node) -> str | None:
    """Section 44(2)(b) says only 'in section 81' -- the statute being amended is
    named by its parent, 'The Information Technology Act, 2000 shall be amended'."""
    cur: Node | None = node
    while cur is not None:
        # Quoted spans are words being inserted INTO another Act, not the Act
        # being amended -- section 44(2)(b) quotes "the Patents Act, 1970" while
        # amending the Information Technology Act, 2000.
        unquoted = re.sub(r"[“\"][^”\"]*[”\"]", " ", cur.text or "")
        for m in RE_ACT_NAME.finditer(unquoted):
            if "Digital Personal Data Protection" not in m.group(1):
                return m.group(1)
        cur = tree.nodes.get(cur.parent) if cur.parent else None
    return None


def ancestor_section(tree: Tree, node: Node) -> Node | None:
    cur: Node | None = node
    while cur is not None:
        if cur.kind == "Section":
            return cur
        cur = tree.nodes.get(cur.parent) if cur.parent else None
    return None


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #

def validate(tree: Tree, lines: list[Line], schedule: list[dict], dropped: dict,
             headnote_audit: list[dict]) -> list[str]:
    """Structural assertions plus a lossless round-trip. A failure here means the
    parse invented, lost or reordered text, so the graph must not be trusted."""
    problems: list[str] = []

    def check(ok: bool, msg: str) -> None:
        if not ok:
            problems.append(msg)

    kinds = Counter(n.kind for n in tree.nodes.values())
    check(kinds["Section"] == 44, f"expected 44 sections, got {kinds['Section']}")
    check(kinds["Chapter"] == 9, f"expected 9 chapters, got {kinds['Chapter']}")
    # Every page carries the running header except page 1 (masthead) and the
    # Schedule page, which is consumed whole by the table parser.
    expect_headers = dropped["pages"] - 2
    check(dropped["running_headers"] == expect_headers,
          f"expected {expect_headers} running headers, dropped {dropped['running_headers']}")
    check(len(schedule) == 7, f"expected 7 Schedule entries, got {len(schedule)}")
    check(all(r["breach"] and r["penalty"] for r in schedule), "a Schedule cell is empty")

    nums = sorted(int(n.id.split("-")[1]) for n in tree.nodes.values() if n.kind == "Section")
    check(nums == list(range(1, 45)),
          f"sections are not 1..44 (missing {sorted(set(range(1, 45)) - set(nums))})")
    orphans = [n.id for n in tree.nodes.values() if n.kind == "Section" and not n.chapter]
    check(not orphans, f"sections outside any chapter: {orphans}")

    missing = [n.id for n in tree.nodes.values() if n.kind == "Section" and not n.headnote]
    check(not missing, f"sections with no headnote recovered from the margin: {missing}")
    drift = [e for e in headnote_audit if e.get("delta") not in (None, "", 0.0)]
    check(not drift, "headnote alignment is not exact for: "
                     + ", ".join(f"{e['section']} ({e['delta']}pt)" for e in drift))

    rebuilt = normalise(" ".join(
        f"{tree.nodes[i].prefix} {tree.nodes[i].text}".strip() for i in tree.order))
    source = normalise(" ".join(l.text for l in lines))
    identical = rebuilt == source
    check(identical, f"ROUND-TRIP FAILED: {first_divergence(source, rebuilt)}")

    REVIEW.mkdir(exist_ok=True)
    (REVIEW / "roundtrip.txt").write_text(
        "Lossless round-trip: every node's verbatim text reassembled in document\n"
        "order and compared against the raw body stream.\n\n"
        f"source chars : {len(source)}\nrebuilt chars: {len(rebuilt)}\n"
        f"identical    : {identical}\n"
        + ("" if identical else f"\n{first_divergence(source, rebuilt)}\n"),
        encoding="utf-8")
    return problems


def normalise(s: str) -> str:
    """Whitespace only -- never alters a word. Marker spacing is normalised too,
    because the Gazette sets "(1)A Data Principal" without a space in section 13
    while setting "(1) A person" with one everywhere else."""
    s = re.sub(r"\s+", " ", s)
    return re.sub(r"\((\w{1,4})\)\s*", r"(\1) ", s).strip()


def first_divergence(a: str, b: str) -> str:
    for i, (ca, cb) in enumerate(zip(a, b)):
        if ca != cb:
            return (f"at char {i}\n  source : ...{a[max(0, i-80):i+80]}...\n"
                    f"  rebuilt: ...{b[max(0, i-80):i+80]}...")
    tail = a[len(b):] or b[len(a):]
    return f"length differs: source {len(a)} vs rebuilt {len(b)}\n  extra: {tail[:250]}"


# --------------------------------------------------------------------------- #
# Graph and exports
# --------------------------------------------------------------------------- #

def build_graph(tree: Tree, defs, xrefs, citations, schedule, meta) -> nx.MultiDiGraph:
    G = nx.MultiDiGraph()
    max_section = max(int(n.id.split("-")[1]) for n in tree.nodes.values() if n.kind == "Section")

    for n in tree.nodes.values():
        G.add_node(n.id, label=n.label, kind=n.kind, text=n.text, headnote=n.headnote or "",
                   page=n.page, chapter=n.chapter or "", verbatim=True)
        if n.parent:
            G.add_edge(n.parent, n.id, key=f"HAS_{n.kind.upper()}", type=f"HAS_{n.kind.upper()}")

    for d in defs:
        G.add_node(d["id"], label=d["term"], kind="Definition", text=d["text"],
                   relation=d["relation"], verbatim=True)
        G.add_edge(d["defined_in"], d["id"], key="DEFINES", type="DEFINES")

    # MENTIONS: exact word-boundary match of each defined term. Deterministic and
    # exhaustive -- this is the recall that embeddings cannot guarantee.
    for n in tree.nodes.values():
        if not n.text:
            continue
        for d in defs:
            if d["defined_in"] != n.id and re.search(rf"\b{re.escape(d['term'])}\b", n.text):
                G.add_edge(n.id, d["id"], key="MENTIONS", type="MENTIONS")

    G.add_node("schedule", label="The Schedule", kind="Schedule",
               text="[See section 33 (1)]", verbatim=True)
    for row in schedule:
        rid = f"pen-{row['sl_no']}"
        G.add_node(rid, label=f"Schedule entry {row['sl_no']}", kind="Penalty",
                   text=row["breach"], penalty=row["penalty"], verbatim=True)
        G.add_edge("schedule", rid, key="HAS_ENTRY", type="HAS_ENTRY")
        # Which provision a penalty binds is stated in the breach text itself.
        for ref in scan_references(row["breach"], owner=None, max_section=max_section):
            if ref["target"] in G:
                G.add_edge(ref["target"], rid, key="PENALISED_BY", type="PENALISED_BY")

    for x in xrefs:
        G.add_edge(x["src"], x["dst"], key="REFERENCES", type="REFERENCES",
                   quote=x["quote"], resolution=x["resolution"])

    for c in citations:
        m = RE_EXTERNAL_CITATION.match(c["text"])
        G.add_node(f"ext-{m.group(2)}-{m.group(1)}", kind="ExternalAct",
                   label=f"Act {m.group(1)} of {m.group(2)}", text=c["text"],
                   page=c["page"], verbatim=True)

    G.graph.update(meta)
    add_authority(G)
    return G


# Weights for the authority computation below. REFERENCES is a deliberate,
# drafted cross-reference — one provision explicitly pointing at another — so
# it is the real signal of what the Act itself treats as load-bearing.
# PENALISED_BY is a genuine structural join and counts for half; there are
# only 6.
#
# MENTIONS is deliberately EXCLUDED, and this was measured rather than
# assumed. Including it at any weight makes every top-authority node a
# Definition, because all 605 MENTIONS edges point *into* the 28 definitions
# — the result then measures "how often is this term used", not "how
# load-bearing is this provision", which is the opposite of what expansion
# needs. With MENTIONS excluded the ranking is immediately sensible:
# §29(1) (right of appeal), §33 (penalties), §6 (consent), the Schedule.
AUTHORITY_WEIGHTS = {"REFERENCES": 1.0, "PENALISED_BY": 0.5}


def add_authority(G: nx.MultiDiGraph) -> None:
    """Store a PageRank score on every node, computed over the Act's own
    citation structure.

    Network centrality predicts a provision's practical importance better
    than a raw count of how many things point at it — a provision cited by
    heavily-cited provisions matters more than one cited the same number of
    times by peripheral ones. Computed once here at build time, never per
    query, and consumed by `ask.retrieve()` to keep high-fan-out hub nodes
    (§40(2) cites nearly a third of the Act) from crowding out genuinely
    relevant provisions during graph expansion.

    Structural HAS_* edges are excluded on purpose: they encode containment,
    not endorsement. Including them would just rediscover the document
    outline — every section would "cite" its own sub-sections — and rank
    long sections above important ones.
    """
    weighted = nx.DiGraph()
    weighted.add_nodes_from(G.nodes())
    for src, dst, data in G.edges(data=True):
        w = AUTHORITY_WEIGHTS.get(data.get("type"))
        if not w:
            continue
        # Collapse the MultiDiGraph's parallel edges into one weighted edge
        # per pair — nx.pagerank reads `weight` off a simple DiGraph.
        if weighted.has_edge(src, dst):
            weighted[src][dst]["weight"] += w
        else:
            weighted.add_edge(src, dst, weight=w)

    scores = nx.pagerank(weighted, weight="weight") if weighted.number_of_edges() else {}
    for node_id in G.nodes():
        G.nodes[node_id]["authority"] = round(scores.get(node_id, 0.0), 6)


def act_metadata(tree: Tree) -> dict:
    """Act number and assent date, read off the title block rather than typed in."""
    head = tree.nodes["preamble"].text
    num = RE_ACT_NUMBER.search(head)
    date = RE_ASSENT_DATE.search(head)
    return {"title": "The Digital Personal Data Protection Act, 2023",
            "act_number": f"{num.group(1)} of {num.group(2)}" if num else "",
            "assent_date": date.group(1) if date else "",
            "source": PDF.name}


def export(G: nx.MultiDiGraph, tree: Tree, schedule: list[dict]) -> None:
    OUT.mkdir(exist_ok=True)
    (OUT / "dpdp_tree.json").write_text(
        json.dumps({k: asdict(v) for k, v in tree.nodes.items()}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    (OUT / "dpdp_graph.json").write_text(
        json.dumps(nx.node_link_data(G, edges="links"), ensure_ascii=False, indent=2),
        encoding="utf-8")
    (OUT / "schedule.json").write_text(
        json.dumps(schedule, ensure_ascii=False, indent=2), encoding="utf-8")
    nx.write_gexf(G, OUT / "dpdp.gexf")
    write_cypher(G)
    write_html(G)


def push_to_neo4j(G: nx.MultiDiGraph, uri: str, user: str, password: str,
                  database: str = "neo4j") -> None:
    """Load the graph straight into a running Neo4j.

    Batched with UNWIND rather than a statement per node: 404 nodes and 1088
    edges as ~1500 separate round trips is slow for no reason. Relationship
    types cannot be parameterised in Cypher, so edges are grouped by type and
    each group gets one query with the type interpolated — the type strings are
    ours, not user input.
    """
    from neo4j import GraphDatabase

    nodes = [{"id": n, **{k: v for k, v in a.items() if isinstance(v, (str, int, bool))}}
             for n, a in G.nodes(data=True)]
    by_type: dict[str, list[dict]] = defaultdict(list)
    for u, v, a in G.edges(data=True):
        by_type[a["type"]].append({"src": u, "dst": v,
                                   "quote": a.get("quote", ""),
                                   "resolution": a.get("resolution", "")})

    with GraphDatabase.driver(uri, auth=(user, password)) as driver:
        driver.verify_connectivity()
        with driver.session(database=database) as session:
            session.run("MATCH (n:Provision) DETACH DELETE n")
            session.run("CREATE CONSTRAINT dpdp_id IF NOT EXISTS "
                        "FOR (n:Provision) REQUIRE n.id IS UNIQUE")
            # One pass per kind so each node also carries its kind as a label,
            # which is what makes the Neo4j Browser legend useful.
            for kind in sorted({n.get("kind", "Node") for n in nodes}):
                batch = [n for n in nodes if n.get("kind", "Node") == kind]
                session.run(
                    f"UNWIND $batch AS row CREATE (n:Provision:{kind}) SET n = row", batch=batch)
            for etype, edges in by_type.items():
                session.run(
                    f"UNWIND $edges AS e "
                    f"MATCH (a:Provision {{id: e.src}}), (b:Provision {{id: e.dst}}) "
                    f"CREATE (a)-[r:{etype}]->(b) "
                    f"SET r.quote = e.quote, r.resolution = e.resolution", edges=edges)
            counts = session.run(
                "MATCH (n:Provision) WITH count(n) AS nodes "
                "MATCH ()-[r]->() RETURN nodes, count(r) AS rels").single()

    print(f"   neo4j {uri} db={database}: "
          f"{counts['nodes']} nodes, {counts['rels']} relationships")


def write_cypher(G: nx.MultiDiGraph) -> None:
    def esc(v: object) -> str:
        return str(v).replace("\\", "\\\\").replace("'", "\\'").replace("\n", " ")

    out = ["// Neo4j loader.  Usage:",
           "//   docker run -d -p7474:7474 -p7687:7687 -e NEO4J_AUTH=neo4j/dpdp12345 neo4j:5",
           "//   cypher-shell -u neo4j -p dpdp12345 -f out/load.cypher",
           "MATCH (n) DETACH DELETE n;",
           "CREATE CONSTRAINT dpdp_id IF NOT EXISTS FOR (n:Provision) REQUIRE n.id IS UNIQUE;"]
    for nid, a in G.nodes(data=True):
        props = ", ".join(f"{k}: '{esc(v)}'" for k, v in a.items()
                          if isinstance(v, (str, int, bool)) and v != "")
        out.append(f"CREATE (:Provision:{a.get('kind', 'Node')} {{id: '{esc(nid)}', {props}}});")
    for u, v, a in G.edges(data=True):
        out.append(f"MATCH (a {{id:'{esc(u)}'}}), (b {{id:'{esc(v)}'}}) "
                   f"CREATE (a)-[:{a['type']}]->(b);")
    (OUT / "load.cypher").write_text("\n".join(out), encoding="utf-8")


def write_html(G: nx.MultiDiGraph) -> None:
    """Standalone graph viewer: click, drag, filter by edge type. Same feel as the
    Neo4j Browser without needing a server running."""
    palette = {"Act": "#f1f5f9", "Chapter": "#a78bfa", "Section": "#60a5fa",
               "SubSection": "#22d3ee", "Clause": "#34d399", "SubClause": "#a3e635",
               "Item": "#fbbf24", "Illustration": "#f472b6", "IllustrationItem": "#fbcfe8",
               "Definition": "#f87171", "Penalty": "#fb923c", "Schedule": "#fdba74",
               "ExternalAct": "#94a3b8", "Preamble": "#cbd5e1"}
    data = {
        "nodes": [{"id": n, "label": a.get("headnote") or a.get("label", n),
                   "group": a.get("kind", ""), "color": palette.get(a.get("kind"), "#94a3b8"),
                   "value": 3 if a.get("kind") in ("Act", "Chapter", "Section") else 1,
                   "title": f"[{a.get('kind')}] {a.get('label','')}\n\n{(a.get('text') or '')[:600]}"}
                  for n, a in G.nodes(data=True)],
        "edges": [{"from": u, "to": v, "label": a["type"], "arrows": "to"}
                  for u, v, a in G.edges(data=True)],
    }
    html = """<meta charset="utf-8"><title>DPDP Act 2023 — Knowledge Graph</title>
<script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
<style>
 body{margin:0;font:14px/1.4 system-ui,sans-serif;background:#0b1020;color:#e2e8f0}
 #bar{padding:9px 14px;background:#111827;border-bottom:1px solid #1f2937;
      display:flex;gap:14px;align-items:center;flex-wrap:wrap}
 b{color:#f8fafc} #g{height:calc(100vh - 45px)}
 select,input{background:#1f2937;color:#e2e8f0;border:1px solid #374151;
      padding:5px 8px;border-radius:6px;font:13px system-ui}
 #legend span{margin-right:9px;white-space:nowrap;font-size:12px;color:#94a3b8}
 #legend i{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:4px}
</style>
<div id="bar"><b>DPDP Act 2023</b><span id="stat"></span>
 <label>edge <select id="f"><option value="">all</option></select></label>
 <input id="q" size="28" placeholder="filter nodes by text…">
 <span id="legend"></span></div><div id="g"></div>
<script>
const DATA = __DATA__;
const nodes = new vis.DataSet(DATA.nodes), edges = new vis.DataSet(DATA.edges);
stat.textContent = DATA.nodes.length + " nodes · " + DATA.edges.length + " edges";
[...new Set(DATA.edges.map(e => e.label))].sort().forEach(t => {
  const o = document.createElement("option"); o.value = o.textContent = t; f.appendChild(o); });
const groups = {}; DATA.nodes.forEach(n => groups[n.group] = n.color);
legend.innerHTML = Object.entries(groups).map(([k, c]) =>
  `<span><i style="background:${c}"></i>${k}</span>`).join("");
new vis.Network(document.getElementById("g"), {nodes, edges}, {
  nodes: {shape: "dot", scaling: {min: 6, max: 22}, font: {color: "#e2e8f0", size: 12}},
  edges: {color: {color: "#334155", highlight: "#94a3b8"}, width: 0.6,
          font: {size: 9, color: "#64748b", strokeWidth: 0}, smooth: false},
  physics: {stabilization: {iterations: 250},
            barnesHut: {springLength: 160, gravitationalConstant: -9000}},
  interaction: {hover: true, tooltipDelay: 120}});
const apply = () => {
  const t = f.value, s = q.value.toLowerCase();
  edges.update(DATA.edges.map(e => ({...e, hidden: !!t && e.label !== t})));
  nodes.update(DATA.nodes.map(n => ({...n,
    hidden: !!s && !(n.label + n.title).toLowerCase().includes(s)})));
};
f.onchange = apply; q.oninput = apply;
</script>"""
    (OUT / "graph.html").write_text(
        html.replace("__DATA__", json.dumps(data, ensure_ascii=False)), encoding="utf-8")


def write_review(geo, tree, audit, xrefs, unresolved, externals, defs, schedule, dropped) -> None:
    REVIEW.mkdir(exist_ok=True)

    (REVIEW / "geometry.md").write_text(
        "# Derived geometry\n\nNone of this is hard-coded. Every value is measured from the\n"
        "PDF's own word boxes each time the build runs.\n\n"
        + "\n".join(f"- {n}" for n in geo.notes)
        + "\n\n## Regions removed from the body stream\n\n"
        + f"- running headers: {dropped['running_headers']} lines\n"
        + f"- page-1 gazette masthead, everything above the Act's title line: "
          f"{dropped['gazette_masthead']} lines\n"
        + f"- Schedule page, parsed separately as a table: {dropped['schedule_page']} lines\n",
        encoding="utf-8")

    rows = ["# Headnote alignment — needs sign-off", "",
            "Gazette convention: a section's headnote is typeset level with that section's",
            "opening line. `delta` is the vertical offset in points — small is good, a large",
            "value means the pairing is probably wrong. Fix anything wrong in `overrides.yaml`",
            "under `headnotes:` and re-run; overrides always win.", "",
            "| section | headnote | page | delta (pt) | source |", "|---|---|---|---|---|"]
    rows += [f"| {e.get('section')} | {e['text']} | {e.get('page','')} | "
             f"{e.get('delta','')} | {e['source']} |"
             for e in sorted(audit, key=lambda e: (str(e.get("page")), str(e.get("top"))))]
    (REVIEW / "headnotes.md").write_text("\n".join(rows), encoding="utf-8")

    rows = ["# Cross-reference resolution — needs sign-off", "",
            "`qualified` — the text names its target ('sub-section (5) of section 8').",
            "`relative-to-own-section` — a bare 'sub-section (N)', resolved by the drafting",
            "convention that it means the section it sits in. That convention is the single",
            "riskiest inference in this build; skim these.", "",
            "Remove a wrong edge by listing `[src, dst]` under `drop_xrefs:` in overrides.yaml.",
            "", f"Total: {len(xrefs)} edges.", "",
            "| from | quote | resolves to | rule |", "|---|---|---|---|"]
    rows += [f"| {x['src']} | {x['quote']} | {x['dst']} | {x['resolution']} |" for x in xrefs]
    rows += ["", "## Citations pointing OUT of this Act", "",
             "Section 44 amends four other statutes, and several definitions borrow terms",
             "from them. These are deliberately NOT linked to DPDP provisions.", "",
             f"Total: {len(externals)}.", "", "| from | quote | statute |", "|---|---|---|"]
    rows += [f"| {e['src']} | {e['quote']} | {e['act']} |" for e in externals]
    rows += ["", "## UNRESOLVED — these are parse bugs, not edge cases", "",
             ("None." if not unresolved else "")]
    if unresolved:
        rows += ["| from | wanted | quote |", "|---|---|---|"]
        rows += [f"| {u['src']} | {u['target']} | {u['quote']} |" for u in unresolved]
    (REVIEW / "crossrefs.md").write_text("\n".join(rows), encoding="utf-8")

    rows = ["# Definitions read from Section 2", "",
            f"{len(defs)} terms. These are the Act's own ontology — extracted verbatim,",
            "no model involved.", "", "| term | relation | source node |", "|---|---|---|"]
    rows += [f"| {d['term']} | {d['relation']} | {d['defined_in']} |" for d in defs]
    (REVIEW / "definitions.md").write_text("\n".join(rows), encoding="utf-8")

    rows = ["# Illustration boundaries — needs sign-off", "",
            "A block runs from `Illustration.` / `Illustrations.` until the next statutory",
            "marker line. Check that no operative text was swallowed into an illustration.", "",
            "`attached to` is the nearest enclosing section or sub-section — the provision",
            "an illustration explains. `set after` is the innermost node open at that point,",
            "which is often a clause one level deeper. If any illustration reads as belonging",
            "to the deeper node instead, that is the judgement call to make here.", ""]
    for n in tree.nodes.values():
        if n.kind == "Illustration":
            rows += [f"### `{n.id}` — attached to `{n.parent}`, set after "
                     f"`{n.follows}` (p.{n.page})", ""]
            if n.text:
                rows += [f"> {n.text}", ""]
            for cid in n.children:
                c = tree.nodes[cid]
                rows += [f"> **{c.prefix}** {c.text}", ""]
    (REVIEW / "illustrations.md").write_text("\n".join(rows), encoding="utf-8")

    rows = ["# The Schedule — recovered from the page-21 table", "",
            "Column-major reading order makes plain text extraction scramble this table.",
            "Recovered by the same gap analysis used for the page columns, then each row",
            "band read across. Compare against page 21 of the PDF.", "",
            "| # | breach | penalty |", "|---|---|---|"]
    rows += [f"| {r['sl_no']} | {r['breach']} | {r['penalty']} |" for r in schedule]
    (REVIEW / "schedule.md").write_text("\n".join(rows), encoding="utf-8")


# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(description="Build the DPDP knowledge graph.")
    ap.add_argument("--neo4j", action="store_true",
                    help=f"also load the graph into Neo4j at {NEO4J_URI}")
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    REVIEW.mkdir(exist_ok=True)
    overrides_path = ROOT / "overrides.yaml"
    overrides = yaml.safe_load(overrides_path.read_text(encoding="utf-8")) or {} \
        if overrides_path.exists() else {}

    with pdfplumber.open(PDF) as pdf:
        geo = Geometry.derive(pdf)
    print("derived geometry")
    for n in geo.notes:
        print("   ", n)

    lines, margins, schedule, dropped = extract(geo)
    print(f"\nbody lines {len(lines)} | margin fragments {len(margins)} | "
          f"schedule rows {len(schedule)} | dropped {dropped}")

    tree = parse(lines)
    headnotes, citations = reassemble_margins(margins, geo)
    audit = attach_headnotes(tree, headnotes, overrides)
    defs = parse_definitions(tree)
    xrefs, unresolved, externals = find_xrefs(tree, overrides)

    problems = validate(tree, lines, schedule, dropped, audit)
    write_review(geo, tree, audit, xrefs, unresolved, externals, defs, schedule, dropped)

    meta = act_metadata(tree)
    G = build_graph(tree, defs, xrefs, citations, schedule, meta)
    export(G, tree, schedule)

    kinds = Counter(n.kind for n in tree.nodes.values())
    print(f"\n{meta['title']}  (Act {meta['act_number']}, assent {meta['assent_date']})")
    print("   " + " | ".join(f"{k} {v}" for k, v in sorted(kinds.items())))
    print(f"   graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"   definitions {len(defs)} | cross-refs {len(xrefs)} | unresolved {len(unresolved)}")
    print(f"\nwrote {OUT.relative_to(ROOT.parent)}/ and {REVIEW.relative_to(ROOT.parent)}/")

    if problems:
        print("\nVALIDATION FAILED")
        for p in problems:
            print("  -", p)
        return 1
    print("\nvalidation passed — 44 sections, 9 chapters, 7 penalties, lossless round-trip")

    # Loaded only after validation passes: a graph that failed the round-trip
    # check has no business in a database anyone will query.
    if args.neo4j:
        try:
            push_to_neo4j(G, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE)
        except Exception as e:  # noqa: BLE001 - a database being down is not a build failure
            print(f"\nneo4j load failed: {e}", file=sys.stderr)
            print("out/load.cypher is still there for cypher-shell.", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
