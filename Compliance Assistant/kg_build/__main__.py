"""
The build pipeline, end to end.

    python -m kg_build              parse, validate, write chunks.json
    python -m kg_build --neo4j      also replace the graph in Neo4j

Ordering is deliberate: nothing is written and nothing is pushed until
`validate()` has confirmed the parsed tree reassembles into the source
document character for character. A build that cannot prove it preserved the
Act's words is a build that must not reach the database.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

# Run as a module from the project root so `backend.indexing` resolves.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend import config                                    # noqa: E402
from backend.indexing import save_chunks                      # noqa: E402
from kg_build import extract, graph as graph_mod              # noqa: E402
from kg_build.chunks import attach_plain_language, build_chunks  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
log = logging.getLogger("kg_build")

PDF_PATH = config.DATA_DIR / "dpdp_act_2023.pdf"
OVERRIDES = config.DATA_DIR / "overrides.yaml"
# Costs hours of local inference to regenerate, so it lives beside the source
# data and is never treated as a disposable build artifact.
PLAIN_LANGUAGE = config.DATA_DIR / "plain_language.json"
CHUNKS_OUT = config.DATA_DIR / "chunks.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--neo4j", action="store_true",
                        help="replace the graph in Neo4j after validation")
    args = parser.parse_args()

    if not PDF_PATH.is_file():
        log.error("source PDF not found: %s", PDF_PATH)
        return 1

    overrides = (yaml.safe_load(OVERRIDES.read_text(encoding="utf-8")) or {}
                 if OVERRIDES.is_file() else {})

    import pdfplumber
    with pdfplumber.open(PDF_PATH) as pdf:
        geometry = extract.Geometry.derive(pdf)
    for note in geometry.notes:
        log.debug("geometry: %s", note)

    lines, margins, schedule, dropped = extract.extract(geometry)
    tree = extract.parse(lines)
    headnotes, citations = extract.reassemble_margins(margins, geometry)
    audit = extract.attach_headnotes(tree, headnotes, overrides)
    definitions = extract.parse_definitions(tree)
    xrefs, unresolved, externals = extract.find_xrefs(tree, overrides)
    metadata = graph_mod.act_metadata(tree)

    # The gate. Nothing is written and nothing is pushed unless the parsed
    # tree reassembles into the source document character for character.
    if problems := extract.validate(tree, lines, schedule, dropped, audit):
        log.error("VALIDATION FAILED — refusing to write or publish this build")
        for problem in problems:
            log.error("  - %s", problem)
        return 1
    log.info("validated: lossless round-trip against the source text")

    # PageRank authority is computed inside build_graph() itself.
    knowledge_graph = graph_mod.build_graph(
        tree, definitions, xrefs, citations, schedule, metadata)
    log.info("graph: %d nodes, %d edges",
             knowledge_graph.number_of_nodes(), knowledge_graph.number_of_edges())

    tree_dict = {k: v.__dict__ if hasattr(v, "__dict__") else v
                 for k, v in tree.nodes.items()}
    chunks = build_chunks(tree_dict, knowledge_graph, schedule)

    plain = (json.loads(PLAIN_LANGUAGE.read_text(encoding="utf-8"))
             if PLAIN_LANGUAGE.is_file() else {})
    if not plain:
        log.warning("no plain-language layer at %s — layperson phrasing will "
                    "match less well", PLAIN_LANGUAGE.name)
    chunks = attach_plain_language(chunks, plain)

    save_chunks(chunks, CHUNKS_OUT)
    log.info("wrote %s (%d chunks)", CHUNKS_OUT.name, len(chunks))

    if args.neo4j:
        graph_mod.push_to_neo4j(
            knowledge_graph, config.NEO4J_URI, config.NEO4J_USER,
            config.NEO4J_PASSWORD, config.NEO4J_DATABASE)
    else:
        log.info("skipped Neo4j (pass --neo4j to publish this build)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
