"""Build data/term_accessions.json.gz: a lower-cased label -> accession map for PSI-MS and PRIDE terms.

The SDRF-Proteomics specification asks for ``NT=<label>;AC=<accession>`` in comment columns such as
comment[instrument], comment[cleavage agent details], comment[label] and comment[dissociation method].
The app's term lists only hold labels, so this map supplies the accessions.

Usage (from the repository root):
    python scripts/build_accession_map.py ontology/psi-ms20251015.obo ontology/pride_cv20251015.obo
"""
import gzip
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_obo(path):
    """Yield (accession, name, synonyms) for every non-obsolete [Term] in an OBO file."""
    opener = gzip.open if path.endswith(".gz") else open
    term = None
    with opener(path, "rt", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if line.startswith("["):
                if term and term.get("id") and term.get("name") and not term.get("obsolete"):
                    yield term["id"], term["name"], term["synonyms"]
                term = {"synonyms": []} if line == "[Term]" else None
                continue
            if term is None or ":" not in line:
                continue
            key, _, value = line.partition(": ")
            if key == "id":
                term["id"] = value
            elif key == "name":
                term["name"] = value
            elif key == "is_obsolete" and value.startswith("true"):
                term["obsolete"] = True
            elif key == "synonym" and ("EXACT" in value) and value.startswith('"'):
                term["synonyms"].append(value[1:].split('"', 1)[0])
    if term and term.get("id") and term.get("name") and not term.get("obsolete"):
        yield term["id"], term["name"], term["synonyms"]


def main(paths):
    # {lower-cased label or exact synonym: {"MS": [accession, name], "PRIDE": [accession, name]}}
    mapping = {}
    terms = [t for path in paths for t in parse_obo(path) if t[0].startswith(("MS:", "PRIDE:"))]
    # Names win over synonyms; within one ontology the first term seen wins.
    for accession, name, _ in terms:
        mapping.setdefault(name.lower(), {}).setdefault(accession.split(":")[0], [accession, name])
    for accession, name, synonyms in terms:
        for syn in synonyms:
            mapping.setdefault(syn.lower(), {}).setdefault(accession.split(":")[0], [accession, name])
    out = os.path.join(ROOT, "data", "term_accessions.json.gz")
    with gzip.open(out, "wt", encoding="utf-8") as fh:
        json.dump(mapping, fh)
    print(f"Wrote {len(mapping)} labels to {out}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
