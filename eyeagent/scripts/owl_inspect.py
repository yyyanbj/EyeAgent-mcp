#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
owl_inspect.py
---------------
Extract classes, properties, individuals, and raw triples from an OWL/RDF ontology.

Usage:
    python owl_inspect.py path/to/ontology.owl --outdir out --format csv --triple-limit 100000

Dependencies:
    pip install owlready2 rdflib

Outputs (in --outdir):
    - classes.csv/tsv          (iri,label,superclasses)
    - object_properties.csv/tsv(iri,label,domain,range)
    - data_properties.csv/tsv  (iri,label,domain,range,datatype_if_any)
    - annotation_properties.csv/tsv(iri,label)
    - individuals.csv/tsv      (iri,label,types)
    - triples.csv/tsv          (subject,predicate,object)
"""
import argparse
import csv
import sys
from pathlib import Path
from typing import Iterable, List

# Silence owlready2 excessive logs
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from owlready2 import get_ontology, Thing, ObjectProperty, DataProperty, AnnotationProperty
from rdflib import Graph, URIRef, BNode, Literal
from rdflib.namespace import RDFS

def write_rows(path: Path, header: List[str], rows: Iterable[List[str]], fmt: str = "csv"):
    path.parent.mkdir(parents=True, exist_ok=True)
    newline = "" if fmt == "csv" else ""
    delimiter = "," if fmt == "csv" else "\t"
    with path.open("w", newline=newline, encoding="utf-8") as f:
        w = csv.writer(f, delimiter=delimiter)
        w.writerow(header)
        for r in rows:
            # Ensure all are strings
            w.writerow([(x if isinstance(x, str) else str(x)) for x in r])

def pretty_label(entity) -> str:
    # Try rdfs:label, fallback to name fragment
    try:
        lbls = getattr(entity, "label", [])
        if lbls:
            return lbls[0]
    except Exception:
        pass
    iri = str(getattr(entity, "iri", getattr(entity, "iri", "")) or getattr(entity, "name", ""))
    if "#" in iri:
        return iri.split("#")[-1]
    return iri.rsplit("/", 1)[-1] if "/" in iri else iri

def iri_of(obj) -> str:
    try:
        return str(obj.iri)
    except Exception:
        try:
            return obj.storid  # owlready internal id
        except Exception:
            return str(obj)

def list_classes(onto):
    for cls in onto.classes():
        if cls is Thing:  # skip the top unless user wants it
            continue
        supers = []
        for sc in getattr(cls, "is_a", []):
            try:
                val = iri_of(sc)
            except Exception:
                val = str(sc)
            supers.append(str(val))
        yield [iri_of(cls), pretty_label(cls), "|".join(map(str, supers)) or ""]


def list_obj_props(onto):
    for p in onto.object_properties():
        dom = [iri_of(d) for d in getattr(p, "domain", [])]
        rng = [iri_of(r) for r in getattr(p, "range", [])]
        yield [iri_of(p), pretty_label(p), "|".join(dom) or "", "|".join(rng) or ""]

def list_data_props(onto):
    for p in onto.data_properties():
        dom = [iri_of(d) for d in getattr(p, "domain", [])]
        rng = [str(r) for r in getattr(p, "range", [])]
        yield [iri_of(p), pretty_label(p), "|".join(dom) or "", "|".join(rng) or ""]

def list_annot_props(onto):
    for p in onto.annotation_properties():
        yield [iri_of(p), pretty_label(p)]

def list_individuals(onto):
    for ind in onto.individuals():
        types = [iri_of(t) for t in getattr(ind, "is_a", []) if t is not Thing]
        yield [iri_of(ind), pretty_label(ind), "|".join(types) or ""]

def dump_triples(in_path: Path, limit: int = 0):
    g = Graph()
    # rdflib usually auto-detects the format; explicit format can be set if needed.
    g.parse(in_path.as_posix())
    count = 0
    for s, p, o in g:
        def fmt_term(t):
            if isinstance(t, URIRef):
                return str(t)
            if isinstance(t, BNode):
                return f"_:{t}"
            if isinstance(t, Literal):
                if t.language:
                    return f'"{t}"@{t.language}'
                if t.datatype:
                    return f'"{t}"^^{t.datatype}'
                return f'"{t}"'
            return str(t)
        yield [fmt_term(s), fmt_term(p), fmt_term(o)]
        count += 1
        if limit and count >= limit:
            break

def main():
    ap = argparse.ArgumentParser(description="Extract structure and triples from an OWL ontology.")
    ap.add_argument("owl_path", help="Path to .owl/.rdf/.ttl file")
    ap.add_argument("--outdir", default="owl_export", help="Directory to write CSV/TSV files")
    ap.add_argument("--format", choices=["csv","tsv"], default="csv", help="Output delimiter format")
    ap.add_argument("--triple-limit", type=int, default=0, help="Limit number of triples exported (0 = no limit)")
    args = ap.parse_args()

    in_path = Path(args.owl_path)
    if not in_path.exists():
        print(f"❌ File not found: {in_path}", file=sys.stderr)
        sys.exit(1)

    outdir = Path(args.outdir)
    fmt = args.format

    # Load ontology with owlready2 (gives higher-level access)
    onto = get_ontology(in_path.as_posix()).load()

    # Write classes
    write_rows(outdir / f"classes.{fmt}", ["iri","label","superclasses"], list_classes(onto), fmt=fmt)
    # Write properties
    write_rows(outdir / f"object_properties.{fmt}", ["iri","label","domain","range"], list_obj_props(onto), fmt=fmt)
    write_rows(outdir / f"data_properties.{fmt}", ["iri","label","domain","range"], list_data_props(onto), fmt=fmt)
    write_rows(outdir / f"annotation_properties.{fmt}", ["iri","label"], list_annot_props(onto), fmt=fmt)
    # Write individuals
    write_rows(outdir / f"individuals.{fmt}", ["iri","label","types"], list_individuals(onto), fmt=fmt)
    # Write triples (raw)
    write_rows(outdir / f"triples.{fmt}", ["subject","predicate","object"], dump_triples(in_path, args.triple_limit), fmt=fmt)

    # Print a short summary
    num_classes = sum(1 for _ in onto.classes()) - 1  # exclude Thing
    num_objp = sum(1 for _ in onto.object_properties())
    num_datap = sum(1 for _ in onto.data_properties())
    num_annp = sum(1 for _ in onto.annotation_properties())
    num_inds = sum(1 for _ in onto.individuals())

    print("✅ Export complete.")
    print(f" • Input: {in_path}")
    print(f" • Output dir: {outdir.resolve()} ({fmt.upper()})")
    print(f" • Classes: {num_classes}")
    print(f" • Object properties: {num_objp}")
    print(f" • Data properties: {num_datap}")
    print(f" • Annotation properties: {num_annp}")
    print(f" • Individuals: {num_inds}")
    print(" • Triples: written (limited by --triple-limit if set)")

if __name__ == "__main__":
    main()
