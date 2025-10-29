#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Usage: Symptom triage CLI for ophthalmology ontology (EDOntoR)

This script takes a free-text symptom description (in English), matches it to ontology symptoms,
recommends relevant ophthalmic exams, and lists possible medical conditions (not medical advice).

How to use:

1. Run from project root (ensure .venv and dependencies installed):
   
     python eyeagent/scripts/symptom_triage.py --text "blurred vision with a central black spot"

2. Or read input from a file:
   
     python eyeagent/scripts/symptom_triage.py --input-file path/to/symptom.txt

3. Output is a JSON object with:
     - matched_symptoms: list of ontology symptom labels and IRIs
     - recommended_exams: list of exam labels and IRIs
     - candidate_conditions: possible disease labels and IRIs
     - notes: disclaimer

Example output:
{
    "input": "blurred vision with a central black spot",
    "matched_symptoms": [ ... ],
    "recommended_exams": [ ... ],
    "candidate_conditions": [ ... ],
    "notes": "Prototype heuristic triage using ontology classes; not medical advice."
}

Notes:
- Only English input is supported.
- Prototype only; not for clinical use.
- Requires EDOntoR.owl and owl_export/disease_symptoms.csv in default locations.
"""

from __future__ import annotations
import argparse
import json
import os
import re
from typing import Dict, List, Tuple, Union

from rdflib import Graph, Namespace, URIRef
from rdflib.namespace import RDF, RDFS, OWL


HERE = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
ONTO_PATH = os.path.join(REPO_ROOT, "eyeagent", "database", "EDOntoR.owl")
EXPORT_DS = os.path.join(REPO_ROOT, "owl_export", "disease_symptoms.csv")

SCHEMA = Namespace("http://schema.org/")
EDO = Namespace("http://edo.org#")
NCIT = Namespace("http://purl.obolibrary.org/obo/ncit.owl#")


CN2EN_SYMPTOM = {}


EXAM_RULES = [
    (re.compile(r"blur|decreas|poor|dim|oscillat|variable|vision", re.I),
     ["VisualAcuityTest", "AmslerGrid", "DilatedEyeExam"]),
    (re.compile(r"distort|wavy|metamorph|central spot|black spot", re.I),
     ["AmslerGrid", "DilatedFundusExamination", "OpticalCoherenceTomography" ]),
    (re.compile(r"floaters|spots|flashes", re.I),
     ["DilatedEyeExam", "DilatedFundusExamination"]),
    (re.compile(r"pain|redness|burn|irritat|photophob|light sensitivity", re.I),
     ["SlitLampExamination", "PupilExamination"]),
    (re.compile(r"halos|halo", re.I),
     ["Gonioscopy", "Tonometry", "Pachymetry"]),
]


CONDITION_RULES = [
    (re.compile(r"distort|wavy|central.*(spot|scotoma)|metamorph|drusen|amsler", re.I), ["WetAMD", "AMD"]),
    (re.compile(r"halos|burn|peripheral vision|visual field|ioP|tonometr|gonioscop|glauc", re.I), ["Glaucoma", "PrimaryOpenAngleGlaucoma"]),
    (re.compile(r"redness|pain|photophob|light sensitivity|uveitis", re.I), ["Uveitis", "AnteriorUveitis"]),
]


def iri_fragment(u: Union[URIRef, str]) -> str:
    s = str(u)
    if "#" in s:
        return s.rsplit("#", 1)[1]
    return s.rsplit("/", 1)[-1]


def load_graph(path: str) -> Graph:
    g = Graph()
    g.parse(path)
    return g


def index_labels(g: Graph) -> Dict[URIRef, str]:
    labels: Dict[URIRef, str] = {}
    for s, p, o in g.triples((None, RDFS.label, None)):
        if isinstance(s, URIRef):
            try:
                labels[s] = str(o)
            except Exception:
                pass
    return labels


def humanize_fragment(s: str) -> str:
    # Insert spaces before capitals and around slashes/underscores; then title-case
    s2 = re.sub(r"([a-z])([A-Z])", r"\1 \2", s)
    s2 = s2.replace("_", " ").replace("/", " ")
    return " ".join(w.capitalize() for w in s2.split())


def normalize_key(label: str) -> str:
    # Lowercase and split camelCase/underscores into tokens for matching
    s = label
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", s)
    s = s.replace("_", " ").lower()
    return re.sub(r"\s+", " ", s).strip()


def collect_by_subclass(g: Graph, parent: URIRef) -> List[URIRef]:
    out: List[URIRef] = []
    for s, p, o in g.triples((None, RDFS.subClassOf, parent)):
        if isinstance(s, URIRef):
            out.append(s)
    return out


def build_catalog(g: Graph) -> Tuple[Dict[str, URIRef], Dict[str, URIRef], Dict[str, URIRef]]:
    labels = index_labels(g)

    # Symptoms: subclasses of NCIT:Symptom, plus named individuals of Symptom
    symptoms: Dict[str, URIRef] = {}
    for s in collect_by_subclass(g, NCIT.Symptom):
        raw = labels.get(s) or iri_fragment(s)
        symptoms[normalize_key(raw)] = s
    for s, _, _ in g.triples((None, RDF.type, NCIT.Symptom)):
        if isinstance(s, URIRef):
            raw = labels.get(s) or iri_fragment(s)
            symptoms.setdefault(normalize_key(raw), s)

    # Exams: subclasses of DiagnosticProcedure or PhysicalExam
    exams: Dict[str, URIRef] = {}
    for parent in (SCHEMA.DiagnosticProcedure, SCHEMA.PhysicalExam):
        for s in collect_by_subclass(g, parent):
            raw = labels.get(s) or iri_fragment(s)
            exams[normalize_key(raw)] = s

    # Conditions: subclasses of DiseaseOrDisorder and MedicalCondition
    conditions: Dict[str, URIRef] = {}
    for parent in (NCIT.DiseaseOrDisorder, SCHEMA.MedicalCondition):
        for s in collect_by_subclass(g, parent):
            raw = labels.get(s) or iri_fragment(s)
            conditions[normalize_key(raw)] = s

    return symptoms, exams, conditions


def normalize_text(text: str) -> str:
    return text.strip()


def best_matches(text: str, vocab: List[str], top_k: int = 5) -> List[str]:
    """Simple token similarity: rank by overlap and substring hits."""
    t = text.lower()
    scores: List[Tuple[float, str]] = []
    for v in vocab:
        vlow = v.lower()
        score = 0.0
        if vlow in t:
            score += 2.0
        # token overlap
        vtoks = set(re.findall(r"[a-zA-Z]+", vlow))
        ttoks = set(re.findall(r"[a-zA-Z]+", t))
        if vtoks and ttoks:
            overlap = len(vtoks & ttoks) / max(1, len(vtoks))
            score += overlap
        if score > 0:
            scores.append((score, v))
    scores.sort(key=lambda x: x[0], reverse=True)
    return [v for _, v in scores[:top_k]]


def recommend_exams(text: str, exams: Dict[str, URIRef]) -> List[Tuple[str, str]]:
    desired: List[str] = []
    for pat, names in EXAM_RULES:
        if pat.search(text):
            desired.extend(names)
    # If no explicit rule hit, use fuzzy matches against catalog
    if not desired:
        desired = best_matches(text, list(exams.keys()), top_k=3)

    out: List[Tuple[str, str]] = []
    seen_iris = set()
    for name in desired:
        # Try direct key
        nk = normalize_key(name)
        candidate_keys = []
        if nk in exams:
            candidate_keys = [nk]
        else:
            # Fuzzy pick best exam for this desired label
            ms = best_matches(name, list(exams.keys()), top_k=1)
            candidate_keys = ms
        for ck in candidate_keys:
            iri = str(exams.get(ck)) if ck in exams else None
            if iri and iri not in seen_iris:
                seen_iris.add(iri)
                out.append((ck, iri))
                break
    return out


def candidate_conditions(text: str, conditions: Dict[str, URIRef]) -> List[Tuple[str, str]]:
    # Rule-based seeds
    seeds: List[str] = []
    for pat, names in CONDITION_RULES:
        if pat.search(text):
            seeds.extend(names)
    # From prior disease_symptoms.csv if present
    try:
        if os.path.isfile(EXPORT_DS):
            import csv
            with open(EXPORT_DS, "r", encoding="utf-8") as f:
                r = csv.DictReader(f)
                for row in r:
                    s = (row.get("symptom_label") or "").lower()
                    c = row.get("condition_label") or ""
                    if s and (s in text.lower() or any(tok in text.lower() for tok in s.split())):
                        if c:
                            seeds.append(c)
    except Exception:
        pass

    # Fuzzy expand into condition catalog
    if not seeds:
        seeds = best_matches(text, list(conditions.keys()), top_k=5)
    # Map to IRIs and deduplicate
    seen = set()
    out: List[Tuple[str, str]] = []
    # Normalize unique order
    seen_seed = set()
    ordered_seeds: List[str] = []
    for s in seeds:
        if s not in seen_seed:
            seen_seed.add(s)
            ordered_seeds.append(s)

    for name in ordered_seeds:
        # exact/substring map to catalog
        cand = None
        for k in conditions.keys():
            if name.lower() == k.lower() or name.lower() in k.lower() or k.lower() in name.lower():
                cand = k
                break
        if cand:
            if cand in seen:
                continue
            seen.add(cand)
            out.append((cand, str(conditions[cand])))
            continue
        # Fallback: try EDO IRI directly if present in graph by common naming
        guess = URIRef(str(EDO) + name.replace(" ", ""))
        try:
            # We'll include the guess if it looks like a plausible EDO term
            if guess:
                out.append((name, str(guess)))
        except Exception:
            pass
    return out[:5]


def main():
    ap = argparse.ArgumentParser(description="Symptom triage (ontology-powered prototype)")
    ap.add_argument("--text", type=str, help="Patient symptom description")
    ap.add_argument("--input-file", type=str, help="Path to a text file with symptom description")
    ap.add_argument("--ontology", type=str, default=ONTO_PATH, help="Path to EDOntoR.owl")
    args = ap.parse_args()

    if not args.text and not args.input_file:
        ap.error("Please provide --text or --input-file")

    text = args.text or ""
    if args.input_file:
        try:
            with open(args.input_file, "r", encoding="utf-8") as f:
                text = f.read().strip()
        except Exception as e:
            raise SystemExit(f"Failed to read input file: {e}")

    text_norm = normalize_text(text)

    if not os.path.isfile(args.ontology):
        raise SystemExit(f"Ontology not found: {args.ontology}")

    g = load_graph(args.ontology)
    symptoms, exams, conditions = build_catalog(g)

    # Match symptoms
    matched_symptom_names = best_matches(text_norm, list(symptoms.keys()), top_k=8)
    matched_symptoms = []
    for name in matched_symptom_names:
        if name in symptoms:
            frag = iri_fragment(symptoms[name])
            matched_symptoms.append({"label": humanize_fragment(frag), "iri": str(symptoms[name])})

    # Recommend exams
    rec_exams = recommend_exams(text_norm, exams)
    recommended_exams = []
    for name, iri in rec_exams:
        frag = iri_fragment(iri)
        recommended_exams.append({"label": humanize_fragment(frag), "iri": iri})

    # Candidate conditions
    cand_conds = candidate_conditions(text_norm, conditions)
    candidate_conditions_out = []
    for name, iri in cand_conds:
        frag = iri_fragment(iri)
        candidate_conditions_out.append({"label": humanize_fragment(frag), "iri": iri})

    result = {
        "input": text,
        "normalized": text_norm,
        "matched_symptoms": matched_symptoms or None,
        "recommended_exams": recommended_exams or None,
        "candidate_conditions": candidate_conditions_out or None,
        "notes": "Prototype heuristic triage using ontology classes; not medical advice."
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
