from __future__ import annotations
from .base_agent import BaseAgent
from .registry import register_agent
from typing import Dict, Any
from eyeagent.triage import common as triage_common


def run_symptom_triage(symptom_text: str) -> Dict[str, Any]:
    g = triage_common.load_graph(triage_common.ONTO_PATH)
    symptoms, exams, conditions = triage_common.build_catalog(g)
    matched_symptom_names = triage_common.best_matches(symptom_text, list(symptoms.keys()), top_k=8)
    matched_symptoms = []
    for name in matched_symptom_names:
        if name in symptoms:
            frag = triage_common.iri_fragment(symptoms[name])
            matched_symptoms.append({"label": triage_common.humanize_fragment(frag), "iri": str(symptoms[name])})
    rec_exams = triage_common.recommend_exams(symptom_text, exams)
    recommended_exams = []
    for name, iri in rec_exams:
        frag = triage_common.iri_fragment(iri)
        recommended_exams.append({"label": triage_common.humanize_fragment(frag), "iri": iri})
    cand_conds = triage_common.candidate_conditions(symptom_text, conditions)
    candidate_conditions_out = []
    for name, iri in cand_conds:
        frag = triage_common.iri_fragment(iri)
        candidate_conditions_out.append({"label": triage_common.humanize_fragment(frag), "iri": iri})
    return {
        "input": symptom_text,
        "matched_symptoms": matched_symptoms or None,
        "recommended_exams": recommended_exams or None,
        "candidate_conditions": candidate_conditions_out or None,
        "notes": "Prototype heuristic triage using ontology classes; not medical advice."
    }

@register_agent
class SymptomTriageAgent(BaseAgent):
    role = "symptom_triage"
    name = "SymptomTriageAgent"
    allowed_tool_ids = []
    system_prompt = "You are a symptom triage agent. Given a patient's symptom description, recommend exams and possible conditions."

    async def a_run(self, context):
        symptom_text = context.get("symptom_text", "")
        result = run_symptom_triage(symptom_text)
        return {
            "agent": self.name,
            "role": self.role,
            "outputs": result,
            "reasoning": "Symptom triage completed."
        }
