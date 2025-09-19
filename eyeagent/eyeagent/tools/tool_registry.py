"""Tool registry and metadata in English.

Provides:
- TOOL_REGISTRY: dictionary of tool metadata keyed by tool_id (same as MCP name where possible)
- Optional YAML/JSON overlay from config (EYEAGENT_TOOLS_FILE or config/tools.yml) to extend/override metadata
- Query helpers by role/modality/disease
"""
from typing import Dict, Any, List, Optional
import os
import json
from loguru import logger

# Tool IDs match MCP tool names for simplicity
TOOL_REGISTRY: Dict[str, Dict[str, Any]] = {
    # Classification
    "classification:modality": {
        "mcp_name": "classification:modality",
        "version": "1.0.0",
        "role": "orchestrator",
        "modalities": ["CFP", "OCT", "FFA"],
    "desc": "Classify imaging modality (CFP/OCT/FFA).",
    "args_schema": {"type": "object", "properties": {"image_path": {"type": "string"}}, "required": ["image_path"]},
        "desc_long": "Predicts the imaging modality to route the downstream pipeline appropriately."
    },
    "classification:laterality": {
        "mcp_name": "classification:laterality",
        "version": "1.0.0",
        "role": "orchestrator",
    "desc": "Classify eye laterality (OD/OS).",
    "args_schema": {"type": "object", "properties": {"image_path": {"type": "string"}}, "required": ["image_path"]},
        "desc_long": "Determines whether the image is from right (OD) or left (OS) eye."
    },
    "classification:cfp_quality": {
        "mcp_name": "classification:cfp_quality",
        "version": "1.2.0",
        "role": "image_analysis",
        "modalities": ["CFP"],
    "desc": "CFP image quality assessment.",
    "args_schema": {"type": "object", "properties": {"image_path": {"type": "string"}}, "required": ["image_path"]},
        "desc_long": "Estimates whether the color fundus photograph is of sufficient quality."
    },
    "classification:multidis": {
        "mcp_name": "classification:multidis",
        "version": "1.4.0",
        "role": "image_analysis",
        "modalities": ["CFP"],
    "desc": "Multi-disease screening classifier.",
    "args_schema": {"type": "object", "properties": {"image_path": {"type": "string"}}, "required": ["image_path"]},
        "desc_long": "Outputs probabilities for multiple diseases such as DR/AMD/Glaucoma to seed specialist grading."
    },
    "classification:cfp_age": {
        "mcp_name": "classification:cfp_age",
        "version": "1.0.0",
        "role": "follow_up",
        "modalities": ["CFP"],
    "desc": "Age estimation from CFP.",
    "args_schema": {"type": "object", "properties": {"image_path": {"type": "string"}}, "required": ["image_path"]},
        "desc_long": "Estimates age bracket to support management planning."
    },

    # Segmentation - CFP
    "segmentation:cfp_DR": {
        "mcp_name": "segmentation:cfp_DR",
        "version": "2.0.0",
        "role": "image_analysis",
        "modalities": ["CFP"],
    "desc": "CFP lesion segmentation for DR-related findings.",
    "args_schema": {"type": "object", "properties": {"image_path": {"type": "string"}}, "required": ["image_path"]},
        "desc_long": "Segments DR-related lesions such as hemorrhages/exudates/etc."
    },
    "segmentation:cfp_artifact": {
        "mcp_name": "segmentation:cfp_artifact",
        "version": "2.0.0",
        "role": "image_analysis",
        "modalities": ["CFP"],
    "desc": "CFP artifact segmentation.",
    "args_schema": {"type": "object", "properties": {"image_path": {"type": "string"}}, "required": ["image_path"]},
        "desc_long": "Segments artifacts that may affect diagnosis."
    },
    "segmentation:cfp_atrophy": {
        "mcp_name": "segmentation:cfp_atrophy",
        "version": "2.0.0",
        "role": "image_analysis",
        "modalities": ["CFP"],
    "desc": "CFP atrophy segmentation.",
    "args_schema": {"type": "object", "properties": {"image_path": {"type": "string"}}, "required": ["image_path"]},
        "desc_long": "Segments atrophic regions on CFP."
    },
    "segmentation:cfp_drusen": {
        "mcp_name": "segmentation:cfp_drusen",
        "version": "2.0.0",
        "role": "image_analysis",
        "modalities": ["CFP"],
    "desc": "CFP drusen segmentation.",
    "args_schema": {"type": "object", "properties": {"image_path": {"type": "string"}}, "required": ["image_path"]},
        "desc_long": "Segments drusen lesions."
    },
    "segmentation:cfp_cnv": {
        "mcp_name": "segmentation:cfp_cnv",
        "version": "2.0.0",
        "role": "image_analysis",
        "modalities": ["CFP"],
    "desc": "CFP CNV segmentation.",
    "args_schema": {"type": "object", "properties": {"image_path": {"type": "string"}}, "required": ["image_path"]},
        "desc_long": "Segments choroidal neovascularization regions."
    },
    "segmentation:cfp_mh": {
        "mcp_name": "segmentation:cfp_mh",
        "version": "2.0.0",
        "role": "image_analysis",
        "modalities": ["CFP"],
    "desc": "CFP macular hole segmentation.",
    "args_schema": {"type": "object", "properties": {"image_path": {"type": "string"}}, "required": ["image_path"]},
        "desc_long": "Segments macular hole related findings."
    },
    "segmentation:cfp_rd": {
        "mcp_name": "segmentation:cfp_rd",
        "version": "2.0.0",
        "role": "image_analysis",
        "modalities": ["CFP"],
    "desc": "CFP retinal detachment segmentation.",
    "args_schema": {"type": "object", "properties": {"image_path": {"type": "string"}}, "required": ["image_path"]},
        "desc_long": "Segments retinal detachment-related regions."
    },
    "segmentation:cfp_scar": {
        "mcp_name": "segmentation:cfp_scar",
        "version": "2.0.0",
        "role": "image_analysis",
        "modalities": ["CFP"],
    "desc": "CFP scar segmentation.",
    "args_schema": {"type": "object", "properties": {"image_path": {"type": "string"}}, "required": ["image_path"]},
        "desc_long": "Segments scar regions."
    },
    "segmentation:cfp_laserscar": {
        "mcp_name": "segmentation:cfp_laserscar",
        "version": "2.0.0",
        "role": "image_analysis",
        "modalities": ["CFP"],
    "desc": "CFP laser scar segmentation.",
    "args_schema": {"type": "object", "properties": {"image_path": {"type": "string"}}, "required": ["image_path"]},
        "desc_long": "Segments laser scar regions."
    },
    "segmentation:cfp_laserspots": {
        "mcp_name": "segmentation:cfp_laserspots",
        "version": "2.0.0",
        "role": "image_analysis",
        "modalities": ["CFP"],
    "desc": "CFP laser spot segmentation.",
    "args_schema": {"type": "object", "properties": {"image_path": {"type": "string"}}, "required": ["image_path"]},
        "desc_long": "Segments laser spots."
    },
    "segmentation:cfp_membrane": {
        "mcp_name": "segmentation:cfp_membrane",
        "version": "2.0.0",
        "role": "image_analysis",
        "modalities": ["CFP"],
    "desc": "CFP membrane segmentation.",
    "args_schema": {"type": "object", "properties": {"image_path": {"type": "string"}}, "required": ["image_path"]},
        "desc_long": "Segments membrane-like lesions."
    },
    "segmentation:cfp_edema": {
        "mcp_name": "segmentation:cfp_edema",
        "version": "2.0.0",
        "role": "image_analysis",
        "modalities": ["CFP"],
    "desc": "CFP edema segmentation.",
    "args_schema": {"type": "object", "properties": {"image_path": {"type": "string"}}, "required": ["image_path"]},
        "desc_long": "Segments edema-related regions."
    },

    # Segmentation - OCT
    "segmentation:oct_layer": {
        "mcp_name": "segmentation:oct_layer",
        "version": "2.0.0",
        "role": "image_analysis",
        "modalities": ["OCT"],
    "desc": "OCT layer segmentation.",
    "args_schema": {"type": "object", "properties": {"image_path": {"type": "string"}}, "required": ["image_path"]},
        "desc_long": "Segments retinal layers in OCT scans."
    },
    "segmentation:oct_PMchovefosclera": {
        "mcp_name": "segmentation:oct_PMchovefosclera",
        "version": "2.0.0",
        "role": "image_analysis",
        "modalities": ["OCT"],
    "desc": "OCT posterior pole/sclera segmentation.",
    "args_schema": {"type": "object", "properties": {"image_path": {"type": "string"}}, "required": ["image_path"]},
        "desc_long": "Segments posterior myopia/choroid/sclera regions (per tool naming)."
    },
    "segmentation:oct_lesion": {
        "mcp_name": "segmentation:oct_lesion",
        "version": "2.0.0",
        "role": "image_analysis",
        "modalities": ["OCT"],
    "desc": "OCT lesion segmentation.",
    "args_schema": {"type": "object", "properties": {"image_path": {"type": "string"}}, "required": ["image_path"]},
        "desc_long": "Segments lesion regions in OCT scans."
    },

    # Segmentation - FFA
    "segmentation:ffa_lesion": {
        "mcp_name": "segmentation:ffa_lesion",
        "version": "2.0.0",
        "role": "image_analysis",
        "modalities": ["FFA"],
    "desc": "FFA lesion segmentation.",
    "args_schema": {"type": "object", "properties": {"image_path": {"type": "string"}}, "required": ["image_path"]},
        "desc_long": "Segments lesion regions in FFA images."
    },

    # Disease-specific classification (subset shown)
    # All listed in the provided tool list are mapped 1:1
    **{tid: {
        "mcp_name": tid,
        "version": "1.0.0",
        "role": "specialist",
        "desc": f"Disease-specific grading/classification: {tid.split(':')[-1]}",
        "desc_long": "Per-disease fine-tuned model for grading or diagnosis."
    } for tid in [
        "disease_specific_cls:AH_finetune",
        "disease_specific_cls:AION_finetune",
        "disease_specific_cls:AMD_finetune",
        "disease_specific_cls:CM_finetune",
        "disease_specific_cls:CNV_finetune",
        "disease_specific_cls:CSCR_finetune",
        "disease_specific_cls:Coats_finetune",
        "disease_specific_cls:DR_finetune",
        "disease_specific_cls:ERM_finetune",
        "disease_specific_cls:HR_finetune",
        "disease_specific_cls:MGS_finetune",
        "disease_specific_cls:MH_finetune",
        "disease_specific_cls:PCV_finetune",
        "disease_specific_cls:RAO_finetune",
        "disease_specific_cls:RD_finetune",
        "disease_specific_cls:ROP_finetune",
        "disease_specific_cls:RP_finetune",
        "disease_specific_cls:RVO_finetune",
        "disease_specific_cls:VHL_finetune",
        "disease_specific_cls:VKH_finetune",
        "disease_specific_cls:WDS_finetune",
        "disease_specific_cls:corneal_ulcer_finetune",
        "disease_specific_cls:epiphora_finetune",
        "disease_specific_cls:glaucoma_finetune",
        "disease_specific_cls:keratoconus_finetune",
        "disease_specific_cls:macular_dystrophy_finetune",
        "disease_specific_cls:metaPM_finetune",
        "disease_specific_cls:nuclear_cataract_1_finetune",
        "disease_specific_cls:nuclear_cataract_2_finetune",
        "disease_specific_cls:nuclear_cataract_3_finetune",
        "disease_specific_cls:nuclear_cataract_4_finetune",
        "disease_specific_cls:nuclear_cataract_finetune",
        "disease_specific_cls:opaque_cornea_finetune",
        "disease_specific_cls:retinoschisis_finetune",
        "disease_specific_cls:toxoplasmosis_finetune",
        "disease_specific_cls:viral_retinitis_finetune",
    ]}
}


def _deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(a)
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_tools_config() -> Dict[str, Any]:
    """Load external tools config from YAML/JSON.

    Search order:
    1) EYEAGENT_TOOLS_FILE
    2) repo_root/config/tools.yml
    3) repo_root/config/tools.json
    """
    # Determine repo root similar to other configs
    try:
        from ..tracing.trace_logger import TraceLogger  # type: ignore
        t = TraceLogger()
        base_dir = os.path.abspath(os.path.join(t.base_dir, os.pardir))
    except Exception:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    candidates: List[str] = []
    env_path = os.getenv("EYEAGENT_TOOLS_FILE")
    if env_path:
        candidates.append(env_path)
    candidates.append(os.path.join(base_dir, "config", "tools.yml"))
    candidates.append(os.path.join(base_dir, "config", "tools.yaml"))
    candidates.append(os.path.join(base_dir, "config", "tools.json"))

    try:
        import yaml  # type: ignore
        has_yaml = True
    except Exception:
        yaml = None  # type: ignore
        has_yaml = False

    for p in candidates:
        if not os.path.isfile(p):
            continue
        try:
            with open(p, "r", encoding="utf-8") as f:
                text = f.read()
            if p.endswith(".yml") or p.endswith(".yaml"):
                if not has_yaml:
                    logger.warning("tools file is YAML but PyYAML not installed; skipping %s", p)
                    continue
                data = yaml.safe_load(text) or {}
            else:
                data = json.loads(text)
            if isinstance(data, dict):
                # Expected shape: { tool_id: {meta...}, ... }
                return data
        except Exception as e:
            logger.warning(f"failed to load tools config from {p}: {e}")
    return {}


def _apply_tools_overlay():
    """Merge external tools config into TOOL_REGISTRY (non-destructive)."""
    overlay = _load_tools_config()
    if not overlay:
        return
    updates = 0
    adds = 0
    for tool_id, meta in overlay.items():
        if not isinstance(meta, dict):
            continue
        if tool_id in TOOL_REGISTRY:
            TOOL_REGISTRY[tool_id] = _deep_merge(TOOL_REGISTRY[tool_id], meta)
            updates += 1
        else:
            # Provide sensible defaults
            meta = {**meta}
            meta.setdefault("mcp_name", tool_id)
            if not meta.get("role"):
                logger.warning(f"tools overlay: new tool '{tool_id}' missing 'role'; please specify 'role' (orchestrator/image_analysis/specialist/follow_up)")
            TOOL_REGISTRY[tool_id] = meta
            adds += 1
    if updates or adds:
        logger.debug(f"[tools] applied overlay: updates={updates} adds={adds}")


# Apply overlay at import time to keep call-sites unchanged
_apply_tools_overlay()


def list_tools(role: Optional[str] = None) -> List[Dict[str, Any]]:
    return [v | {"tool_id": k} for k, v in TOOL_REGISTRY.items() if role is None or v.get("role") == role]


def get_tool(tool_id: str) -> Optional[Dict[str, Any]]:
    t = TOOL_REGISTRY.get(tool_id)
    if t:
        return t | {"tool_id": tool_id}
    return None


def tools_for_modalities(modality: str) -> List[Dict[str, Any]]:
    out = []
    for k, v in TOOL_REGISTRY.items():
        mods = v.get("modalities")
        if mods and modality in mods:
            out.append(v | {"tool_id": k})
    return out


def specialist_tools(diseases: List[str]) -> List[Dict[str, Any]]:
    return [v | {"tool_id": k} for k, v in TOOL_REGISTRY.items() if v.get("role") == "specialist" and any(d.lower() in k.lower() for d in diseases)]


def role_tool_ids(role: str) -> List[str]:
    return [k for k, v in TOOL_REGISTRY.items() if v.get("role") == role]

# ---- Enhanced resolver: map common disease names/keywords to tool ids --------
_DISEASE_KEYWORD_TO_TOOLID = {
    # DR / Diabetic Retinopathy
    "dr": "disease_specific_cls:DR_finetune",
    "diabetic retinopathy": "disease_specific_cls:DR_finetune",
    # AMD
    "amd": "disease_specific_cls:AMD_finetune",
    "age-related macular degeneration": "disease_specific_cls:AMD_finetune",
    # CNV / PCV
    "cnv": "disease_specific_cls:CNV_finetune",
    "pcv": "disease_specific_cls:PCV_finetune",
    # Glaucoma
    "glaucoma": "disease_specific_cls:glaucoma_finetune",
    # Macular hole
    "mh": "disease_specific_cls:MH_finetune",
    "macular hole": "disease_specific_cls:MH_finetune",
    # Retinal detachment
    "rd": "disease_specific_cls:RD_finetune",
    "retinal detachment": "disease_specific_cls:RD_finetune",
    # Vein/artery occlusions
    "rvo": "disease_specific_cls:RVO_finetune",
    "rao": "disease_specific_cls:RAO_finetune",
    # ERM
    "erm": "disease_specific_cls:ERM_finetune",
    "epiretinal membrane": "disease_specific_cls:ERM_finetune",
    # Inflammations / syndromes
    "vkh": "disease_specific_cls:VKH_finetune",
    "vhl": "disease_specific_cls:VHL_finetune",
    "aion": "disease_specific_cls:AION_finetune",
    "coats": "disease_specific_cls:Coats_finetune",
    "cscr": "disease_specific_cls:CSCR_finetune",
    "central serous": "disease_specific_cls:CSCR_finetune",
    # Infectious / degenerative
    "toxoplasmosis": "disease_specific_cls:toxoplasmosis_finetune",
    "viral retinitis": "disease_specific_cls:viral_retinitis_finetune",
    # Others
    "retinoschisis": "disease_specific_cls:retinoschisis_finetune",
    "rp": "disease_specific_cls:RP_finetune",
    "retinitis pigmentosa": "disease_specific_cls:RP_finetune",
    "rop": "disease_specific_cls:ROP_finetune",
    "keratoconus": "disease_specific_cls:keratoconus_finetune",
    "macular dystrophy": "disease_specific_cls:macular_dystrophy_finetune",
    "metapm": "disease_specific_cls:metaPM_finetune",
    "pathologic myopia": "disease_specific_cls:metaPM_finetune",
    "cataract": "disease_specific_cls:nuclear_cataract_finetune",
    "nuclear cataract": "disease_specific_cls:nuclear_cataract_finetune",
    "corneal ulcer": "disease_specific_cls:corneal_ulcer_finetune",
    "epiphora": "disease_specific_cls:epiphora_finetune",
    # Less common abbreviations from registry
    "mgs": "disease_specific_cls:MGS_finetune",
    "ah": "disease_specific_cls:AH_finetune",
    "hr": "disease_specific_cls:HR_finetune",
}

def resolve_specialist_tools(diseases: List[str]) -> List[Dict[str, Any]]:
    """Resolve disease names to specialist tool metadata using substring match + keyword mapping.

    - First, use the original substring match against tool ids.
    - Then, augment with keyword mapping from common names/abbreviations to concrete tool ids.
    """
    diseases = diseases or []
    base = specialist_tools(diseases)
    selected: Dict[str, Dict[str, Any]] = {t["tool_id"]: t for t in base}
    # Keyword mapping
    for d in diseases:
        norm = (d or "").strip().lower()
        for kw, tool_id in _DISEASE_KEYWORD_TO_TOOLID.items():
            if kw in norm and tool_id in TOOL_REGISTRY:
                meta = TOOL_REGISTRY[tool_id] | {"tool_id": tool_id}
                selected[tool_id] = meta
    return list(selected.values())

# Optional: provide a function to dump current server-known tools for syncing into local descriptions
def current_server_tools() -> List[Dict[str, Any]]:
    out = []
    for tid, meta in TOOL_REGISTRY.items():
        m = {"id": tid, **{k: v for k, v in meta.items() if k in ("version", "role", "modalities", "desc", "desc_long")}}
        out.append(m)
    return out

