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
from typing import Tuple

# Tool IDs match MCP tool names for simplicity
TOOL_REGISTRY: Dict[str, Dict[str, Any]] = {
    # Multimodal conversion/generation
    "multimodal:fundus2oct": {
        "mcp_name": "multimodal:fundus2oct",
        "version": "0.1.0",
        "role": "multimodal",
        "modalities": ["CFP"],
        "desc": "Generate a pseudo-OCT volume from a fundus (CFP) image and export montage/frames/GIF.",
        "args_schema": {
            "type": "object",
            "properties": {
                "image_path": {"type": "string"},
                "sampling_steps": {"type": "integer"},
                "slices": {"type": "integer"},
                "height": {"type": "integer"},
                "width": {"type": "integer"}
            },
            "required": ["image_path"]
        },
        "desc_long": "Creates a lightweight 3D-like OCT slice stack from CFP with deterministic placeholders when heavy models are unavailable. Outputs include montage PNG, frame directory, and animated GIF."
    },
    "multimodal:fundus2eyeglobe": {
        "mcp_name": "multimodal:fundus2eyeglobe",
        "version": "0.1.0",
        "role": "multimodal",
        "modalities": ["CFP"],
        "desc": "Generate a simple eye-globe point cloud and PNG/GIF views from a fundus image.",
        "args_schema": {
            "type": "object",
            "properties": {
                "image_path": {"type": "string"},
                "eye_category": {"type": "string", "enum": ["OD", "OS"]},
                "sampling_steps": {"type": "integer"},
                "SE": {"type": "number"},
                "AL": {"type": "number"},
                "num_points": {"type": "integer"}
            },
            "required": ["image_path"]
        },
        "desc_long": "Produces a synthetic eye globe geometry (PLY) and projections for visualization; accepts optional eye category (OD/OS) and biometric parameters (SE, AL)."
    },
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

    # Knowledge/RAG tools
    "rag:query": {
        "mcp_name": "rag:query",
        "version": "1.0.0",
        "role": "knowledge",
        "desc": "Query internal ophthalmology knowledge base (RAG)",
        "args_schema": {"type": "object", "properties": {"query": {"type": "string"}, "top_k": {"type": "integer"}}, "required": ["query"]},
        "desc_long": "Retrieves top relevant passages/documents from an ophthalmology corpus to support decisions."
    },
    "web_search:pubmed": {
        "mcp_name": "web_search:pubmed",
        "version": "1.0.0",
        "role": "knowledge",
        "desc": "Search PubMed for ophthalmology literature",
        "args_schema": {"type": "object", "properties": {"query": {"type": "string"}, "top_k": {"type": "integer"}}, "required": ["query"]},
        "desc_long": "Queries PubMed and returns recent and relevant studies."
    },
    "web_search:tavily": {
        "mcp_name": "web_search:tavily",
        "version": "1.0.0",
        "role": "knowledge",
        "desc": "Web search via Tavily API",
        "args_schema": {"type": "object", "properties": {"query": {"type": "string"}, "top_k": {"type": "integer"}}, "required": ["query"]},
        "desc_long": "General web search tool to supplement knowledge when RAG isn't sufficient."
    },

    # Disease-specific classification (subset shown)
    # All listed in the provided tool list are mapped 1:1
    **{tid: {
        "mcp_name": tid,
        "version": "1.0.0",
        "role": "specialist",
        "desc": f"Disease-specific grading/classification: {tid.split(':')[-1]}",
        "args_schema": {"type": "object", "properties": {"image_path": {"type": "string"}}, "required": ["image_path"]},
        "desc_long": "Per-disease fine-tuned model for grading or diagnosis."
    } for tid in [
        "disease_specific_cls:AH",
        "disease_specific_cls:AION",
        "disease_specific_cls:AMD",
        "disease_specific_cls:CM",
        "disease_specific_cls:CNV",
        "disease_specific_cls:CSCR",
        "disease_specific_cls:Coats",
        "disease_specific_cls:DR",
        "disease_specific_cls:ERM",
        "disease_specific_cls:HR",
        "disease_specific_cls:MGS",
        "disease_specific_cls:MH",
        "disease_specific_cls:PCV",
        "disease_specific_cls:RAO",
        "disease_specific_cls:RD",
        "disease_specific_cls:ROP",
        "disease_specific_cls:RP",
        "disease_specific_cls:RVO",
        "disease_specific_cls:VHL",
        "disease_specific_cls:VKH",
        "disease_specific_cls:WDS",
        "disease_specific_cls:corneal_ulcer",
        "disease_specific_cls:epiphora",
        "disease_specific_cls:glaucoma",
        "disease_specific_cls:keratoconus",
        "disease_specific_cls:macular_dystrophy",
        "disease_specific_cls:metaPM",
        # "disease_specific_cls:nuclear_cataract_1",
        # "disease_specific_cls:nuclear_cataract_2",
        # "disease_specific_cls:nuclear_cataract_3",
        # "disease_specific_cls:nuclear_cataract_4",
        "disease_specific_cls:nuclear_cataract",
        "disease_specific_cls:opaque_cornea",
        "disease_specific_cls:retinoschisis",
        "disease_specific_cls:toxoplasmosis",
        "disease_specific_cls:viral_retinitis",
    ]}
}

# Basic disease name/abbreviation mapping to enrich disease_specific results.
# Keys are upper tokens (normalized) without suffixes like  or numeric tails.
_DISEASE_NAME_MAP: Dict[str, Tuple[str, str]] = {
    "DR": ("Diabetic Retinopathy", "DR"),
    "AMD": ("Age-related Macular Degeneration", "AMD"),
    "RVO": ("Retinal Vein Occlusion", "RVO"),
    "RD": ("Retinal Detachment", "RD"),
    "CNV": ("Choroidal Neovascularization", "CNV"),
    "PCV": ("Polypoidal Choroidal Vasculopathy", "PCV"),
    "ERM": ("Epiretinal Membrane", "ERM"),
    "AION": ("Anterior Ischemic Optic Neuropathy", "AION"),
    "CSCR": ("Central Serous Chorioretinopathy", "CSCR"),
    "VKH": ("Vogt–Koyanagi–Harada disease", "VKH"),
    "VHL": ("Von Hippel–Lindau disease", "VHL"),
    "ROP": ("Retinopathy of Prematurity", "ROP"),
    "RP": ("Retinitis Pigmentosa", "RP"),
    "MH": ("Macular Hole", "MH"),
    "GLAUCOMA": ("Glaucoma", "GLC"),
    "KERATOCONUS": ("Keratoconus", "KC"),
    "NUCLEAR CATARACT": ("Nuclear Cataract", "NC"),
    "CORNEAL ULCER": ("Corneal Ulcer", "CU"),
    # Fallbacks will be generated heuristically
}

def _humanize(s: str) -> str:
    s = s.replace("_", " ").strip()
    return " ".join(w.capitalize() if w else w for w in s.split())

def disease_names_for_tool(tool_id: str) -> Dict[str, str]:
    """Return {'full': ..., 'abbr': ...} derived from a disease_specific tool_id.

    Heuristics:
    - Parse token after 'disease_specific_cls:' and before optional suffix like ''
    - Try uppercase map; fallback to humanized full name and acronym abbreviation.
    """
    try:
        if not isinstance(tool_id, str) or not tool_id.startswith("disease_specific_cls:"):
            return {}
        token = tool_id.split(":", 1)[1]
        token = token.rsplit("", 1)[0]
        # Remove trailing _<digit> variants (e.g., nuclear_cataract_4)
        import re
        token = re.sub(r"_\d+$", "", token)
        up = token.replace("_", " ").upper()
        if up in _DISEASE_NAME_MAP:
            full, abbr = _DISEASE_NAME_MAP[up]
            return {"full": full, "abbr": abbr}
        # Common composite mapping attempts
        if "NUCLEAR" in up and "CATARACT" in up:
            return {"full": "Nuclear Cataract", "abbr": "NC"}
        # Generic fallback: humanize and build acronym
        full = _humanize(token)
        abbr = "".join(w[0].upper() for w in full.split() if w and w[0].isalpha())[:5]
        return {"full": full, "abbr": abbr}
    except Exception:
        return {}


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
        repo_root = os.path.abspath(os.path.join(t.base_dir, os.pardir))
    except Exception:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    # Both new preferred location (eyeagent/config) and legacy (config) are supported
    eye_cfg = os.path.join(repo_root, "eyeagent", "config")
    legacy_cfg = os.path.join(repo_root, "config")

    candidates: List[str] = []
    env_path = os.getenv("EYEAGENT_TOOLS_FILE")
    if env_path:
        candidates.append(env_path)
    for d in (eye_cfg, legacy_cfg):
        candidates.append(os.path.join(d, "tools.yml"))
        candidates.append(os.path.join(d, "tools.yaml"))
        candidates.append(os.path.join(d, "tools.json"))

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
    "dr": "disease_specific_cls:DR",
    "diabetic retinopathy": "disease_specific_cls:DR",
    # AMD
    "amd": "disease_specific_cls:AMD",
    "age-related macular degeneration": "disease_specific_cls:AMD",
    # CNV / PCV
    "cnv": "disease_specific_cls:CNV",
    "pcv": "disease_specific_cls:PCV",
    # Glaucoma
    "glaucoma": "disease_specific_cls:glaucoma",
    # Macular hole
    "mh": "disease_specific_cls:MH",
    "macular hole": "disease_specific_cls:MH",
    # Retinal detachment
    "rd": "disease_specific_cls:RD",
    "retinal detachment": "disease_specific_cls:RD",
    # Vein/artery occlusions
    "rvo": "disease_specific_cls:RVO",
    "rao": "disease_specific_cls:RAO",
    # ERM
    "erm": "disease_specific_cls:ERM",
    "epiretinal membrane": "disease_specific_cls:ERM",
    # Inflammations / syndromes
    "vkh": "disease_specific_cls:VKH",
    "vhl": "disease_specific_cls:VHL",
    "aion": "disease_specific_cls:AION",
    "coats": "disease_specific_cls:Coats",
    "cscr": "disease_specific_cls:CSCR",
    "central serous": "disease_specific_cls:CSCR",
    # Infectious / degenerative
    "toxoplasmosis": "disease_specific_cls:toxoplasmosis",
    "viral retinitis": "disease_specific_cls:viral_retinitis",
    # Others
    "retinoschisis": "disease_specific_cls:retinoschisis",
    "rp": "disease_specific_cls:RP",
    "retinitis pigmentosa": "disease_specific_cls:RP",
    "rop": "disease_specific_cls:ROP",
    "keratoconus": "disease_specific_cls:keratoconus",
    "macular dystrophy": "disease_specific_cls:macular_dystrophy",
    "metapm": "disease_specific_cls:metaPM",
    "pathologic myopia": "disease_specific_cls:metaPM",
    "cataract": "disease_specific_cls:nuclear_cataract",
    "nuclear cataract": "disease_specific_cls:nuclear_cataract",
    "corneal ulcer": "disease_specific_cls:corneal_ulcer",
    "epiphora": "disease_specific_cls:epiphora",
    # Less common abbreviations from registry
    "mgs": "disease_specific_cls:MGS",
    "ah": "disease_specific_cls:AH",
    "hr": "disease_specific_cls:HR",
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

