from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
from pathlib import Path
import cv2
import numpy as np
import torch

from eyetools.core.tool_base import ToolBase

# Inlined task mapping and lesion/color configuration (removed langchain_tool_src dependency)
TASK_MAPPING = {
    "cfp_artifact": "artifact",
    "cfp_DR": "DR",
    "cfp_atrophy": "atrophy",
    "cfp_drusen": "drusen",
    "cfp_cnv": "cnv",
    "cfp_mh": "mh",
    "cfp_rd": "rd",
    "cfp_scar": "scar",
    "cfp_laserscar": "laserscar",
    "cfp_laserspots": "laserspots",
    "cfp_membrane": "membrane",
    "cfp_edema": "edema",
    "oct_layer": "octlayer",
    "oct_PMchovefosclera": "PMchovefosclera",
    "oct_lesion": "octlesion",
    "ffa_lesion": "falesion",
}

def _get_config(task: str):
    # Returns (lesions_dict, model_id)
    if task == "cfp_DR":
        lesions = {"MA": (255, 0, 0), "HE": (127, 0, 0), "EX": (255, 255, 0), "CWS": (255, 255, 255)}
    elif task == "cfp_laserscar":
        lesions = {"laser scar": (127, 127, 127)}
    elif task == "cfp_laserspots":
        lesions = {"laser spots": (255, 255, 255)}
    elif task == "cfp_atrophy":
        lesions = {"atrophy": (127, 127, -1)}
    elif task == "cfp_drusen":
        lesions = {"drusen": (255, 255, 0)}
    elif task == "cfp_scar":
        lesions = {"scar": (255, 255, 255)}
    elif task == "cfp_cnv":
        lesions = {"CNV": (255, 255, 255)}
    elif task == "cfp_rd":
        lesions = {"RD": (255, 255, 255)}
    elif task == "cfp_mh":
        lesions = {"mh": (0, 127, 127)}
    elif task == "cfp_membrane":
        lesions = {"membrane": (255, 255, 100)}
    elif task == "cfp_edema":
        lesions = {"edema": (0, 255, 255)}
    elif task == "cfp_artifact":
        lesions = {"artifact": (255, 255, 255)}
    elif task == "oct_layer":
        ls = [
            "optic disc","RNFL","GCL","IPL","INL","OPL","ONL","ELM","ISOS","RPE","choroidal"
        ]
        nvalues = [13,160,77,102,51,133,26,255,179,230,200]
        lesions = {ls[i]: (nvalues[i],)*3 for i in range(len(ls))}
        lesions["choroidal"] = (-1, -1, 200)
        lesions["chostro"] = (200, 200, 200)
        lesions["fovea"] = (0, 127, 0)
    elif task == "oct_PMchovefosclera":
        lesions = {"fovea": (0, 127, 0), "choroidal": (-1, -1, 230), "chostro": (230,230,230), "sclera": (255,255,0)}
    elif task == "oct_lesion":
        lesions = {"MH": (0, 255, 255), "fluid": (0,0,255), "PED": (255,255,0)}
    elif task == "ffa_lesion":
        base = {
            "block": (0,0,255),
            "CNV": (120,20,20),
            "leakage": (0,0,139),
            "edema": (160,82,45),
            "WD": (255,215,0),
            "staining": (255,140,0),
            "disc": (0,128,0),
            "NPA": (186,85,211),
            "NV": (255,100,100),
            "laser": (0,191,255),
            "hypofluorescence": (0,255,255),
            "macula": (50,205,50),
            "MA": (255,0,0),
        }
        lesions = base
    else:
        raise ValueError(f"Unsupported task {task}")

    cfp = ["artifact","DR","atrophy","drusen","cnv","mh","rd","scar","laserscar","laserspots","membrane","edema"]
    octs = ["octlayer","PMchovefosclera","octlesion"]
    fa = ["falesion"]
    taskd = {v:i for i,v in enumerate(cfp)}
    taskd.update({v: i+21 for i,v in enumerate(octs)})
    taskd.update({v: i+31 for i,v in enumerate(fa)})
    model_id = taskd[TASK_MAPPING[task]]
    return lesions, model_id


def _load_predictor(model_id: int, weights_root: Path):
    print(f"[seg] loading predictor model_id={model_id} weights_root={weights_root}")
    from batchgenerators.utilities.file_and_folder_operations import subdirs, join
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from nnunetv2.utilities.file_path_utilities import convert_trainer_plans_config_to_identifier

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False,
    )
    startswith = "Dataset%03.0d" % model_id
    candidates = subdirs(weights_root, prefix=startswith, join=False)
    if not candidates:
        raise FileNotFoundError(f"Segmentation weights not found for id={model_id} prefix={startswith} in {weights_root}")
    dataset_dir = candidates[0]
    model_folder = join(
        weights_root,
        dataset_dir,
        convert_trainer_plans_config_to_identifier("nnUNetTrainer", "nnUNetPlans", "2d"),
    )
    print(f"[seg] resolved model_folder={model_folder}")
    predictor.initialize_from_trained_model_folder(
        model_folder,
        use_folds=(0,),
        checkpoint_name="checkpoint_final.pth",
    )
    return predictor


def _remove_small(mask: np.ndarray, min_size: int) -> np.ndarray:
    nb_components, output, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    sizes = stats[1:, -1]
    cleaned = np.zeros_like(mask)
    for i in range(nb_components - 1):
        if sizes[i] >= min_size:
            cleaned[output == i + 1] = 1
    return cleaned


def _colorize(seg: np.ndarray, lesions: Dict, min_object: int) -> Tuple[np.ndarray, Dict[str, Any]]:
    rgb = np.zeros((*seg.shape, 3), dtype=np.uint8)
    stats_counts: Dict[str, int] = {}
    stats_areas: Dict[str, list] = {}
    for idx, (name, color) in enumerate(lesions.items(), 1):
        if isinstance(color, int):
            color = (color, color, color)
        mask = (seg == idx).astype(np.uint8)
        if name != "MA":
            mask = _remove_small(mask, min_object)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        areas = [float(cv2.contourArea(c)) for c in contours]
        stats_counts[name] = len(contours)
        stats_areas[name] = areas
        for c in range(3):
            if c < len(color) and color[c] > 0:
                rgb[..., c] = np.where(mask == 1, color[c], rgb[..., c])
    return rgb[:, :, ::-1], {"counts": stats_counts, "areas": stats_areas}


@dataclass
class SegmentationVariantMeta:
    task: str
    min_object_size: int = 1


class SegmentationTool(ToolBase):
    def __init__(self, meta: Dict[str, Any], params: Dict[str, Any]):
        super().__init__(meta, params)
        self.task: str = params.get("task") or meta.get("params", {}).get("task")
        if self.task not in TASK_MAPPING:
            raise ValueError(f"Unsupported segmentation task {self.task}")
        self.temp_dir = Path(params.get("base_path", "temp/segmentation")) / self.task
        self.weights_root = Path(params.get("weights_root", "weights/segmentation"))
        self.min_object_size = int(params.get("min_object_size", 1))
        # optional offset if weight directory numbering differs from internal mapping
        self.model_id_offset = int(params.get("model_id_offset", 0))
        self.predictor = None
        self.lesions: Dict[str, Any] = {}
        self.model_id: Optional[int] = None

    def prepare(self):  # lightweight
        # create dirs
        for sub in ("images", "pred", "rgb", "merge", "overlay"):
            p = self.temp_dir / sub
            p.mkdir(parents=True, exist_ok=True)
        return True

    def load_model(self):
        if self.predictor is not None:
            return
        self.lesions, self.model_id = _get_config(self.task)
        if self.model_id_offset:
            self.model_id = self.model_id + self.model_id_offset
        self.predictor = _load_predictor(self.model_id, self.weights_root)
        self._model_loaded = True

    # Request format: {"inputs": {"image_path": str}}
    def predict(self, request: Dict[str, Any]):
        inputs = request.get("inputs") or request
        image_path = inputs.get("image_path")
        if not image_path:
            raise ValueError("image_path missing in request.inputs")
        image_path = str(image_path)
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        self.ensure_model_loaded()
        file_name = Path(image_path).name
        # copy image
        dst = self.temp_dir / "images" / file_name
        try:
            import shutil
            shutil.copy(image_path, dst)
        except Exception:
            pass
        start = time.time()
        # run predictor sequential
        result = self.predictor.predict_from_files_sequential(
            [[str(dst)]],
            str(self.temp_dir / "pred"),
            False,
            True,
            None,
        )
        # load seg
        pred_dir = self.temp_dir / "pred"
        seg_path = pred_dir / f"{Path(image_path).stem}.png"
        import cv2 as _cv2, os as _os
        seg = _cv2.imread(str(seg_path), _cv2.IMREAD_GRAYSCALE)
        if seg is None:
            # Attempt fallback: if only one png exists in pred_dir, use it
            pngs = list(pred_dir.glob('*.png'))
            if len(pngs) == 1:
                seg_path = pngs[0]
                seg = _cv2.imread(str(seg_path), _cv2.IMREAD_GRAYSCALE)
            else:
                # Try nii.gz
                niis = list(pred_dir.glob('*.nii.gz'))
                if niis:
                    try:
                        import nibabel as nib
                        nii = nib.load(str(niis[0]))
                        data = nii.get_fdata().astype('uint8')
                        seg = data.squeeze()
                        # Save a png for downstream steps
                        _cv2.imwrite(str(pred_dir / (niis[0].stem + '.png')), seg)
                        seg_path = pred_dir / (niis[0].stem + '.png')
                    except Exception as e:  # pragma: no cover
                        pass
        if seg is None:
            listing = [p.name for p in pred_dir.glob('*')]
            raise RuntimeError(f"Segmentation output missing. Expected {seg_path.name}. Dir listing={listing}")
        # Normalize binary-like masks (0,255) to {0,1}
        uniques = np.unique(seg)
        if set(uniques.tolist()) <= {0, 255}:
            seg = (seg > 0).astype(np.uint8)
        orig = _cv2.imread(image_path)
        h, w = orig.shape[:2]
        seg = _cv2.resize(seg, (w, h), interpolation=_cv2.INTER_NEAREST)
        colorized, stats = _colorize(seg, self.lesions, self.min_object_size)
        merged = np.hstack([orig, colorized])
        out_name = f"{Path(image_path).stem}_{Path(image_path).suffix}"
        _cv2.imwrite(str(self.temp_dir / "merge" / out_name), merged)
        _cv2.imwrite(str(self.temp_dir / "rgb" / out_name), colorized)
        # overlay: zero out seg>0
        overlay = orig.copy()
        overlay[seg > 0] = 0
        overlay = _cv2.addWeighted(overlay, 1, colorized, 1, 0)
        _cv2.imwrite(str(self.temp_dir / "overlay" / out_name), overlay)
        latency = time.time() - start
        return {
            "task": self.task,
            **stats,
            "output_paths": {
                "merged": str(self.temp_dir / "merge" / out_name),
                "colorized": str(self.temp_dir / "rgb" / out_name),
                "overlay": str(self.temp_dir / "overlay" / out_name),
            },
            "inference_time": round(latency, 3),
        }


def load_tool(task: str, **kw):
    return SegmentationTool({"id": f"segmentation:{task}", "entry": "tool_impl:SegmentationTool"}, {"task": task, **kw})

__all__ = ["SegmentationTool", "load_tool"]
