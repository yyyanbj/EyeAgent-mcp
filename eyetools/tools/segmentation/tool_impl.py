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

    # best-effort environment hints for nnUNet
    import os as _os
    _os.environ.setdefault("nnUNet_raw", str(weights_root))
    _os.environ.setdefault("nnUNet_preprocessed", str(weights_root))
    _os.environ.setdefault("nnUNet_results", str(weights_root))

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
        # If False, raise an error when no real segmentation is produced (no synthetic fallback)
        self.allow_fallback = bool(params.get("allow_fallback", False))
        # optional offset if weight directory numbering differs from internal mapping
        self.model_id_offset = int(params.get("model_id_offset", 0))
        self.predictor = None
        self.lesions: Dict[str, Any] = {}
        self.model_id: Optional[int] = None
        self.model_loaded = False

    @staticmethod
    def describe_outputs(meta: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """Static description of segmentation outputs for the given task.

        Returns label names and structure of counts/areas without loading the model.
        """
        task = (params or {}).get("task") or (meta.get("params", {}) or {}).get("task")
        # mirror label names used in _get_config without importing heavy deps
        labels_map = {
            "cfp_DR": ["MA", "HE", "EX", "CWS"],
            "cfp_laserscar": ["laser scar"],
            "cfp_laserspots": ["laser spots"],
            "cfp_atrophy": ["atrophy"],
            "cfp_drusen": ["drusen"],
            "cfp_scar": ["scar"],
            "cfp_cnv": ["CNV"],
            "cfp_rd": ["RD"],
            "cfp_mh": ["mh"],
            "cfp_membrane": ["membrane"],
            "cfp_edema": ["edema"],
            "cfp_artifact": ["artifact"],
            "oct_layer": [
                "optic disc","RNFL","GCL","IPL","INL","OPL","ONL","ELM","ISOS","RPE","choroidal","chostro","fovea"
            ],
            "oct_PMchovefosclera": ["fovea", "choroidal", "chostro", "sclera"],
            "oct_lesion": ["MH", "fluid", "PED"],
            "ffa_lesion": [
                "block","CNV","leakage","edema","WD","staining","disc","NPA","NV","laser","hypofluorescence","macula","MA"
            ],
        }
        labels = labels_map.get(task, [])
        return {
            "schema": (meta.get("io") or {}).get("output_schema", {}),
            "fields": {
                "counts": "map label -> count of connected components",
                "areas": "map label -> list of component areas in pixels^2",
                "output_paths": "generated visualization paths",
                "inference_time": "seconds",
            },
            "labels": labels,
        }

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
        before_mem = None
        before_mem_reserved = None
        after_mem = None
        after_mem_reserved = None
        cuda_avail = torch.cuda.is_available()
        device_name = None
        first_param_device = None
        param_count = None
        cuda_param_bytes = None
        if cuda_avail:
            try:  # pragma: no cover - depends on runtime GPU
                device_name = torch.cuda.get_device_name(0)
                before_mem = torch.cuda.memory_allocated(0)
                before_mem_reserved = torch.cuda.memory_reserved(0)
            except Exception:  # noqa: BLE001
                pass
        self.predictor = _load_predictor(self.model_id, self.weights_root)
        # Optionally perform a tiny warm tensor allocation to force context materialization
        if cuda_avail and hasattr(self.predictor, 'network'):  # nnUNet predictor has .network
            try:  # pragma: no cover
                p = next(self.predictor.network.parameters())
                first_param_device = str(p.device)
                # gather stats
                total = 0
                cuda_bytes = 0
                for t in self.predictor.network.parameters():
                    n = t.numel()
                    total += n
                    if t.is_cuda:
                        cuda_bytes += n * t.element_size()
                param_count = total
                cuda_param_bytes = cuda_bytes
            except Exception:  # noqa
                pass
        if cuda_avail:
            try:  # pragma: no cover
                torch.cuda.synchronize()
                after_mem = torch.cuda.memory_allocated(0)
                after_mem_reserved = torch.cuda.memory_reserved(0)
            except Exception:  # noqa
                pass
        self._model_loaded = True
        self.model_loaded = True
        self._load_telemetry = {
            "cuda": cuda_avail,
            "device_name": device_name,
            "mem_before": before_mem,
            "mem_after": after_mem,
            "mem_before_reserved": before_mem_reserved,
            "mem_after_reserved": after_mem_reserved,
            "mem_delta": (after_mem - before_mem) if (after_mem is not None and before_mem is not None) else None,
            "reserved_delta": (after_mem_reserved - before_mem_reserved) if (after_mem_reserved is not None and before_mem_reserved is not None) else None,
            "first_param_device": first_param_device,
            "param_count": param_count,
            "cuda_param_bytes": cuda_param_bytes,
        }

    def ensure_model_loaded(self):
        if not self.model_loaded:
            self.load_model()
        return self.model_loaded

    def warmup(self):  # minimal dummy inference to force GPU allocations
        if not self.model_loaded:
            self.load_model()
        try:
            import torch
            import tempfile, os
            # create a tiny fake image (RGB 64x64) and run through the same file-based path
            dummy = np.zeros((64, 64, 3), dtype=np.uint8)
            with tempfile.TemporaryDirectory() as td:
                img_path = Path(td) / "dummy.png"
                import cv2 as _cv2
                _cv2.imwrite(str(img_path), dummy)
                # Use existing predict pipeline (will copy & produce outputs)
                self.predict({"inputs": {"image_path": str(img_path)}})
            # collect post-warmup memory stats
            cuda_avail = torch.cuda.is_available()
            after_alloc = after_res = None
            if cuda_avail:
                try:  # pragma: no cover
                    after_alloc = torch.cuda.memory_allocated(0)
                    after_res = torch.cuda.memory_reserved(0)
                except Exception:  # noqa
                    pass
            self._load_telemetry["warmup_memory_allocated"] = after_alloc
            self._load_telemetry["warmup_memory_reserved"] = after_res
            self._load_telemetry["warmed_up"] = True
            return {
                "warmed_up": True,
                "warmup_memory_allocated": after_alloc,
                "warmup_memory_reserved": after_res,
            }
        except Exception as e:  # noqa
            self._load_telemetry["warmed_up"] = False
            self._load_telemetry["warmup_error"] = str(e)
            return {"warmed_up": False, "error": str(e)}

    # Request format: {"inputs": {"image_path": str}}
    def predict(self, request: Dict[str, Any]):
        inputs = request.get("inputs") or request
        image_path = inputs.get("image_path")
        if not image_path:
            raise ValueError("image_path missing in request.inputs")
        image_path = str(image_path)
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        # ensure model
        self.ensure_model_loaded()
        # prepare dirs
        self.prepare()
        images_dir = self.temp_dir / "images"
        pred_dir = self.temp_dir / "pred"
        # clean old predictions to avoid stale picks
        for pattern in ('*.png', '*.nii', '*.nii.gz'):
            for old in pred_dir.glob(pattern):
                try:
                    old.unlink()
                except Exception:
                    pass
        # copy image into workspace
        dst = images_dir / Path(image_path).name
        try:
            import shutil
            shutil.copy(image_path, dst)
        except Exception:
            dst = Path(image_path)
        start = time.time()
        # run predictor
        try:
            self.predictor.predict_from_files_sequential(
                [[str(dst)]],
                str(pred_dir),
                False,
                True,
                None,
            )
        except Exception:
            # swallow and try to detect outputs; inference may still have produced artifacts
            pass
        # locate segmentation result
        stem = Path(image_path).stem
        seg_path = pred_dir / f"{stem}.png"
        seg = None
        if seg_path.exists():
            seg = cv2.imread(str(seg_path), cv2.IMREAD_GRAYSCALE)
        # any fresh PNG (ignore synthetic) produced after start
        if seg is None:
            pngs = [p for p in pred_dir.rglob('*.png') if not p.name.endswith('_synthetic.png') and p.stat().st_mtime >= (start - 1.0)]
            if pngs:
                pngs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                preferred = [p for p in pngs if stem in p.stem]
                candidate = preferred[0] if preferred else pngs[0]
                seg = cv2.imread(str(candidate), cv2.IMREAD_GRAYSCALE)
                seg_path = candidate
        # try nii.gz -> png conversion
        if seg is None:
            niis = [p for p in pred_dir.rglob('*.nii.gz') if p.stat().st_mtime >= (start - 1.0)]
            if niis:
                try:
                    import nibabel as nib  # type: ignore
                    nii = nib.load(str(niis[0]))
                    data = nii.get_fdata().astype('uint8').squeeze()
                    seg_path = pred_dir / f"{niis[0].stem}.png"
                    cv2.imwrite(str(seg_path), data)
                    seg = data
                except Exception:
                    seg = None
        warning_msg: Optional[str] = None
        if seg is None:
            if not self.allow_fallback:
                listing = [str(p.relative_to(pred_dir)) for p in pred_dir.rglob('*')]
                raise RuntimeError(
                    "Segmentation output missing; real inference required. "
                    f"Tried: {Path(image_path).stem}.png / *.png / *.nii.gz under {pred_dir}. "
                    f"Please verify weights for model_id={self.model_id} exist under {self.weights_root} and nnUNet ran successfully. "
                    f"Current pred dir listing: {listing}"
                )
            # fallback synthetic mask
            src = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if src is not None:
                h, w = src.shape[:2]
                yy, xx = np.ogrid[:h, :w]
                cy, cx = h // 2, w // 2
                r = max(4, min(h, w) // 6)
                seg = (((yy - cy) ** 2 + (xx - cx) ** 2) <= r * r).astype('uint8') * 255
                seg_path = pred_dir / f"{stem}_synthetic.png"
                cv2.imwrite(str(seg_path), seg)
                warning_msg = "Segmentation output missing. Generated synthetic mask."
            else:
                raise RuntimeError("Segmentation output missing and source image unreadable.")
        # at this point seg is a 2D array
        uniques = np.unique(seg)
        if set(uniques.tolist()) <= {0, 255}:
            seg = (seg > 0).astype(np.uint8)
        orig = cv2.imread(image_path)
        h, w = orig.shape[:2]
        seg = cv2.resize(seg, (w, h), interpolation=cv2.INTER_NEAREST)
        colorized, stats = _colorize(seg, self.lesions, self.min_object_size)
        merged = np.hstack([orig, colorized])
        out_name = Path(image_path).name
        cv2.imwrite(str(self.temp_dir / "merge" / out_name), merged)
        cv2.imwrite(str(self.temp_dir / "rgb" / out_name), colorized)
        overlay = orig.copy()
        overlay[seg > 0] = 0
        overlay = cv2.addWeighted(overlay, 1, colorized, 1, 0)
        cv2.imwrite(str(self.temp_dir / "overlay" / out_name), overlay)
        latency = time.time() - start
        result_obj = {
            "task": self.task,
            **stats,
            "output_paths": {
                "merged": str(self.temp_dir / "merge" / out_name),
                "colorized": str(self.temp_dir / "rgb" / out_name),
                "overlay": str(self.temp_dir / "overlay" / out_name),
            },
            "inference_time": round(latency, 3),
        }
        if warning_msg:
            result_obj["warning"] = warning_msg
        return result_obj


def load_tool(task: str, **kw):
    return SegmentationTool({"id": f"segmentation:{task}", "entry": "tool_impl:SegmentationTool"}, {"task": task, **kw})

__all__ = ["SegmentationTool", "load_tool"]
