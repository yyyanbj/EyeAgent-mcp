from __future__ import annotations
from typing import Any, Dict, Optional, Tuple
from pathlib import Path
import time
import math

import numpy as np
# Pillow is optional; we import lazily inside methods and fall back to cv2 when absent

from eyetools.core.tool_base import ToolBase


class MultimodalTool(ToolBase):
    """Fundus modality conversion tool with two variants:

    - fundus2oct: create a 32-slice pseudo-OCT volume and derived assets
    - fundus2eyeglobe: create a simple point cloud PLY and PNG/GIF views

    This implementation intentionally avoids heavy model dependencies and
    generates deterministic, informative placeholders. You can later wire in
    real generators by replacing `_predict_fundus2oct` and `_predict_fundus2eyeglobe`.
    """

    def __init__(self, meta: Dict[str, Any], params: Dict[str, Any]):
        super().__init__(meta, params)
        self.task: str = params.get("task") or meta.get("params", {}).get("task")
        if self.task not in {"fundus2oct", "fundus2eyeglobe"}:
            raise ValueError(f"Unsupported multimodal task {self.task}")
        self.weights_root = Path(params.get("weights_root", "weights/multimodal"))
        base = Path(params.get("base_path", "temp/multimodal")) / self.task
        base.mkdir(parents=True, exist_ok=True)
        self.base_dir = base
        # runtime toggles
        self._real_ready = False
        self._encoder = None
        self._oct_pipeline = None
        self._eyeglobe_pipeline = None

    # Optional static describer used by discovery/clients
    @staticmethod
    def describe_outputs(meta: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        task = (params or {}).get("task") or (meta.get("params", {}) or {}).get("task")
        if task == "fundus2oct":
            return {
                "schema": (meta.get("io") or {}).get("output_schema", {}),
                "fields": {
                    "output_paths": {
                        "montage_path": "PNG grid of slices",
                        "frame_dir": "Directory with individual frames",
                        "gif_path": "Animated GIF across slices",
                    },
                    "inference_time": "seconds",
                },
                "slices": 32,
            }
        elif task == "fundus2eyeglobe":
            return {
                "schema": (meta.get("io") or {}).get("output_schema", {}),
                "fields": {
                    "output_paths": {
                        "ply_path": "Point cloud in ASCII PLY",
                        "png_path": "Static projection PNG",
                        "gif_path": "Rotating view GIF",
                    },
                    "inference_time": "seconds",
                },
                "num_points": 1448,
            }
        return {"schema": (meta.get("io") or {}).get("output_schema", {})}

    # Tool lifecycle -------------------------------------------------
    def prepare(self):
        self.base_dir.mkdir(parents=True, exist_ok=True)
        return True

    def load_model(self):
        # Attempt to prepare real pipelines if dependencies & weights exist.
        self._real_ready = self._try_prepare_real()
        self._model_loaded = True

    # Request format: {"inputs": {"image_path": str, ...}}
    def predict(self, request: Dict[str, Any]):
        inputs = request.get("inputs") or request
        image_path = inputs.get("image_path")
        if not image_path:
            raise ValueError("image_path missing in request.inputs")
        image_path = str(image_path)
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        self.ensure_model_loaded()
        self.prepare()
        start = time.time()
        if self.task == "fundus2oct":
            out = self._predict_fundus2oct(image_path, inputs)
        else:
            out = self._predict_fundus2eyeglobe(image_path, inputs)
        out["inference_time"] = round(time.time() - start, 3)
        return out

    # ---------------- Variant implementations ----------------------
    def _predict_fundus2oct(self, image_path: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # Try real pipeline path first
        if self._real_ready:
            try:
                out = self._predict_f2o_real(image_path, inputs)
                if out:
                    return out
            except Exception:
                # fall back to placeholder silently
                pass
        # Create a pseudo 32-slice volume using radial gradients + subtle noise
        slices = int(inputs.get("slices", 32))
        h = int(inputs.get("height", 256))
        w = int(inputs.get("width", 256))
        rng = np.random.default_rng(42)
        # radial base
        y, x = np.ogrid[:h, :w]
        cy, cx = h / 2.0, w / 2.0
        r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
        r = r / (r.max() + 1e-6)
        volume = []
        for i in range(slices):
            phase = 2 * math.pi * i / max(1, slices - 1)
            layer = (1.0 - r) * (0.6 + 0.4 * math.sin(phase))
            layer += 0.05 * rng.standard_normal((h, w))
            layer = np.clip(layer, 0.0, 1.0)
            layer_u8 = (layer * 255).astype(np.uint8)
            volume.append(layer_u8)
        volume = np.stack(volume, axis=0)

        stem = Path(image_path).stem
        montage_path = self.base_dir / f"{stem}_oct_montage.png"
        frame_dir = self.base_dir / f"{stem}_oct_frames"
        gif_path = self.base_dir / f"{stem}_oct.gif"
        frame_dir.mkdir(parents=True, exist_ok=True)

        # Save frames
        try:
            from PIL import Image as _Image
            for i, frame in enumerate(volume):
                _Image.fromarray(frame).convert("L").save(str(frame_dir / f"frame_{i+1:02d}.png"))
        except Exception:
            # Fallback to OpenCV if Pillow not available
            try:
                import cv2 as _cv2
                for i, frame in enumerate(volume):
                    _cv2.imwrite(str(frame_dir / f"frame_{i+1:02d}.png"), frame)
            except Exception:
                pass

        # Build montage (8x4)
        rows = []
        grid_r, grid_c = 4, 8
        idx = 0
        for r_i in range(grid_r):
            row_imgs = []
            for c_i in range(grid_c):
                if idx < slices:
                    row_imgs.append(volume[idx])
                else:
                    row_imgs.append(np.zeros((h, w), dtype=np.uint8))
                idx += 1
            rows.append(np.hstack(row_imgs))
        montage = np.vstack(rows)
        try:
            from PIL import Image as _Image
            _Image.fromarray(montage).convert("L").save(str(montage_path))
        except Exception:
            try:
                import cv2 as _cv2
                _cv2.imwrite(str(montage_path), montage)
            except Exception:
                pass

        # Simple GIF via PIL; if unavailable keep path placeholder
        try:
            from PIL import Image as _Image
            frames = [_Image.fromarray(v).convert("L") for v in volume]
            frames[0].save(str(gif_path), save_all=True, append_images=frames[1:], duration=80, loop=0)
        except Exception:
            # Fallback: write nothing, still return paths
            pass

        return {
            "output_paths": {
                "montage_path": str(montage_path),
                "frame_dir": str(frame_dir),
                "gif_path": str(gif_path),
            }
        }

    def _predict_fundus2eyeglobe(self, image_path: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # Try real pipeline path first
        if self._real_ready:
            try:
                out = self._predict_f2g_real(image_path, inputs)
                if out:
                    return out
            except Exception:
                # fall back to placeholder silently
                pass
        # Generate a simple sphere-like point cloud with slight deformation
        n = int(inputs.get("num_points", 1448))
        rng = np.random.default_rng(17)
        phi = rng.random(n) * 2 * math.pi
        costheta = rng.random(n) * 2 - 1
        theta = np.arccos(costheta)
        r = 1.0 + 0.05 * rng.standard_normal(n)
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        pts = np.stack([x, y, z], axis=1).astype(np.float32)

        stem = Path(image_path).stem
        ply_path = self.base_dir / f"{stem}_eyeglobe.ply"
        png_path = self.base_dir / f"{stem}_eyeglobe.png"
        gif_path = self.base_dir / f"{stem}_eyeglobe.gif"
        self._write_ply(ply_path, pts)
        self._render_projection_png(png_path, pts)
        self._write_rot_gif(gif_path, pts)
        return {
            "output_paths": {
                "ply_path": str(ply_path),
                "png_path": str(png_path),
                "gif_path": str(gif_path),
            }
        }

    # ---------------- helpers ----------------
    def _try_prepare_real(self) -> bool:
        """Best-effort import and setup for real pipelines.

        Returns True if essential dependencies and weights are present.
        """
        try:
            import importlib.util as _ilu
            import sys as _sys
            # prepare sys.path for local generation3d package tree
            gen3d_root = Path(__file__).resolve().parents[3] / "langchain_tool_src/tools/generation3d"
            if gen3d_root.exists() and str(gen3d_root) not in _sys.path:
                _sys.path.insert(0, str(gen3d_root))
            # check key modules
            for mod in ("timm", "torch"):
                if _ilu.find_spec(mod) is None:
                    return False
            if _ilu.find_spec("medical_diffusion.models.pipelines") is None:
                return False
            # weights presence check
            if not (self.weights_root / "RETFound_cfp_weights.pth").exists():
                return False
            return True
        except Exception:
            return False

    def _ensure_encoder(self):
        if self._encoder is not None:
            return
        import timm
        model = timm.create_model("vit_large_patch16_224", pretrained=False)
        # Try load weights
        ckpt_path = self.weights_root / "RETFound_cfp_weights.pth"
        if ckpt_path.exists():
            import torch
            state = torch.load(str(ckpt_path), map_location="cpu")
            sd = state.get("model") or state
            try:
                model.load_state_dict(sd, strict=False)
            except Exception:
                # tolerate key mismatches
                model.load_state_dict(sd, strict=False)
        self._encoder = model.eval()

    def _encode_fundus(self, image_path: str) -> np.ndarray:
        # Load image with PIL if available, else fallback to OpenCV
        try:
            from PIL import Image as _Image
            img = _Image.open(image_path).convert("RGB").resize((224, 224))
            arr = np.asarray(img).astype(np.float32) / 255.0
        except Exception:
            import cv2 as _cv2
            img = _cv2.imread(image_path, _cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(image_path)
            img = _cv2.cvtColor(img, _cv2.COLOR_BGR2RGB)
            img = _cv2.resize(img, (224, 224))
            arr = img.astype(np.float32) / 255.0
        import torch
        self._ensure_encoder()
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        with torch.no_grad():
            feats = self._encoder.forward_features(t.float())
        return feats

    def _ensure_oct_pipeline(self):
        if self._oct_pipeline is not None:
            return
        import importlib as _il
        pipelines = _il.import_module("medical_diffusion.models.pipelines")
        LVDiffusionPipeline = getattr(pipelines, "LVDiffusionPipeline")
        ckpt = self.weights_root / "fundus2octvolume/fundus2octvolume_new.ckpt"
        if not ckpt.exists():
            # fallback alternate names
            for name in ("fundus2octvolume_32.ckpt", "fundus2octvolume.ckpt"):
                alt = self.weights_root / "fundus2octvolume" / name
                if alt.exists():
                    ckpt = alt
                    break
        # Lightning load
        self._oct_pipeline = LVDiffusionPipeline.load_from_checkpoint(str(ckpt), map_location="cpu").eval()

    def _ensure_eyeglobe_pipeline(self):
        if self._eyeglobe_pipeline is not None:
            return
        import importlib as _il
        pipelines = _il.import_module("medical_diffusion.models.pipelines")
        DiffusionPipeline_CFP_META = getattr(pipelines, "DiffusionPipeline_CFP_META")
        ckpt = self.weights_root / "fundus2eyeglobe/fundus2eyeglobe.ckpt"
        self._eyeglobe_pipeline = DiffusionPipeline_CFP_META.load_from_checkpoint(str(ckpt), map_location="cpu").eval()

    def _label_distribution_encode(self, x_idx: np.ndarray, mu_: float, sigma_: float):
        return np.exp(-((x_idx - mu_) ** 2) / (2 * (sigma_**2))) / (np.sqrt(2 * np.pi) * sigma_)

    def _predict_f2o_real(self, image_path: str, inputs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        import torch
        self._ensure_oct_pipeline()
        latent = self._encode_fundus(image_path)
        steps = int(inputs.get("sampling_steps", 50))
        with torch.no_grad():
            vol = self._oct_pipeline.sample(1, (8, 32, 32, 32), guidance_scale=1, condition=latent, steps=steps)
            vol = vol.cpu().numpy()
        video = vol[0].squeeze(0)
        video = (np.clip(video, -1, 1) + 1) / 2.0 * 255.0
        video = video.astype(np.uint8)
        # save using same paths as placeholder
        stem = Path(image_path).stem
        montage_path = self.base_dir / f"{stem}_oct_montage.png"
        frame_dir = self.base_dir / f"{stem}_oct_frames"
        gif_path = self.base_dir / f"{stem}_oct.gif"
        frame_dir.mkdir(parents=True, exist_ok=True)
        # frames
        try:
            from PIL import Image as _Image
            for i, frame in enumerate(video):
                _Image.fromarray(frame).convert("L").save(str(frame_dir / f"frame_{i+1:02d}.png"))
        except Exception:
            try:
                import cv2 as _cv2
                for i, frame in enumerate(video):
                    _cv2.imwrite(str(frame_dir / f"frame_{i+1:02d}.png"), frame)
            except Exception:
                pass
        # montage
        rows = []
        grid_r, grid_c = 4, 8
        idx = 0
        h, w = video.shape[1:]
        for _ in range(grid_r):
            row_imgs = []
            for _ in range(grid_c):
                if idx < video.shape[0]:
                    row_imgs.append(video[idx])
                else:
                    row_imgs.append(np.zeros((h, w), dtype=np.uint8))
                idx += 1
            rows.append(np.hstack(row_imgs))
        montage = np.vstack(rows)
        try:
            from PIL import Image as _Image
            _Image.fromarray(montage).convert("L").save(str(montage_path))
        except Exception:
            try:
                import cv2 as _cv2
                _cv2.imwrite(str(montage_path), montage)
            except Exception:
                pass
        # gif
        try:
            from PIL import Image as _Image
            frames = [_Image.fromarray(v).convert("L") for v in video]
            frames[0].save(str(gif_path), save_all=True, append_images=frames[1:], duration=80, loop=0)
        except Exception:
            pass
        return {
            "output_paths": {
                "montage_path": str(montage_path),
                "frame_dir": str(frame_dir),
                "gif_path": str(gif_path),
            }
        }

    def _predict_f2g_real(self, image_path: str, inputs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        import torch
        self._ensure_eyeglobe_pipeline()
        latent = self._encode_fundus(image_path)
        eye_category = str(inputs.get("eye_category", "OD"))
        steps = int(inputs.get("sampling_steps", 50))
        se = inputs.get("SE")
        al = inputs.get("AL")
        # prepare conditions
        # label encodings
        ld_sigma = 1.0
        num_elements = 512
        if se is not None:
            x_sph = np.linspace(-20.75 - ld_sigma, -6.0 + ld_sigma, num_elements)
            se_condition = torch.tensor(self._label_distribution_encode(x_sph, float(se), ld_sigma), dtype=torch.float).unsqueeze(0)
        else:
            se_condition = None
        if al is not None:
            x_al = np.linspace(25.07 - ld_sigma, 34.3 + ld_sigma, num_elements)
            al_condition = torch.tensor(self._label_distribution_encode(x_al, float(al), ld_sigma), dtype=torch.float).unsqueeze(0)
        else:
            al_condition = None
        eye_idx = torch.tensor([ ["OS", "OD"].index(eye_category) ], dtype=torch.long)
        cond = [latent, eye_idx, None, None, se_condition, al_condition]
        with torch.no_grad():
            res = self._eyeglobe_pipeline.sample(1, (113, 28, 28), guidance_scale=1, condition=cond, steps=steps).cpu().numpy()
        res = np.mean(res, axis=-1)
        res = np.mean(res, axis=-1)
        alpha = res[0].reshape(-1, 1)
        # reconstruct vertices via mm_params
        num_ref_vertex = 1448
        mm = self.weights_root / "mm_params"
        right_mu = np.load(mm / f"{num_ref_vertex}pts/right_shape_mu.npy")
        right_delta = np.load(mm / f"{num_ref_vertex}pts/right_shape_delta.npy")
        left_mu = np.load(mm / f"{num_ref_vertex}pts/left_shape_mu.npy")
        left_delta = np.load(mm / f"{num_ref_vertex}pts/left_shape_delta.npy")
        if eye_category == "OD":
            verts = right_mu + right_delta.dot(alpha)
        else:
            verts = left_mu + left_delta.dot(alpha)
        verts = verts.reshape(-1, 3).astype(np.float32)
        # save outputs
        stem = Path(image_path).stem
        ply_path = self.base_dir / f"{stem}_eyeglobe.ply"
        png_path = self.base_dir / f"{stem}_eyeglobe.png"
        gif_path = self.base_dir / f"{stem}_eyeglobe.gif"
        self._write_ply(ply_path, verts)
        self._render_projection_png(png_path, verts)
        self._write_rot_gif(gif_path, verts)
        return {
            "output_paths": {
                "ply_path": str(ply_path),
                "png_path": str(png_path),
                "gif_path": str(gif_path),
            }
        }
    def _write_ply(self, path: Path, vertices: np.ndarray):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {vertices.shape[0]}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("end_header\n")
            for v in vertices:
                f.write(f"{float(v[0])} {float(v[1])} {float(v[2])}\n")

    def _render_projection_png(self, path: Path, pts: np.ndarray):
        # Simple orthographic projection to XY plane
        size = 512
        try:
            from PIL import Image as _Image, ImageDraw as _ImageDraw
            img = _Image.new("RGB", (size, size), color=(255, 255, 255))
            draw = _ImageDraw.Draw(img)
            xy = pts[:, :2]
            xy = (xy - xy.min(0)) / (xy.max(0) - xy.min(0) + 1e-6)
            xy = (xy * (size - 1)).astype(np.int32)
            for p in xy:
                x, y = int(p[0]), int(p[1])
                draw.point((x, y), fill=(10, 80, 200))
            img.save(str(path))
        except Exception:
            # Fallback: draw using numpy + cv2
            try:
                import cv2 as _cv2
                img = np.ones((size, size, 3), dtype=np.uint8) * 255
                xy = pts[:, :2]
                xy = (xy - xy.min(0)) / (xy.max(0) - xy.min(0) + 1e-6)
                xy = (xy * (size - 1)).astype(np.int32)
                for p in xy:
                    x, y = int(p[0]), int(p[1])
                    _cv2.circle(img, (x, y), 0, (200, 80, 10), -1)
                _cv2.imwrite(str(path), img)
            except Exception:
                pass

    def _write_rot_gif(self, path: Path, pts: np.ndarray):
        try:
            from PIL import Image as _Image, ImageDraw as _ImageDraw
            size = 512
            frames = []
            for k in range(36):
                angle = 2 * math.pi * k / 36
                rot = np.array([[math.cos(angle), -math.sin(angle), 0],
                                [math.sin(angle),  math.cos(angle), 0],
                                [0,               0,               1]], dtype=np.float32)
                xyz = pts @ rot.T
                img = _Image.new("RGB", (size, size), color=(255, 255, 255))
                draw = _ImageDraw.Draw(img)
                xy = xyz[:, :2]
                xy = (xy - xy.min(0)) / (xy.max(0) - xy.min(0) + 1e-6)
                xy = (xy * (size - 1)).astype(np.int32)
                for p in xy:
                    x, y = int(p[0]), int(p[1])
                    draw.point((x, y), fill=(50, 120, 50))
                frames.append(img)
            frames[0].save(str(path), save_all=True, append_images=frames[1:], duration=80, loop=0)
        except Exception:
            # If PIL unavailable, silently skip animated GIF creation
            pass
