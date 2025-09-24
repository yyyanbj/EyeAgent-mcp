# Multimodal tool (fundus modality conversion)

This tool bundle exposes two variants under a single package:

- fundus2oct: generate a pseudo-OCT volume visualization (montage, frames, GIF)
- fundus2eyeglobe: generate a simple eye-globe point cloud PLY and PNG/GIF views

It is designed to be lightweight and to run in the shared `py312` environment without heavy 3D generation dependencies by default. You can later replace the placeholder generation with your model-powered pipelines.

Inputs
- image_path: path to a CFP image
- fundus2eyeglobe optional: eye_category (OD/OS), SE, AL
- fundus2oct optional: sampling_steps

Outputs
- output_paths: mapping of generated file paths (montage/frames/gifs/ply/png)
- inference_time: seconds

Config
See `config.yaml` for variants and environment.

Environment
- Uses `envs/py312-multimodal` by default (subprocess load). This isolates heavier deps (pytorch-lightning, einops, msssim) required by the real generation pipelines.
- If these dependencies or weights aren’t available, the tool falls back to a lightweight placeholder path that still produces artifacts (Pillow + NumPy).

Run a quick demo
- In-process (fast placeholder works everywhere):
	uv run python scripts/run_multimodal_demo.py --variant fundus2oct --image examples/test_images/retinal\ vessel.jpg --mode fallback
	uv run python scripts/run_multimodal_demo.py --variant fundus2eyeglobe --image examples/test_images/retinal\ vessel.jpg --mode fallback

- Via ToolManager (spawns subprocess using `envs/py12-multimodal`):
	uv run python scripts/run_multimodal_demo.py --variant fundus2oct --image examples/test_images/retinal\ vessel.jpg --manager
	uv run python scripts/run_multimodal_demo.py --variant fundus2eyeglobe --image examples/test_images/retinal\ vessel.jpg --manager

Outputs are written under `temp/demo_multimodal/<variant>/...` and include montage/frames/GIF (fundus2oct) or PLY/PNG/GIF (fundus2eyeglobe).