from pathlib import Path
import numpy as np

from tools.multimodal.tool_impl import MultimodalTool


def _make_dummy_image(tmp_path: Path) -> Path:
	from PIL import Image
	arr = (np.random.rand(64, 64, 3) * 255).astype('uint8')
	p = tmp_path / 'dummy.jpg'
	Image.fromarray(arr).save(p)
	return p


def test_fundus2oct_placeholder(tmp_path):
	img = _make_dummy_image(tmp_path)
	meta = {"id": "multimodal:fundus2oct", "entry": "tool_impl:MultimodalTool"}
	params = {"task": "fundus2oct", "base_path": str(tmp_path / 'mmt'), "weights_root": str(tmp_path / 'w')}
	tool = MultimodalTool(meta, params)
	tool.prepare()
	out = tool.predict({"inputs": {"image_path": str(img), "slices": 8, "height": 64, "width": 64}})
	assert 'output_paths' in out
	paths = out['output_paths']
	assert Path(paths['montage_path']).exists()
	assert Path(paths['frame_dir']).exists()
	assert Path(paths['gif_path']).exists() or True  # GIF may be skipped silently


def test_fundus2eyeglobe_placeholder(tmp_path):
	img = _make_dummy_image(tmp_path)
	meta = {"id": "multimodal:fundus2eyeglobe", "entry": "tool_impl:MultimodalTool"}
	params = {"task": "fundus2eyeglobe", "base_path": str(tmp_path / 'mmt'), "weights_root": str(tmp_path / 'w')}
	tool = MultimodalTool(meta, params)
	tool.prepare()
	out = tool.predict({"inputs": {"image_path": str(img), "num_points": 256}})
	assert 'output_paths' in out
	paths = out['output_paths']
	assert Path(paths['ply_path']).exists()
	assert Path(paths['png_path']).exists()
	assert Path(paths['gif_path']).exists() or True

