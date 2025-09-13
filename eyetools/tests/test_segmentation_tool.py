import os
from pathlib import Path
import pytest

from eyetools.core.registry import ToolRegistry
from eyetools.core.loader import discover_tools
from eyetools.core.tool_manager import ToolManager


def test_segmentation_config_discovery(tmp_path):
    reg = ToolRegistry()
    errors = discover_tools([Path('tools')], reg, [])
    assert not errors
    seg_variants = [m for m in reg.list() if m.package == 'segmentation']
    assert seg_variants, 'segmentation variants not discovered'
    # pick one variant
    sample = next(m for m in seg_variants if m.variant == 'cfp_artifact')
    assert sample.runtime.get('load_mode') == 'auto'


def test_segmentation_mock_predict(tmp_path, monkeypatch):
    """Lightweight test: monkeypatch SegmentationTool model load + predictor to avoid heavy nnUNet dependency."""
    # tools is a separate top-level package (not nested under eyetools)
    from tools.segmentation.tool_impl import SegmentationTool
    import numpy as np, cv2

    meta = {"id": "segmentation:cfp_artifact", "entry": "tool_impl:SegmentationTool"}
    params = {"task": "cfp_artifact", "base_path": str(tmp_path / 'segtemp'), "weights_root": str(tmp_path / 'w')}
    tool = SegmentationTool(meta, params)

    class DummyPredictor:
        def predict_from_files_sequential(self, inputs, out_dir, *a, **k):
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            tgt = Path(inputs[0][0])
            seg_path = Path(out_dir) / f"{tgt.stem}.png"
            dummy = (np.random.rand(32,32) > 0.7).astype('uint8')
            cv2.imwrite(str(seg_path), dummy*255)
            return True

    def fake_load():
        tool.lesions = {"artifact": (255,255,255)}
        tool.model_id = 0
        tool.predictor = DummyPredictor()
        tool._model_loaded = True

    monkeypatch.setattr(tool, 'load_model', fake_load)
    tool.prepare()
    img = (np.random.rand(64,64,3)*255).astype('uint8')
    img_path = tmp_path / 'input.png'
    cv2.imwrite(str(img_path), img)
    res = tool.predict({"inputs": {"image_path": str(img_path)}})
    assert res['task'] == 'cfp_artifact'
    assert 'output_paths' in res
    for k in ['merged','colorized','overlay']:
        assert Path(res['output_paths'][k]).exists()


@pytest.mark.skip(reason="Requires heavy nnUNet weights; skip in default CI")
def test_segmentation_inference_smoke():
    reg = ToolRegistry()
    discover_tools([Path('tools')], reg, [])
    tm = ToolManager(registry=reg, workspace_root=Path('.'))
    # find artifact variant
    meta = next(m for m in reg.list() if m.package == 'segmentation' and m.variant == 'cfp_artifact')
    # create dummy image
    import numpy as np, cv2
    img_path = 'dummy_seg.jpg'
    cv2.imwrite(img_path, (np.zeros((32,32,3))+255).astype('uint8'))
    try:
        with pytest.raises(FileNotFoundError):
            tm.predict(meta.id, {"inputs": {"image_path": img_path}})
    finally:
        if os.path.exists(img_path):
            os.remove(img_path)
