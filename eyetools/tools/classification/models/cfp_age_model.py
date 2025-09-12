import torch
import torch.nn as nn
import timm
from typing import Optional

class CfpAgeModel(nn.Module):
    """Age regression model using timm res2net50d backbone."""
    def __init__(self, model_arch: str = "res2net50d", pretrained: bool = True):
        super().__init__()
        self.model = timm.create_model(model_arch, in_chans=3, num_classes=1, pretrained=pretrained)

    def forward(self, x):
        return self.model(x)

    def load_checkpoint(self, checkpoint_path: str, device: Optional[torch.device] = None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state = torch.load(checkpoint_path, map_location=device)
        if isinstance(state, dict) and 'state_dict' in state:
            state = state['state_dict']
        self.model.load_state_dict(state, strict=False)
        return None
