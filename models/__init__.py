# Models package - auto-discovers all models on first use

import torch
from .registry import create_model, list_models, get_model_config, get_model_info, print_model_catalog

# Import ablation models to register them
# from . import hssfn_ablations


class InputShapeWrapper(torch.nn.Module):
    """
    A wrapper to automatically reshape 5D HSI tensors to 4D for models
    that expect a standard 2D image input format.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # If input is 5D [B, 1, D, H, W], reshape to 4D [B, D, H, W]
        if x.dim() == 5 and x.shape[1] == 1:
            x = x.squeeze(1)
        return self.model(x)

    # Propagate attributes like `parameters` to the original model
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


__all__ = ['create_model', 'list_models', 'get_model_config', 'get_model_info',
           'print_model_catalog', 'InputShapeWrapper']
