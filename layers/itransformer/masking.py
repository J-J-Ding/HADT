# Adapted from thuml/iTransformer @ c2426e68ca13f74aaec08045c5c724d8ad328124
# Upstream: utils/masking.py; license: third_party/licenses/itransformer-LICENSE
import torch

class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask
