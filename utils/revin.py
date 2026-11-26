import torch
import torch.nn as nn


class RevINNoAffine(nn.Module):
    """Instance normalization over the history axis, with reversible statistics."""

    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = float(eps)

    def normalize(self, x):
        # x: [B,L,C]
        mean = x.mean(dim=1, keepdim=True).detach()
        var = x.var(dim=1, keepdim=True, unbiased=False).detach()
        std = torch.sqrt(var + self.eps)
        return (x - mean) / std, mean, std

    def denormalize(self, quantiles, mean, std):
        # quantiles: [B,H,C,Q], mean/std: [B,1,C]
        return quantiles * std.unsqueeze(-1) + mean.unsqueeze(-1)
