import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MedianAnchoredNonCrossingHead(nn.Module):
    """
    Output order:
    q05 < q10 < q25 < q50 < q75 < q90 < q95
    by construction, using positive cumulative increments.
    """

    def __init__(self, d_model=128, min_gap=1e-4, initial_gap=0.55):
        super().__init__()
        self.projection = nn.Linear(d_model, 7)
        self.min_gap = float(min_gap)

        # Small non-zero weights allow gradients to reach the backbone
        # from the first optimization step.
        nn.init.normal_(self.projection.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.projection.bias)

        inverse_softplus = math.log(math.expm1(float(initial_gap)))
        with torch.no_grad():
            self.projection.bias[1:7].fill_(inverse_softplus)

    def forward(self, features):
        # features: [BC,H,D]
        raw = self.projection(features)

        median = raw[..., 0:1]
        lower_gaps = F.softplus(raw[..., 1:4]) + self.min_gap
        upper_gaps = F.softplus(raw[..., 4:7]) + self.min_gap

        # Cumulative lower distances initially ordered q25,q10,q05;
        # reverse to final ascending quantile order q05,q10,q25.
        lower = (
            median - torch.cumsum(lower_gaps, dim=-1)
        ).flip(dims=[-1])

        upper = median + torch.cumsum(upper_gaps, dim=-1)

        return torch.cat([lower, median, upper], dim=-1)
