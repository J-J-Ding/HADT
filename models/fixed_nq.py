"""Historical zero-offset HADT control."""
from torch import nn


class FixedNQ(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.__dict__.update(backbone.__dict__)
        self.ablation='no_offsets'

    def forward(self, *args, **kwargs):
        from models.hadt_nq import HADTNQ
        return HADTNQ.forward(self, *args, **kwargs)
