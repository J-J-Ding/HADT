"""Historical patch-direct control; not official PatchTST."""
from models.linear_nq import DirectQuantileControl

class PatchNQ(DirectQuantileControl):
    def __init__(self, backbone):
        super().__init__(backbone, patch=True)
