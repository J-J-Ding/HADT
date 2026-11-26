import torch
import torch.nn as nn


class OverlappingPatchEmbedding(nn.Module):
    def __init__(self, seq_len, patch_len=16, stride=8, d_model=128):
        super().__init__()
        if patch_len <= 0 or stride <= 0 or patch_len > seq_len:
            raise ValueError("Require 0 < stride, 0 < patch_len <= seq_len.")

        self.seq_len = int(seq_len)
        self.patch_len = int(patch_len)
        self.stride = int(stride)
        self.n_patches = (self.seq_len - self.patch_len) // self.stride + 1

        self.projection = nn.Linear(self.patch_len, d_model)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, self.n_patches, d_model)
        )
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def forward(self, x):
        # x: [B,L,C]
        if x.ndim != 3:
            raise ValueError(f"Expected [B,L,C], got {tuple(x.shape)}")
        b, length, channels = x.shape
        if length != self.seq_len:
            raise ValueError(
                f"Configured L={self.seq_len}, but received L={length}."
            )

        # [B,C,L] -> [B,C,N,P]
        patches = x.transpose(1, 2).unfold(
            dimension=-1,
            size=self.patch_len,
            step=self.stride,
        )
        # [B,C,N,D]
        z = self.projection(patches)
        z = z + self.position_embedding[:, None, :, :]
        # Shared channel-independent encoder: [B*C,N,D]
        z = z.reshape(b * channels, self.n_patches, -1)
        return z, b, channels
