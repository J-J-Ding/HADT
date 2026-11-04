import torch
import torch.nn as nn


class TemporalTransformerBlock(nn.Module):
    """Pre-LN Transformer block over the patch/time-token axis."""

    def __init__(self, d_model=128, n_heads=8, d_ff=256, dropout=0.1):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")

        self.norm1 = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        normalized = self.norm1(x)
        attn_out, _ = self.attention(
            normalized, normalized, normalized, need_weights=False
        )
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class TemporalTransformerEncoder(nn.Module):
    def __init__(
        self, d_model=128, n_heads=8, e_layers=3, d_ff=256, dropout=0.1
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            TemporalTransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
            )
            for _ in range(e_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)
