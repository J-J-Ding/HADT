import torch
import torch.nn as nn

from utils.revin import RevINNoAffine
from layers.hadt_nq.patch_embedding import OverlappingPatchEmbedding
from layers.hadt_nq.transformer_encoder import TemporalTransformerEncoder
from layers.hadt_nq.horizon_deformable import (
    HorizonDeformableAttention,
    derive_n_points,
)
from utils.quantile_heads import MedianAnchoredNonCrossingHead


class HADTNQ(nn.Module):
    """
    Input:
        x: [B,L,C]
    Output:
        quantiles: [B,H,C,7]

    Output values are in the globally standardized data scale because
    model-side RevIN is reversed before returning.
    """

    def __init__(
        self,
        seq_len,
        pred_len,
        patch_len=16,
        stride=8,
        d_model=128,
        n_heads=8,
        e_layers=3,
        d_ff=256,
        dropout=0.1,
        n_points="auto",
        max_offset=1.5,
        max_horizon=720,
        horizon_chunk_size=64,
        gradient_checkpointing=True,
        min_quantile_gap=1e-4,
        quantiles=(0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95),
    ):
        super().__init__()

        if tuple(float(q) for q in quantiles) != (
            0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95
        ):
            raise ValueError("HADT-NQ V1.1 uses the frozen seven-quantile set.")
        if pred_len > max_horizon:
            raise ValueError("pred_len cannot exceed max_horizon.")

        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.quantiles = tuple(quantiles)

        self.revin = RevINNoAffine()
        self.patch_embedding = OverlappingPatchEmbedding(
            seq_len=seq_len,
            patch_len=patch_len,
            stride=stride,
            d_model=d_model,
        )

        n_patches = self.patch_embedding.n_patches
        resolved_k = (
            derive_n_points(n_patches)
            if n_points == "auto"
            else int(n_points)
        )
        if resolved_k < 1 or resolved_k > n_patches:
            raise ValueError(
                f"Invalid K={resolved_k}; require 1 <= K <= N={n_patches}."
            )

        self.n_patches = n_patches
        self.n_points = resolved_k
        self.d_model = int(d_model)

        self.encoder = TemporalTransformerEncoder(
            d_model=d_model,
            n_heads=n_heads,
            e_layers=e_layers,
            d_ff=d_ff,
            dropout=dropout,
        )

        self.decoder = HorizonDeformableAttention(
            d_model=d_model,
            n_heads=n_heads,
            n_points=resolved_k,
            max_horizon=max_horizon,
            max_offset=max_offset,
            horizon_chunk_size=horizon_chunk_size,
            dropout=dropout,
            gradient_checkpointing=gradient_checkpointing,
        )

        self.quantile_head = MedianAnchoredNonCrossingHead(
            d_model=d_model,
            min_gap=min_quantile_gap,
        )

    def forward(self, x, return_aux=False, intervention=None):
        if x.ndim != 3:
            raise ValueError(f"Expected x:[B,L,C], got {tuple(x.shape)}")

        batch_size, seq_len, channels = x.shape
        if seq_len != self.seq_len:
            raise ValueError(
                f"Model expects L={self.seq_len}, received L={seq_len}."
            )

        intervention = intervention or getattr(self, "ablation", None)
        if intervention not in (None, "no_offsets", "no_encoder", "no_revin"):
            raise ValueError(f"Unknown intervention: {intervention}")
        x_norm, mean, std = self.revin.normalize(x)
        if intervention == "no_revin":
            x_norm = x
            mean = torch.zeros_like(mean)
            std = torch.ones_like(std)

        tokens, _, _ = self.patch_embedding(x_norm)
        encoded = tokens if intervention == "no_encoder" else self.encoder(tokens)

        decoder_output = self.decoder(
            encoded, pred_len=self.pred_len,
            disable_offsets=intervention == "no_offsets", return_attention=return_aux,
        )
        decoded, offsets, positions = decoder_output[:3]

        q_norm = self.quantile_head(decoded)  # [B*C,H,7]
        q_norm = q_norm.reshape(
            batch_size, channels, self.pred_len, 7
        ).permute(0, 2, 1, 3).contiguous()

        quantiles = self.revin.denormalize(q_norm, mean, std)

        if not return_aux:
            return quantiles

        offsets = offsets.reshape(
            batch_size, channels, self.pred_len,
            self.decoder.n_heads, self.n_points
        )
        positions = positions.reshape(
            batch_size, channels, self.pred_len,
            self.decoder.n_heads, self.n_points
        )
        aux = {
            "attention": decoder_output[3].reshape(
                batch_size, channels, self.pred_len, self.decoder.n_heads, self.n_points
            ),
            "normalized_input": x_norm,
            "tokens": tokens.reshape(batch_size, channels, self.n_patches, self.d_model),
            "encoded": encoded.reshape(batch_size, channels, self.n_patches, self.d_model),
            "offsets": offsets,
            "positions": positions,
            "n_patches": self.n_patches,
            "n_points": self.n_points,
        }
        return quantiles, aux
