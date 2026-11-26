import math
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


def derive_n_points(n_patches):
    """Frozen auto rule: K=min(N,max(4,min(8,ceil(N/5))))."""
    import math
    n = int(n_patches)
    return min(n, max(4, min(8, math.ceil(n / 5))))


class HorizonDeformableAttention(nn.Module):
    """
    Horizon-conditioned deformable cross-attention.

    Input history features: [BC,N,D]
    Output future features: [BC,H,D]

    The horizon dimension is chunked. Advanced indexing gathers only the
    sampled K/V values; it does not expand the complete N-token sequence
    across all H prediction steps.
    """

    def __init__(
        self,
        d_model=128,
        n_heads=8,
        n_points=8,
        max_horizon=720,
        max_offset=1.5,
        horizon_chunk_size=64,
        dropout=0.1,
        gradient_checkpointing=True,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")

        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.head_dim = self.d_model // self.n_heads
        self.n_points = int(n_points)
        self.max_offset = float(max_offset)
        self.horizon_chunk_size = int(horizon_chunk_size)
        self.gradient_checkpointing = bool(gradient_checkpointing)

        self.horizon_embedding = nn.Embedding(int(max_horizon), self.d_model)
        self.context_projection = nn.Linear(self.d_model, self.d_model)
        self.query_norm = nn.LayerNorm(self.d_model)

        self.q_projection = nn.Linear(self.d_model, self.d_model)
        self.k_projection = nn.Linear(self.d_model, self.d_model)
        self.v_projection = nn.Linear(self.d_model, self.d_model)

        self.offset_projection = nn.Linear(
            self.d_model, self.n_heads * self.n_points
        )
        nn.init.zeros_(self.offset_projection.weight)
        nn.init.zeros_(self.offset_projection.bias)

        self.output_projection = nn.Linear(self.d_model, self.d_model)
        self.norm1 = nn.LayerNorm(self.d_model)
        self.norm2 = nn.LayerNorm(self.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model * 2, self.d_model),
        )
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _linear_sample(features, positions):
        """
        features:  [BC,N,M,Dh]
        positions: [BC,Hc,M,K]
        returns:   [BC,Hc,M,K,Dh]

        Advanced indexing materializes only sampled features, not a
        [BC,Hc,M,N,Dh] expanded copy of the full sequence.
        """
        bc, n_tokens, n_heads, head_dim = features.shape
        left = torch.floor(positions).long().clamp(0, n_tokens - 1)
        right = (left + 1).clamp(max=n_tokens - 1)
        weight = (positions - left.to(positions.dtype)).unsqueeze(-1)

        batch_index = torch.arange(
            bc, device=features.device
        ).view(bc, 1, 1, 1)
        head_index = torch.arange(
            n_heads, device=features.device
        ).view(1, 1, n_heads, 1)

        f_left = features[batch_index, left, head_index]
        f_right = features[batch_index, right, head_index]
        return (1.0 - weight) * f_left + weight * f_right

    def _forward_chunk(self, query_base, key, value, reference, disable_offsets=False):
        # query_base: [BC,Hc,D]
        bc, h_chunk, _ = query_base.shape
        m = self.n_heads
        k_points = self.n_points
        dh = self.head_dim

        query = self.q_projection(query_base).view(
            bc, h_chunk, m, dh
        )

        offsets = self.max_offset * torch.tanh(
            self.offset_projection(query_base)
        ).view(bc, h_chunk, m, k_points)

        if disable_offsets:
            offsets = offsets * 0.0

        positions = (reference + offsets).clamp(
            min=0.0, max=float(key.shape[1] - 1)
        )

        sampled_key = self._linear_sample(key, positions)
        sampled_value = self._linear_sample(value, positions)

        scores = (query.unsqueeze(-2) * sampled_key).sum(dim=-1)
        scores = scores / math.sqrt(dh)
        attention = torch.softmax(scores, dim=-1)

        attended = (
            attention.unsqueeze(-1) * sampled_value
        ).sum(dim=-2)

        attended = attended.reshape(bc, h_chunk, self.d_model)
        attended = self.output_projection(attended)

        hidden = self.norm1(
            query_base + self.dropout(attended)
        )
        output = hidden + self.dropout(
            self.ffn(self.norm2(hidden))
        )
        return output, offsets, positions, attention

    def forward(self, history_features, pred_len, disable_offsets=False, return_attention=False):
        # history_features: [BC,N,D]
        bc, n_tokens, d_model = history_features.shape
        if d_model != self.d_model:
            raise ValueError("Unexpected encoder feature dimension.")
        if pred_len > self.horizon_embedding.num_embeddings:
            raise ValueError(
                f"pred_len={pred_len} exceeds max_horizon="
                f"{self.horizon_embedding.num_embeddings}."
            )
        if self.n_points > n_tokens:
            raise ValueError("Sampling points K cannot exceed patch count N.")

        context = history_features.mean(dim=1)  # [BC,D]
        horizon_ids = torch.arange(
            pred_len, device=history_features.device
        )
        horizon_embed = self.horizon_embedding(horizon_ids)  # [H,D]
        query_base = self.query_norm(
            self.context_projection(context).unsqueeze(1)
            + horizon_embed.unsqueeze(0)
        )  # [BC,H,D]

        key = self.k_projection(history_features).view(
            bc, n_tokens, self.n_heads, self.head_dim
        )
        value = self.v_projection(history_features).view(
            bc, n_tokens, self.n_heads, self.head_dim
        )

        reference = torch.linspace(
            0,
            max(n_tokens - 1, 0),
            self.n_points,
            device=history_features.device,
            dtype=history_features.dtype,
        ).view(1, 1, 1, self.n_points)

        outputs, offsets_all, positions_all, attention_all = [], [], [], []
        chunk_size = max(1, self.horizon_chunk_size)

        for start in range(0, pred_len, chunk_size):
            end = min(start + chunk_size, pred_len)
            query_chunk = query_base[:, start:end, :]

            use_checkpoint = (
                self.gradient_checkpointing
                and self.training
                and torch.is_grad_enabled()
            )

            if use_checkpoint:
                out, offsets, positions, attention = checkpoint(
                    self._forward_chunk,
                    query_chunk,
                    key,
                    value,
                    reference,
                    disable_offsets,
                    use_reentrant=False,
                )
            else:
                out, offsets, positions, attention = self._forward_chunk(
                    query_chunk, key, value, reference, disable_offsets
                )

            outputs.append(out)
            offsets_all.append(offsets)
            positions_all.append(positions)
            if return_attention:
                attention_all.append(attention)

        result = (
            torch.cat(outputs, dim=1),
            torch.cat(offsets_all, dim=1),
            torch.cat(positions_all, dim=1),
        )

        if return_attention:
            return (*result, torch.cat(attention_all, dim=1))
        return result
