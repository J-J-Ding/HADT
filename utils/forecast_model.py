"""Task adapter: keep a native point model and append the existing ordered head."""
import torch
from torch import nn
from utils.quantile_heads import MedianAnchoredNonCrossingHead
from utils.losses import DEFAULT_QUANTILES


class PostPointQuantileModel(nn.Module):
    def __init__(self, point_model, configs, model_name, output_mode, multi_argument=False):
        super().__init__()
        if output_mode not in ('point', 'quantile'):
            raise ValueError('output_mode must be point or quantile')
        self.point_model = point_model
        self.seq_len, self.pred_len = configs.seq_len, configs.pred_len
        self.n_vars = configs.enc_in
        self.model_name, self.output_mode = model_name, output_mode
        self.multi_argument = multi_argument
        self.quantiles = tuple(DEFAULT_QUANTILES)
        self.quantile_adapter = 'post_point_shared_head' if output_mode == 'quantile' else None
        self.quantile_head = MedianAnchoredNonCrossingHead(d_model=1) if output_mode == 'quantile' else None
        # Default to full context. Only the stateless shared DLinear configuration
        # has been approved for arbitrary channel subsets and chunked inference.
        independent = model_name == 'dlinear' and not configs.individual
        self.supports_channel_subsets = independent
        self.supports_channel_chunking = independent
        self.default_channel_sampling = False
        self.supported_interventions = ()

    def forward(self, x, return_aux=False, intervention=None):
        if intervention is not None:
            raise ValueError(f'{self.model_name} does not support intervention={intervention}')
        if x.ndim != 3 or x.shape[1] != self.seq_len:
            raise ValueError(f'Expected [B,{self.seq_len},C], got {tuple(x.shape)}')
        if not self.supports_channel_subsets and x.shape[2] != self.n_vars:
            raise ValueError(f'{self.model_name} requires all {self.n_vars} variables in their saved order')
        # Some upstream normalization uses in-place subtraction/division. Protect
        # the caller's input (including diagnostic histories) without detaching it.
        p = self.point_model(x.clone(), None, None, None) if self.multi_argument else self.point_model(x.clone())
        if isinstance(p, tuple):
            p = p[0]
        b, h, c = p.shape
        if (h, c) != (self.pred_len, x.shape[2]):
            raise ValueError('Native point forecast has unexpected shape')
        if self.output_mode == 'point':
            prediction = p
        else:
            features = p.permute(0, 2, 1).reshape(b * c, h, 1)
            q = self.quantile_head(features)
            prediction = q.reshape(b, c, h, 7).permute(0, 2, 1, 3).contiguous()
        return (prediction, {}) if return_aux else prediction
