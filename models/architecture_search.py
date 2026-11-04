"""Explicit candidate architectures; selection never consumes test metrics."""
import copy
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from models.paper_baselines import DirectQuantileControl, initialize_raw_projection
from models.transformer_encoder import TemporalTransformerEncoder


def candidates():
    return [
        dict(id='transformer3', encoder='transformer', depth=3, decoder='deformable'),
        dict(id='identity', encoder='identity', depth=0, decoder='deformable'),
        dict(id='mlp3', encoder='mlp', depth=3, decoder='deformable'),
        dict(id='cnn3', encoder='cnn', depth=3, decoder='deformable'),
        dict(id='cnn_transformer', encoder='hybrid', depth=2, decoder='deformable'),
        dict(id='transformer1', encoder='transformer', depth=1, decoder='deformable'),
        dict(id='cnn64_lr3e4', encoder='cnn', depth=3, decoder='deformable',
             width=64, learning_rate=3e-4),
        dict(id='mlp64_lr3e4', encoder='mlp', depth=3, decoder='deformable',
             width=64, learning_rate=3e-4),
        dict(id='linear', encoder='identity', depth=0, decoder='linear'),
        dict(id='patch_direct', encoder='transformer', depth=3, decoder='direct'),
        dict(id='mlp_direct', encoder='identity', depth=0, decoder='mlp_direct'),
        dict(id='cnn_direct', encoder='cnn', depth=3, decoder='direct'),
    ]


def candidate_config(config, candidate, small_test=False):
    config = copy.deepcopy(config)
    config['model']['e_layers'] = candidate['depth']
    if 'width' in candidate and not small_test:
        config['model']['d_model'] = candidate['width']
        config['model']['d_ff'] = 2 * candidate['width']
    if 'learning_rate' in candidate:
        config['training']['learning_rate'] = candidate['learning_rate']
    return config


class LinearConv1d(nn.Conv1d):
    """Same zero-padded dilated cross-correlation, using GEMM on short patch sequences."""
    def forward(self, x):
        kernel = self.kernel_size[0]
        dilation = self.dilation[0]
        padded = F.pad(x, (self.padding[0], self.padding[0]))
        windows = padded.unfold(-1, 1 + (kernel - 1) * dilation, self.stride[0])[..., ::dilation]
        windows = windows.permute(0, 2, 1, 3).reshape(x.shape[0], -1, x.shape[1] * kernel)
        return F.linear(windows, self.weight.reshape(self.out_channels, -1), self.bias).transpose(1, 2)


class ResidualTokenBlock(nn.Module):
    """Mix features with an MLP or neighboring patch positions with a CNN."""
    def __init__(self, width, hidden, dropout, convolution=False, dilation=1):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.convolution = convolution
        if convolution:
            self.net = nn.Sequential(
                LinearConv1d(width, hidden, kernel_size=3, padding=dilation, dilation=dilation),
                nn.GELU(), nn.Dropout(dropout), LinearConv1d(hidden, width, kernel_size=1))
        else:
            self.net = nn.Sequential(nn.Linear(width, hidden), nn.GELU(),
                                     nn.Dropout(dropout), nn.Linear(hidden, width))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        z = self.norm(x)
        if self.convolution:
            z = self.net(z.transpose(1, 2)).transpose(1, 2)
        else:
            z = self.net(z)
        return x + self.dropout(z)


def build_candidate(config, horizon, candidate, small_test=False):
    from experiments.runner import _build_model
    cfg = candidate_config(config, candidate, small_test)
    model = _build_model(cfg, horizon)
    c = cfg['model']
    width, hidden, dropout = c['d_model'], c['d_ff'], c['dropout']
    kind, depth = candidate['encoder'], candidate['depth']
    if kind == 'identity':
        model.encoder = nn.Identity()
    elif kind in ('mlp', 'cnn', 'hybrid'):
        blocks = [ResidualTokenBlock(width, hidden, dropout, kind != 'mlp', 2**i)
                  for i in range(depth)]
        if kind == 'hybrid':
            blocks.append(TemporalTransformerEncoder(width, c['n_heads'], 1, hidden, dropout))
        else:
            blocks.append(nn.LayerNorm(width))
        model.encoder = nn.Sequential(*blocks)
    elif kind != 'transformer':
        raise ValueError('Unknown encoder: ' + kind)
    decoder = candidate['decoder']
    if decoder == 'deformable':
        return model
    if decoder == 'direct':
        return DirectQuantileControl(model, patch=True)
    if decoder in ('linear', 'mlp_direct'):
        direct = DirectQuantileControl(model, patch=False)
        if decoder == 'mlp_direct':
            projection = nn.Linear(hidden, horizon * 7)
            initialize_raw_projection(projection, horizon)
            direct.output_projection = nn.Sequential(
                nn.Linear(model.seq_len, hidden), nn.GELU(), nn.Dropout(dropout), projection)
        return direct
    raise ValueError('Unknown decoder: ' + decoder)


def select_by_validation(records, seeds):
    groups = {}
    for record in records:
        candidate_id = record['candidate_id']
        seed = int(record['seed'])
        group = groups.setdefault(candidate_id, {})
        if seed in group:
            raise ValueError('Duplicate candidate/seed validation record')
        value = float(record['best_val_pinball'])
        if not math.isfinite(value) or value < 0:
            raise ValueError('Invalid validation loss')
        group[seed] = value
    ranked = [dict(candidate_id=candidate_id, mean_val=float(np.mean(list(group.values()))),
                   seeds=sorted(group))
              for candidate_id, group in groups.items() if set(group) == set(seeds)]
    if not ranked:
        raise ValueError('No complete validation groups')
    return min(ranked, key=lambda r: (r['mean_val'], r['candidate_id']))
