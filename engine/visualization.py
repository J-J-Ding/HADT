"""Bounded, reproducible diagnostic figures; no test-driven sample selection."""
from pathlib import Path
import copy
import re
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path.with_suffix('.png'), dpi=160, bbox_inches='tight')
    fig.savefig(path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)


def save_forecast_figures(x, y, q, scaler, names, sample_indices, output_dir, channel_indices=None):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    x = scaler.inverse_transform(x)
    y = scaler.inverse_transform(y)
    point = q.ndim == 3
    q = scaler.inverse_transform(q) if point else q * scaler.scale_[None, None, :, None] + scaler.mean_[None, None, :, None]
    prediction_arrays = {'point_predictions' if point else 'quantiles': q}
    np.savez_compressed(out / 'forecast_samples.npz', history=x, target=y, **prediction_arrays,
                        sample_indices=sample_indices, feature_columns=np.array(names),
                        channel_indices=np.arange(len(names)) if channel_indices is None else channel_indices,
                        quantile_levels=[.05, .1, .25, .5, .75, .9, .95])
    for c, name in enumerate(names):
        fig, axes = plt.subplots(len(x), 1, figsize=(13, 3.4 * len(x)), squeeze=False)
        for i, ax in enumerate(axes[:, 0]):
            t = np.arange(1, y.shape[1] + 1)
            ax.plot(np.arange(-x.shape[1]+1, 1), x[i, :, c], color='0.45', label='Observed history')
            if not point:
                for lo, hi, label, color in [(0, 6, '90% interval', '#dce9fa'),
                                             (1, 5, '80% interval', '#b1cfee'),
                                             (2, 4, '50% interval', '#6fa4d4')]:
                    ax.fill_between(t, q[i, :, c, lo], q[i, :, c, hi], color=color, label=label)
            ax.plot(t, y[i, :, c], color='#202020', label='Actual', linewidth=1.3)
            ax.plot(t, q[i, :, c] if point else q[i, :, c, 3], color='#c84531',
                    label='Point prediction' if point else 'Predicted median', linewidth=1.3)
            ax.axvline(0, color='0.4', linestyle='--')
            ax.set(title=f'{name} | fixed test window {sample_indices[i]}', xlabel='Hours from forecast origin', ylabel='Original units')
            ax.grid(alpha=.15)
        axes[0, 0].legend(ncol=3, fontsize=8)
        safe = re.sub(r'[^a-zA-Z0-9_-]', '_', str(name))
        _save(fig, out / f'forecast_{c:02d}_{safe}')


@torch.no_grad()
def create_run_figures(model, bundle, device, run_dir, metrics, max_channels=None):
    model.eval()
    run_dir = Path(run_dir)
    out = run_dir / 'figures'
    ds = bundle.test_loader.dataset
    indices = sorted(set([0, len(ds)//2, len(ds)-1]))
    samples = [ds[i] for i in indices]
    n_channels = len(bundle.feature_columns)
    if max_channels is not None and int(max_channels) < 1:
        raise ValueError("max_channels must be positive or None")
    channel_indices = np.arange(n_channels)
    if max_channels is not None and n_channels > int(max_channels):
        channel_indices = np.linspace(0, n_channels - 1, int(max_channels), dtype=int)
    names = [bundle.feature_columns[i] for i in channel_indices]
    plot_scaler = copy.copy(bundle.scaler)
    for key in ('mean_', 'scale_', 'data_min_', 'data_max_', 'data_range_'):
        setattr(plot_scaler, key, getattr(bundle.scaler, key)[channel_indices])
    full_x = torch.stack([s[0] for s in samples]).to(device)
    full_context = not getattr(model, 'supports_channel_subsets', True)
    x = full_x if full_context else full_x[:, :, channel_indices]
    y = torch.stack([s[1] for s in samples])[:, :, channel_indices].numpy()
    predictions, auxiliary = [], []
    # One window at a time bounds diagnostic activation memory.
    for item in x:
        pred, aux = model(item[None], return_aux=True)
        if full_context:
            pred = pred[:, :, channel_indices]
        predictions.append(pred.cpu().numpy())
        auxiliary.append({k: v.cpu().numpy() for k, v in aux.items() if torch.is_tensor(v)})
    q = np.concatenate(predictions)
    plot_x = full_x[:, :, channel_indices]
    save_forecast_figures(plot_x.cpu().numpy(), y, q, plot_scaler, names, indices, out, channel_indices=channel_indices)
    if auxiliary[0]:
        aux = {k: np.concatenate([a[k] for a in auxiliary]) for k in auxiliary[0]}
        np.savez_compressed(out / 'intermediate_steps.npz', **aux, sample_indices=indices, channel_indices=channel_indices)
        # Inspect OT when available; choice made independently of predictions.
        c = names.index('OT') if 'OT' in names else 0
        fig, axes = plt.subplots(2, 3, figsize=(16, 8))
        axes[0, 0].plot(aux['normalized_input'][0, :, c])
        axes[0, 0].set(title='RevIN normalized history', xlabel='History step')
        for ax, key, title in [(axes[0, 1], 'tokens', 'Patch embedding'),
                               (axes[0, 2], 'encoded', 'Encoder representation')]:
            im = ax.imshow(aux[key][0, c].T, aspect='auto', origin='lower', cmap='viridis')
            ax.set(title=title, xlabel='History patch', ylabel='Hidden dimension')
            fig.colorbar(im, ax=ax)
        offsets, positions, attention = (aux[k][0, c] for k in ('offsets', 'positions', 'attention'))
        im = axes[1, 0].imshow(offsets.mean(1).T, aspect='auto', origin='lower', cmap='coolwarm',
                              vmin=-model.decoder.max_offset if model.decoder.max_offset else -1,
                              vmax=model.decoder.max_offset or 1)
        axes[1, 0].set(title='Offsets (head mean, patch units)', xlabel='Forecast horizon', ylabel='Sampling point')
        fig.colorbar(im, ax=axes[1, 0])
        for k in range(model.n_points):
            axes[1, 1].scatter(np.arange(model.pred_len)+1, positions[:, 0, k],
                               s=10+150*attention[:, 0, k], alpha=.65, label=f'point {k}')
        axes[1, 1].set(title='Head 0 sampling (size = attention)', xlabel='Forecast horizon', ylabel='History patch position')
        im = axes[1, 2].imshow(attention[:, 0].T, aspect='auto', origin='lower', vmin=0, vmax=1)
        axes[1, 2].set(title='Head 0 attention weights', xlabel='Forecast horizon', ylabel='Sampling point')
        fig.colorbar(im, ax=axes[1, 2])
        fig.suptitle(f'Intermediate diagnostics | {names[c]} | first fixed test window')
        _save(fig, out / 'intermediate_steps')
    hist = pd.read_csv(run_dir / 'training_history.csv')
    fig, ax = plt.subplots(figsize=(9, 4))
    point = getattr(model, 'output_mode', 'quantile') == 'point'
    train_key, val_key = ('train_mse','val_mse') if point else ('train_pinball','val_pinball')
    ax.plot(hist.epoch.to_numpy(), hist[train_key].to_numpy(), label='Train')
    ax.plot(hist.epoch.to_numpy(), hist[val_key].to_numpy(), label='Validation')
    best_epoch = int(hist.loc[hist[val_key].idxmin(), 'epoch'])
    ax.axvline(best_epoch, linestyle='--', color='0.4', label=f'Best epoch {best_epoch}')
    ax.set(xlabel='Epoch', ylabel='Standardized MSE' if point else 'Standardized pinball loss', title='Training and checkpoint selection')
    ax.legend()
    _save(fig, out / 'training_history')
    if point:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    nominal = [.5, .8, .9]
    axes[0].plot(nominal, [metrics[f'PICP{int(v*100)}'] for v in nominal], 'o-', label='Observed coverage')
    axes[0].plot([0, 1], [0, 1], '--', color='0.5', label='Ideal')
    axes[0].set(xlabel='Nominal coverage', ylabel='Observed coverage', title='Test interval calibration')
    axes[0].legend()
    horizon_file = run_dir / 'horizon_metrics.csv'
    if horizon_file.exists():
        frame = pd.read_csv(horizon_file)
        for v in nominal:
            axes[1].plot(frame.horizon_step.to_numpy(), frame[f'PICP{int(v*100)}'].to_numpy(), label=f'{v:.0%}')
            axes[1].axhline(v, linestyle=':', alpha=.4)
    axes[1].set(xlabel='Forecast horizon', ylabel='Coverage', title='Coverage by horizon')
    if horizon_file.exists():
        axes[1].legend()
    _save(fig, out / 'calibration')
