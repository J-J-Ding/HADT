"""Evaluate saved HADT-NQ medians and naive forecasts on identical test windows."""
import argparse
import html
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_provider.data_factory import build_data_bundle
from engine.checkpoint import load_checkpoint
from experiments.runner import _build_model, _resolve_device


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir
    resolved = json.loads((run / 'resolved_config.json').read_text())
    config = resolved['config']
    torch.set_num_threads(config['runtime'].get('torch_threads', 4))
    device = _resolve_device(config)
    horizon = resolved['pred_len']
    bundle = build_data_bundle(config, resolved['dataset'], horizon)
    model = _build_model(config, horizon).to(device)
    load_checkpoint(run / 'best_checkpoint.pt', model, map_location=device)
    model.eval()
    names = bundle.feature_columns
    scale = bundle.scaler.scale_
    accum = {name: dict(ae=np.zeros(len(names)), se=np.zeros(len(names)))
             for name in ['HADT_NQ', 'Persistence', 'Seasonal_96']}
    count = 0
    y_sum = np.zeros(len(names))
    y_sq_sum = np.zeros(len(names))
    covered = np.zeros(len(names))
    width = np.zeros(len(names))
    pinball = np.zeros(len(names))
    tau = np.asarray(config['model']['quantiles'])
    with torch.inference_mode():
        for x, y in tqdm(bundle.test_loader, desc='Model and naive baselines', mininterval=5):
            q = model(x.to(device)).cpu().numpy().astype(np.float64)
            truth = y.numpy().astype(np.float64)
            history = x.numpy().astype(np.float64)
            predictions = dict(HADT_NQ=q[..., 3], Persistence=history[:, -1:, :],
                               Seasonal_96=history[:, -96:, :][:, np.arange(horizon) % 96, :])
            for name, prediction in predictions.items():
                error = prediction - truth
                accum[name]['ae'] += np.abs(error).sum(axis=(0, 1))
                accum[name]['se'] += np.square(error).sum(axis=(0, 1))
            count += truth.shape[0] * horizon
            y_sum += truth.sum(axis=(0, 1))
            y_sq_sum += np.square(truth).sum(axis=(0, 1))
            covered += ((truth >= q[..., 0]) & (truth <= q[..., 6])).sum(axis=(0, 1))
            width += (q[..., 6] - q[..., 0]).sum(axis=(0, 1))
            error = truth[..., None] - q
            pinball += np.maximum(tau * error, (tau - 1) * error).sum(axis=(0, 1, 3)) / len(tau)
    variance_sum = y_sq_sum - y_sum ** 2 / count
    rows = []
    for name, stat in accum.items():
        for channel, feature in enumerate(names):
            rows.append(dict(model=name, feature=feature,
                             MAE=float(stat['ae'][channel] / count * scale[channel]),
                             RMSE=float(np.sqrt(stat['se'][channel] / count) * scale[channel]),
                             MSE_standardized=float(stat['se'][channel] / count),
                             R2=float(1 - stat['se'][channel] / variance_sum[channel])
                             if variance_sum[channel] > 0 else None))
    points = pd.DataFrame(rows)
    points.to_csv(run / 'point_baseline_metrics.csv', index=False)
    intervals = pd.DataFrame(dict(feature=names, PICP90=covered / count,
                                 MPIW90=width / count * scale,
                                 QuantileLoss_standardized=pinball / count))
    intervals.to_csv(run / 'per_channel_intervals.csv', index=False)
    summary = dict(test_windows=len(bundle.test_loader.dataset), horizon=horizon,
                   elements_per_channel=count,
                   evaluation='All overlapping test windows; original units per variable; train-only scaler.',
                   aggregate_MSE_standardized={name: float(stat['se'].mean() / count)
                                               for name, stat in accum.items()},
                   OT=points[points.feature == 'OT'].to_dict('records'),
                   OT_intervals=intervals[intervals.feature == 'OT'].to_dict('records'))
    (run / 'baseline_comparison.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    title = html.escape(f"{resolved['dataset']} · H{horizon} · seed{resolved['seed']}")
    page = ('<!doctype html><meta charset="utf-8"><title>Wind farm evaluation</title>'
            '<style>body{font:16px sans-serif;max-width:1200px;margin:32px auto;padding:20px}'
            'table{border-collapse:collapse;margin:24px 0}th,td{padding:8px;border:1px solid #ddd}</style>'
            f'<h1>{title}</h1><p>全部测试滑动窗口，同一时间划分。MAE、RMSE 为各变量原始单位；'
            'MSE_standardized 为训练集标准差归一化后的均方误差。HADT_NQ 使用预测中位数。'
            'Persistence 保持最后值，Seasonal_96 重复前一天。单种子探索实验。</p>'
            '<p><a href="figures/predictions/index.html">查看预测曲线</a> · '
            '<a href="point_baseline_metrics.csv">下载点预测指标</a> · '
            '<a href="per_channel_intervals.csv">下载区间指标</a></p>'
            '<h2>OT 点预测</h2>' + points[points.feature == 'OT'].to_html(index=False, float_format=lambda v: f'{v:.5f}') +
            '<h2>各变量 90% 区间覆盖率与宽度</h2>' + intervals.to_html(index=False, float_format=lambda v: f'{v:.5f}') +
            '<h2>各变量点预测</h2>' + points.to_html(index=False, float_format=lambda v: f'{v:.5f}'))
    (run / 'evaluation.html').write_text(page, encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
