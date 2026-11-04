"""Reproducible sequential training and full-test component analysis on one GPU."""
import argparse
import copy
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
import torch
from utils.config import load_config
from experiments.runner import run_single_experiment, _build_model, _resolve_device
from data_provider.data_factory import build_data_bundle
from engine.checkpoint import load_checkpoint
from engine.evaluator import evaluate_test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/etth1_gpu3.yaml')
    parser.add_argument('--variants', nargs='+', default=['full', 'no_offsets', 'no_encoder', 'no_revin'],
                        choices=['full', 'no_offsets', 'no_encoder', 'no_revin'])
    parser.add_argument('--reuse-existing', action='store_true', help='Reuse only completed runs with identical configuration')
    args = parser.parse_args()
    config = load_config(args.config)
    torch.set_num_threads(int(config['runtime'].get('torch_threads', 4)))
    device = _resolve_device(config)
    if device.type != 'cuda':
        raise RuntimeError('This experiment explicitly requires CUDA.')
    root = Path(config['experiment']['output_dir']) / 'ETTh1_analysis'
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for variant in args.variants:
        cfg = copy.deepcopy(config)
        cfg['experiment']['run_name'] = f'ETTh1_analysis/{variant}'
        cfg['ablation'] = None if variant == 'full' else variant
        run = root / variant / 'ETTh1/H96/seed2024'
        if args.reuse_existing and (run / 'metrics.json').exists():
            resolved = json.loads((run / 'resolved_config.json').read_text())['config']
            expected = {k: v for k, v in resolved.items() if k not in ('data_scaler', 'feature_columns')}
            if expected != cfg:
                raise RuntimeError(f'Configuration mismatch in {run}; refusing to reuse.')
            metrics = json.loads((run / 'metrics.json').read_text())
            if not (run / 'figures/calibration.png').exists():
                from engine.visualization import create_run_figures
                restored = _build_model(cfg, 96).to(device)
                load_checkpoint(run / 'best_checkpoint.pt', restored, map_location=device)
                data = build_data_bundle(cfg, 'ETTh1', 96)
                create_run_figures(restored, data, device, run, metrics)
                del restored, data
        else:
            metrics = run_single_experiment(cfg, 'ETTh1', 96, 2024)
        rows.append(dict(variant=variant, kind='retrained', **metrics))
        pd.DataFrame(rows).to_csv(root / 'component_comparison.csv', index=False)
        if variant == 'full':
            model = _build_model(cfg, 96).to(device)
            load_checkpoint(run / 'best_checkpoint.pt', model, map_location=device)
            bundle = build_data_bundle(cfg, 'ETTh1', 96)
            for intervention in ['no_offsets', 'no_encoder', 'no_revin']:
                values = evaluate_test(model, bundle.test_loader, device, bundle.scaler,
                                       cfg['evaluation']['channel_chunk_size'], cfg['model']['quantiles'],
                                       run / 'interventions' / intervention, intervention=intervention)
                rows.append(dict(variant=intervention, kind='inference_intervention', **values))
            del model, bundle
            torch.cuda.empty_cache()
    frame = pd.DataFrame(rows)
    frame.to_csv(root / 'component_comparison.csv', index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    labels = frame['kind'] + '\n' + frame['variant']
    for ax, col in zip(axes, ['QuantileLoss_standardized', 'PICP90']):
        ax.bar(range(len(frame)), frame[col], color=['#377ba8' if k == 'retrained' else '#cb8c4e' for k in frame.kind])
        ax.set_xticks(range(len(frame)))
        ax.set_xticklabels(labels, rotation=35, ha='right', fontsize=8)
        ax.set_ylabel(col)
    axes[1].axhline(.9, color='0.4', linestyle='--', label='Nominal 90%')
    axes[1].legend()
    fig.suptitle('Component comparisons | retrained ablations and fitted-model interventions')
    fig.tight_layout()
    fig.savefig(root / 'component_comparison.png', dpi=160)
    fig.savefig(root / 'component_comparison.pdf')
    plt.close(fig)
    print(frame[['variant', 'kind', 'QuantileLoss_standardized', 'PICP90', 'CrossingRate']].to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
