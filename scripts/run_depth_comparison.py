"""ETTh1 encoder-depth experiment, retaining the established bypass definition of depth 0."""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
import yaml
from utils.config import load_config
from experiments.runner import run_single_experiment

ROOT = Path('outputs/ETTh1_depth_comparison')
BASE = Path('outputs/ETTh1_analysis')


def validate_saved(run, expected):
    resolved = json.loads((run / 'resolved_config.json').read_text())
    actual = copy.deepcopy(resolved['config'])
    actual.pop('data_scaler', None)
    actual.pop('feature_columns', None)
    if actual != expected:
        raise RuntimeError(f'Configuration mismatch: {run}')
    checkpoint = torch.load(run / 'best_checkpoint.pt', map_location='cpu', weights_only=False)
    history = pd.read_csv(run / 'training_history.csv')
    metrics = json.loads((run / 'metrics.json').read_text())
    assert checkpoint['epoch'] == int(history.loc[history.val_pinball.idxmin(), 'epoch'])
    assert np.isclose(metrics['best_val_pinball'], history.val_pinball.min())
    assert checkpoint['config'] == resolved['config']
    assert (run / 'figures/calibration.pdf').exists(), f'Incomplete figures: {run}'
    assert metrics['dataset'] == 'ETTh1' and metrics['pred_len'] == 96 and metrics['seed'] == 2024
    return resolved, checkpoint, history, metrics


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '3' or not torch.cuda.is_available():
        raise RuntimeError('Run with physical GPU 3 exposed through CUDA_VISIBLE_DEVICES=3.')
    config = load_config('configs/etth1_gpu3.yaml')
    torch.set_num_threads(int(config['runtime']['torch_threads']))
    ROOT.mkdir(parents=True, exist_ok=True)
    rows, histories, forecasts, references = [], {}, {}, []
    shared_data = None
    for depth in [0, 1, 2, 3]:
        cfg = copy.deepcopy(config)
        if depth in [0, 3]:
            variant = 'no_encoder' if depth == 0 else 'full'
            cfg['experiment']['run_name'] = f'ETTh1_analysis/{variant}'
            cfg['ablation'] = 'no_encoder' if depth == 0 else None
            run = BASE / variant / 'ETTh1/H96/seed2024'
            reused = True
        else:
            cfg['model']['e_layers'] = depth
            cfg['experiment']['run_name'] = f'ETTh1_depth_comparison/layers{depth}'
            cfg['ablation'] = None
            run = ROOT / f'layers{depth}/ETTh1/H96/seed2024'
            reused = (run / 'figures/calibration.pdf').exists()
            (ROOT / f'layers{depth}.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
            if not reused:
                run_single_experiment(cfg, 'ETTh1', 96, 2024)
        resolved, checkpoint, hist, m = validate_saved(run, cfg)
        data = dict(scaler=resolved['scaler'], bounds=resolved['split_bounds'],
                    names=resolved['feature_columns'], rows=resolved['n_rows'])
        if shared_data is None:
            shared_data = data
        assert data == shared_data, 'Data or scaling mismatch between experiments'
        state = checkpoint['model_state']
        used_parameters = sum(v.numel() for k, v in state.items()
                              if not (depth == 0 and k.startswith('encoder.')))
        rows.append(dict(layers=depth, zero_means_encoder_bypass=depth == 0,
                         best_epoch=checkpoint['epoch'], epochs_run=len(hist),
                         val_pinball=m['best_val_pinball'], test_pinball=m['QuantileLoss_standardized'],
                         PICP50=m['PICP50'], PICP80=m['PICP80'], PICP90=m['PICP90'],
                         PINAW90=m['PINAW90'], NormalizedWinkler90=m['NormalizedWinkler90'],
                         CrossingRate=m['CrossingRate'], training_seconds=m['training_seconds'],
                         seconds_per_epoch=m['training_seconds']/len(hist),
                         active_parameters=used_parameters, reused=reused, source_dir=str(run)))
        histories[depth] = hist
        forecasts[depth] = dict(np.load(run / 'figures/forecast_samples.npz'))
        if depth:
            np.testing.assert_array_equal(forecasts[depth]['target'], forecasts[0]['target'])
            np.testing.assert_array_equal(forecasts[depth]['sample_indices'], forecasts[0]['sample_indices'])
        references.append(dict(depth=depth, source_dir=str(run), reused=reused))
        pd.DataFrame(rows).to_csv(ROOT / 'comparison.csv', index=False)
        print(f'Completed depth {depth}: val={m["best_val_pinball"]:.6f}, test={m["QuantileLoss_standardized"]:.6f}', flush=True)
    frame = pd.DataFrame(rows)
    selected = int(frame.loc[frame.val_pinball.idxmin(), 'layers'])
    manifest = dict(dataset_sha256=hashlib.sha256(Path('dataset/ETT-small/ETTh1.csv').read_bytes()).hexdigest(),
                    physical_gpu=3, GPU=torch.cuda.get_device_name(0), python=sys.executable,
                    torch=torch.__version__, seed=2024, pred_len=96,
                    selected_by_validation=selected, runs=references, data=shared_data,
                    zero_definition='Entire encoder including terminal LayerNorm bypassed; original dormant encoder retained in checkpoint.')
    (ROOT / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    def save(fig, name):
        fig.tight_layout()
        for ext in ['png','pdf']:
            fig.savefig(ROOT / f'{name}.{ext}', dpi=160, bbox_inches='tight')
        plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0,0].plot(frame.layers.to_numpy(),frame.val_pinball.to_numpy(),'o-',label='Validation')
    axes[0,0].plot(frame.layers.to_numpy(),frame.test_pinball.to_numpy(),'s-',label='Test')
    axes[0,0].set(title='Pinball loss (lower is better)',ylabel='Standardized loss')
    axes[0,0].legend()
    axes[0,1].plot(frame.layers.to_numpy(),frame.PICP90.to_numpy(),'o-')
    axes[0,1].axhline(.9,linestyle='--',color='0.4',label='Nominal 90%')
    axes[0,1].set(title='90% interval coverage',ylabel='PICP90')
    axes[0,1].legend()
    axes[1,0].plot(frame.layers.to_numpy(),frame.PINAW90.to_numpy(),'o-')
    axes[1,0].set(title='90% normalized interval width',ylabel='PINAW90')
    axes[1,1].bar(frame.layers.to_numpy(),frame.seconds_per_epoch.to_numpy())
    axes[1,1].set(title='Observed training + validation time / epoch',ylabel='Seconds')
    for ax in axes.flat:
        ax.set_xticks([0,1,2,3]);ax.set_xlabel('Encoder layers (0 = bypass)');ax.grid(alpha=.15)
    save(fig,'depth_comparison')
    fig,axes=plt.subplots(1,2,figsize=(13,4))
    for d,h in histories.items():
        axes[0].plot(h.epoch.to_numpy(),h.train_pinball.to_numpy(),label=f'{d} layers')
        axes[1].plot(h.epoch.to_numpy(),h.val_pinball.to_numpy(),label=f'{d} layers')
    for ax,name in zip(axes,['Training','Validation']):
        ax.set(title=name,xlabel='Epoch',ylabel='Pinball loss');ax.legend();ax.grid(alpha=.15)
    save(fig,'learning_curves')
    c = list(forecasts[0]['feature_columns']).index('OT')
    fig,axes=plt.subplots(2,2,figsize=(13,7))
    t=np.arange(1,97)
    for d,ax in enumerate(axes.flat):
        f=forecasts[d];q=f['quantiles'][-1,:,c];y=f['target'][-1,:,c]
        ax.fill_between(t,q[:,0],q[:,6],color='#b8d4ed',label='90% interval')
        ax.plot(t,y,color='black',label='Actual')
        ax.plot(t,q[:,3],color='#c84531',label='Predicted median')
        ax.set(title=f'{d} layers | same fixed last test window',xlabel='Forecast hour',ylabel='OT original units')
        ax.legend(fontsize=8);ax.grid(alpha=.15)
    save(fig,'forecast_OT_depths')
    lines=['# ETTh1 Transformer 编码器层数对比','',
           'djjTF，物理 GPU 3（RTX 5090），L=336，H=96，seed=2024，7 变量，时间顺序 60/20/20 划分。',
           '统一 batch=32、最多30轮、验证早停7轮、学习率与其余模型设置。各组独立按验证集损失选优。',
           '0层表示跳过整个编码器（包括末尾 LayerNorm），与上一轮 no_encoder 一致；不是只设置 e_layers=0。',
           '0、3层复用已核验配置、标准化参数和数据边界的历史实验；1、2层为本轮补跑。', '',
           '| 层数 | 最佳轮/实际轮 | 验证 pinball↓ | 测试 pinball↓ | PICP90 | PINAW90 | 每轮秒数 | 前向参与参数量 |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f'| {r["layers"]} | {r["best_epoch"]}/{r["epochs_run"]} | {r["val_pinball"]:.6f} | {r["test_pinball"]:.6f} | {r["PICP90"]:.2%} | {r["PINAW90"]:.6f} | {r["seconds_per_epoch"]:.2f} | {r["active_parameters"]:,} |')
    lines += ['',f'按验证集损失选择：**{selected} 层**。测试指标用于描述所选模型的表现，不用于调层数。',
              '覆盖率须结合区间宽度和损失判断；每轮耗时包含训练和验证，仅为本机观测，不是严格基准测试。',
              '各层数初始化消耗的随机数不同，相同 seed 不保证共享部分权重逐元素相同。本轮是单 seed 探索，不能认定统计显著性。',
              '0层检查点仍保存未参与前向的编码器；表中参数量排除了这部分，不把实际分配的模型参数量当作有效结构大小。', '',
              '![指标对比](depth_comparison.png)','', '![训练与验证曲线](learning_curves.png)','',
              '![同一OT测试窗口预测](forecast_OT_depths.png)','', '## 原始结果目录','']
    for r in rows:
        lines.append(f'- {r["layers"]} 层：`{r["source_dir"]}`（含最佳参数、预测图、训练历史）。')
    lines += ['', '复现：', '', '```bash', 'conda activate djjTF',
              'CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=4 \\',
              '  python scripts/run_depth_comparison.py', '```',
              '入口自动核验并复用完成的结果；配置不一致时拒绝复用。汇总数据见 comparison.csv，数据指纹和复用来源见 manifest.json。']
    (ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(frame.to_string(index=False),flush=True)
    print('Validation-selected depth:',selected,flush=True)


if __name__ == '__main__':
    main()
