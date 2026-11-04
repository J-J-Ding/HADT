"""Independent checkpoint/data/forecast audit of completed architecture-study runs."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch

from data_provider.csv_dataset import read_multivariate_csv
from data_provider.scaler import StandardScaler
from scripts.run_architecture_study import sha, source_digest, atomic_json
from scripts.export_architecture_study import records, groups, verify, OUT, ROOT


def main():
    validation = records('validation_complete')
    tested = records('complete')
    verify(validation + tested)
    datasets = {}
    audited = []
    source_hash = source_digest()
    for record in validation:
        path = Path(record['source_dir'])
        if record['code_sha256'] != source_hash:
            raise ValueError('Current source differs from recorded run: ' + str(path))
        if sha(path / 'best_checkpoint.pt') != record['checkpoint_sha256']:
            raise ValueError('Checkpoint digest differs: ' + str(path))
        cp = torch.load(path / 'best_checkpoint.pt', map_location='cpu', weights_only=False)
        resolved = json.loads((path / 'resolved_config.json').read_text())
        cfg = resolved['config']
        dataset = record['dataset']
        if dataset not in datasets:
            file = Path(cfg['data']['root_path']) / cfg['data']['files'][dataset]
            raw, names = read_multivariate_csv(file, cfg['data']['time_column'])
            ratios = cfg['data']['split_ratios']['ETT' if dataset.startswith('ETT') else 'other']
            train_end = int(len(raw) * ratios[0])
            val_end = train_end + int(len(raw) * ratios[1])
            scaler = StandardScaler().fit(raw[:train_end])
            datasets[dataset] = dict(raw=raw, names=names, scaler=scaler,
                                      data_sha256=sha(file), val_end=val_end, train_end=train_end)
        data = datasets[dataset]
        assert data['data_sha256'] == record['data_sha256']
        assert data['names'] == resolved['feature_columns']
        assert list(resolved['split_bounds']['train']) == [0, data['train_end']]
        assert list(resolved['split_bounds']['val']) == [data['train_end'], data['val_end']]
        assert list(resolved['split_bounds']['test']) == [data['val_end'], len(data['raw'])]
        for key, value in data['scaler'].state_dict().items():
            np.testing.assert_allclose(cp['scaler'][key], value, rtol=0, atol=1e-12)
        assert np.isclose(cp['best_val_loss'], record['best_val_pinball'], rtol=0, atol=1e-12)
        assert cp['epoch'] == record['best_epoch']
        assert cp['config'] == cfg
        history = pd.read_csv(path / 'training_history.csv')
        assert cp['best_val_loss'] <= history.val_pinball.min() + 1e-8
        if (path / 'complete.json').exists():
            sample = np.load(path / 'forecast_samples.npz')
            horizon = record['horizon']
            channels = sample['channel_indices']
            windows = len(data['raw']) - data['val_end'] - horizon + 1
            np.testing.assert_array_equal(sample['sample_indices'], [0, windows // 2, windows - 1])
            assert sample['quantiles'].shape == (3, horizon, len(channels), 7)
            assert np.isfinite(sample['quantiles']).all()
            assert (np.diff(sample['quantiles'], axis=-1) >= -1e-10).all()
            for i, window in enumerate(sample['sample_indices']):
                start = data['val_end'] + int(window)
                np.testing.assert_allclose(sample['target'][i], data['raw'][start:start + horizon, channels], atol=1e-5, rtol=1e-5)
                np.testing.assert_allclose(sample['history'][i], data['raw'][start - cfg['data']['seq_len']:start, channels], atol=1e-5, rtol=1e-5)
            completed = json.loads((path / 'complete.json').read_text())
            assert completed['selection_sha256'] == sha(ROOT / 'selection.json')
            assert completed['test_windows'] == windows
            assert completed['test_elements'] == windows * horizon * resolved['n_vars']
        audited.append(str(path))
    aggregates = groups(tested)
    if aggregates:
        exported = pd.read_csv(OUT / 'architecture_aggregates.csv')
        assert len(exported) == len(aggregates)
        for row in aggregates:
            match = exported[(exported.dataset == row['dataset']) & (exported.horizon == row['horizon']) &
                             (exported.candidate_id == row['candidate_id'])].iloc[0]
            for key, value in row.items():
                if isinstance(value, float):
                    assert np.isclose(match[key], value, rtol=1e-12, atol=1e-12)
    report = dict(status='passed', validation_runs=len(validation), test_runs=len(tested),
                  aggregate_groups=len(aggregates), audited=audited,
                  checks=['source/data/checkpoint SHA256', 'checkpoint validation selection',
                          'train-only scaler', 'split boundaries', 'fixed forecast sample truth',
                          'quantile ordering', 'complete-seed aggregates', 'frozen selection digest'],
                  scope='Artifact audit; does not rerun neural predictions over every full test dataset.')
    atomic_json(OUT / 'architecture_independent_audit.json', report)
    print(json.dumps({k: v for k, v in report.items() if k not in ('audited', 'checks')}))


if __name__ == '__main__':
    main()
