"""GPU3-only, restartable validation search and predeclared transfer matrix."""
import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from data_provider.data_factory import build_data_bundle
from engine.checkpoint import load_checkpoint
from engine.evaluator import evaluate_test
from engine.trainer import Trainer
from models.architecture_search import candidates, candidate_config, build_candidate, select_by_validation
from scripts.run_paper_experiments import configuration, samples
from utils.logger import create_logger
from utils.seed import set_seed

ROOT = Path('outputs/architecture_study')
SEEDS = [2024, 2025, 2026]


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def sha(path):
    result = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def source_digest():
    paths = []
    for directory in ['models', 'data_provider', 'engine', 'metrics', 'losses', 'utils']:
        paths.extend(Path(directory).glob('*.py'))
    paths.extend([Path(__file__), Path('scripts/run_paper_experiments.py'), Path('experiments/runner.py')])
    paths.extend([Path('configs/etth1_gpu3.yaml'), Path('configs/traffic_gpu3.yaml')])
    return hashlib.sha256(''.join(str(p) + sha(p) for p in sorted(set(paths))).encode()).hexdigest()


def check_fingerprint(path, signature):
    fingerprint = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    marker = Path(path) / 'provenance.json'
    if not marker.exists() and any((Path(path) / name).exists() for name in ['validation_complete.json', 'complete.json']):
        raise RuntimeError('Completed cache is missing provenance: ' + str(path))
    if marker.exists() and json.loads(marker.read_text()) != dict(fingerprint=fingerprint, **signature):
        raise RuntimeError('Refusing incompatible cached run: ' + str(path))
    return dict(fingerprint=fingerprint, **signature)


def freeze_inputs(paths):
    return {str(path): sha(path) for path in paths}


def verify_inputs(signature):
    for path, expected in signature.items():
        if sha(path) != expected:
            raise RuntimeError('Frozen study input changed: ' + path)


def export():
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='')
    subprocess.run([sys.executable, 'scripts/export_architecture_study.py'],
                   env=environment, check=True)


def train_job(stage, candidate, dataset, horizon, seed, source_hash):
    if source_digest() != source_hash:
        raise RuntimeError('Training implementation changed during study')
    config = configuration(dataset, 'hadt')
    config['experiment']['run_name'] = 'architecture_study'
    config = candidate_config(config, candidate)
    config['paper'] = dict(protocol='architecture-v1', stage=stage, candidate=candidate,
                           selection='ETTh1 H96 validation only')
    if stage == 'screen':
        config['training']['epochs'] = 8
    # Bound the per-channel activation population on large datasets without dropping observations.
    config['visualization'] = {'enabled': False}
    data_file = Path(config['data']['root_path']) / config['data']['files'][dataset]
    data_sha = sha(data_file)
    frozen = json.loads((ROOT / 'protocol.json').read_text())['input_hashes']
    # Source digests cover configuration; bind each dataset to its study-start bytes as well.
    if frozen[str(data_file)] != data_sha:
        raise RuntimeError('Dataset changed since protocol freeze: ' + str(data_file))
    path = ROOT / stage / candidate['id'] / dataset / f'H{horizon}' / f'seed{seed}'
    signature = dict(config=copy.deepcopy(config), candidate=candidate, dataset=dataset,
                     horizon=horizon, seed=seed, data_sha256=data_sha, code_sha256=source_hash)
    marker = check_fingerprint(path, signature)
    if (path / 'validation_complete.json').exists():
        record = json.loads((path / 'validation_complete.json').read_text())
        from scripts.export_architecture_study import verify
        verify([record])
        return record
    path.mkdir(parents=True, exist_ok=True)
    atomic_json(path / 'provenance.json', marker)
    atomic_json(ROOT / 'current_job.json', dict(stage=stage, candidate=candidate['id'], dataset=dataset,
                                               horizon=horizon, seed=seed, source_dir=str(path), status='training'))
    set_seed(seed)
    bundle = build_data_bundle(config, dataset, horizon)
    config['data_scaler'] = bundle.scaler.state_dict()
    config['feature_columns'] = bundle.feature_columns
    model = build_candidate(config, horizon, candidate).to('cuda:0')
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    resolved = dict(config=config, dataset=dataset, pred_len=horizon, seed=seed,
                    n_vars=bundle.n_vars, n_rows=bundle.n_rows, feature_columns=bundle.feature_columns,
                    split_bounds=bundle.split_bounds, scaler=bundle.scaler.state_dict(), n_params=parameters)
    atomic_json(path / 'resolved_config.json', resolved)
    logger = create_logger(path / 'train.log')
    logger.info(f'Architecture study {stage} {candidate["id"]} {dataset} H{horizon} seed{seed}; '
                f'environment={sys.prefix}; physical GPU={os.environ.get("CUDA_VISIBLE_DEVICES")}')
    train = config['training']
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train['learning_rate']),
                                 weight_decay=float(train['weight_decay']))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=train['scheduler_factor'], patience=train['scheduler_patience'],
        min_lr=train['min_learning_rate'])
    trainer = Trainer(model, optimizer, scheduler, torch.device('cuda:0'), config, path, logger)
    checkpoint_path, best, seconds = trainer.fit(bundle.train_loader, bundle.val_loader)
    cp = load_checkpoint(checkpoint_path, model, map_location='cuda:0')
    if source_digest() != source_hash or sha(data_file) != data_sha:
        raise RuntimeError('Inputs changed during training')
    record = dict(candidate_id=candidate['id'], candidate=candidate, dataset=dataset, horizon=horizon,
                  seed=seed, stage=stage, best_val_pinball=best, best_epoch=cp['epoch'],
                  training_seconds=seconds, n_params=parameters, source_dir=str(path),
                  data_sha256=data_sha, code_sha256=source_hash,
                  checkpoint_sha256=sha(checkpoint_path), selection_metric='validation pinball',
                  test_evaluated=False)
    atomic_json(path / 'validation_complete.json', record)
    del cp, trainer, optimizer, scheduler, model, bundle
    torch.cuda.empty_cache()
    return record


def test_job(record, source_hash):
    path = Path(record['source_dir'])
    if (path / 'complete.json').exists():
        completed = json.loads((path / 'complete.json').read_text())
        if completed['checkpoint_sha256'] != sha(path / 'best_checkpoint.pt'):
            raise RuntimeError('Checkpoint changed: ' + str(path))
        from scripts.export_architecture_study import verify
        verify([completed])
        return completed
    if not (ROOT / 'selection.json').exists():
        raise RuntimeError('Test evaluation requires frozen selection')
    if source_digest() != source_hash:
        raise RuntimeError('Source changed before evaluation')
    config = json.loads((path / 'resolved_config.json').read_text())['config']
    data_file = Path(config['data']['root_path']) / config['data']['files'][record['dataset']]
    if sha(data_file) != record['data_sha256'] or sha(path / 'best_checkpoint.pt') != record['checkpoint_sha256']:
        raise RuntimeError('Data or checkpoint changed before evaluation')
    bundle = build_data_bundle(config, record['dataset'], record['horizon'])
    model = build_candidate(config, record['horizon'], record['candidate']).to('cuda:0')
    load_checkpoint(path / 'best_checkpoint.pt', model, map_location='cuda:0')
    population = PopulationLoader(bundle.test_loader)
    metrics = evaluate_test(model, population, torch.device('cuda:0'), bundle.scaler,
                            config['evaluation']['channel_chunk_size'], config['model']['quantiles'], path)
    expected_windows = len(bundle.test_loader.dataset)
    expected_elements = expected_windows * record['horizon'] * bundle.n_vars
    if population.windows != expected_windows or population.elements != expected_elements:
        raise RuntimeError('Evaluation omitted test windows or variables')
    method = 'hadt' if record['candidate']['decoder'] == 'deformable' else 'direct'
    samples(model, bundle, torch.device('cuda:0'), record['horizon'], path, method)
    completed = dict(record, **metrics)
    completed['test_evaluated'] = True
    completed['test_windows'] = population.windows
    completed['test_elements'] = population.elements
    completed['selection_sha256'] = sha(ROOT / 'selection.json')
    atomic_json(path / 'complete.json', completed)
    del model, bundle
    torch.cuda.empty_cache()
    return completed


class PopulationLoader:
    """Count the population actually traversed by the existing full-channel evaluator."""
    def __init__(self, loader):
        self.loader = loader
        self.windows = 0
        self.elements = 0

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        for x, y in self.loader:
            self.windows += len(y)
            self.elements += y.numel()
            yield x, y


def evaluation_queue(winner):
    methods = list(dict.fromkeys([winner, 'transformer3', 'linear']))
    jobs = []
    # First confirm transfer to a related dataset, then fill the missing long horizons.
    settings = [('ETTh1', 96), ('ETTh2', 96)]
    settings += [(dataset, horizon) for horizon in [336, 720, 192] for dataset in ['ETTh1', 'ETTh2']]
    settings += [(dataset, 96) for dataset in ['ETTm1', 'ETTm2', 'Weather', 'Electricity', 'Traffic']]
    settings += [(dataset, horizon) for horizon in [192, 336, 720]
                 for dataset in ['ETTm1', 'ETTm2', 'Weather', 'Electricity', 'Traffic']]
    for dataset, horizon in settings:
        for method in methods:
            for seed in SEEDS:
                jobs.append(dict(dataset=dataset, horizon=horizon, candidate_id=method, seed=seed))
    return jobs


def run(stage):
    ROOT.mkdir(parents=True, exist_ok=True)
    source_hash = source_digest()
    base_config = configuration('ETTh1', 'hadt')
    input_paths = [Path('configs/etth1_gpu3.yaml'), Path('configs/traffic_gpu3.yaml')]
    input_paths += [Path(base_config['data']['root_path']) / name for name in base_config['data']['files'].values()]
    protocol = dict(version='architecture-v1', source_hash=source_hash, candidates=candidates(),
                    screening_epochs=8, confirmation_epochs=30, seeds=SEEDS,
                    selection_dataset='ETTh1', selection_horizon=96,
                    selection_rule='top 3 screen validation + original; minimum mean confirmation validation',
                    GPU='3', python=sys.executable, created_utc=datetime.now(timezone.utc).isoformat())
    protocol_file = ROOT / 'protocol.json'
    if protocol_file.exists():
        old = json.loads(protocol_file.read_text())
        if old['source_hash'] != source_hash or old['candidates'] != candidates():
            raise RuntimeError('Study source or candidate protocol changed; use a new output namespace')
        verify_inputs(old['input_hashes'])
    else:
        protocol['input_hashes'] = freeze_inputs(input_paths)
        atomic_json(protocol_file, protocol)
    screen = []
    for candidate in candidates():
        screen.append(train_job('screen', candidate, 'ETTh1', 96, 2024, source_hash))
        export()
    atomic_json(ROOT / 'screen_records.json', screen)
    if stage == 'screen':
        return
    shortlist = [r['candidate_id'] for r in sorted(screen, key=lambda r: (r['best_val_pinball'], r['candidate_id']))[:3]]
    shortlist = list(dict.fromkeys(shortlist + ['transformer3']))
    by_id = {c['id']: c for c in candidates()}
    confirmation = []
    for candidate_id in shortlist:
        for seed in SEEDS:
            confirmation.append(train_job('confirm', by_id[candidate_id], 'ETTh1', 96, seed, source_hash))
            export()
    selected = select_by_validation(confirmation, SEEDS)
    verify_inputs(json.loads(protocol_file.read_text())['input_hashes'])
    selection = dict(**selected, candidate=by_id[selected['candidate_id']],
                     rule=protocol['selection_rule'], shortlist=shortlist, source_hash=source_hash,
                     confirmation_records=confirmation, test_used=False)
    selection_file = ROOT / 'selection.json'
    if selection_file.exists() and json.loads(selection_file.read_text()) != selection:
        raise RuntimeError('Frozen selection differs')
    if not selection_file.exists():
        atomic_json(selection_file, selection)
    export()
    if stage == 'confirm':
        return
    queue = evaluation_queue(selected['candidate_id'])
    atomic_json(ROOT / 'evaluation_queue.json', queue)
    for index, job in enumerate(queue):
        candidate_id = job['candidate_id']
        job_stage = 'confirm' if job['dataset'] == 'ETTh1' and job['horizon'] == 96 and candidate_id in shortlist else 'transfer'
        record = train_job(job_stage, by_id[candidate_id], job['dataset'], job['horizon'], job['seed'], source_hash)
        test_job(record, source_hash)
        atomic_json(ROOT / 'progress.json', dict(status='running', completed_jobs=index + 1, total_jobs=len(queue),
                                                last_completed=job))
        export()
    atomic_json(ROOT / 'progress.json', dict(status='complete', completed_jobs=len(queue), total_jobs=len(queue)))
    atomic_json(ROOT / 'current_job.json', dict(status='complete'))
    export()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['screen', 'confirm', 'full'], default='full')
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '3' or os.environ.get('CUDA_DEVICE_ORDER') != 'PCI_BUS_ID':
        raise RuntimeError('Physical GPU3 isolation required')
    if Path(sys.prefix).name != 'djjTF' or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('djjTF and exactly one visible CUDA device required')
    torch.set_num_threads(4)
    ROOT.mkdir(parents=True, exist_ok=True)
    with open(ROOT / 'worker.lock', 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock.write(str(os.getpid()))
        lock.flush()
        try:
            if (ROOT / 'failure.json').exists():
                (ROOT / 'failure.json').unlink()
            run(args.stage)
        except Exception as error:
            atomic_json(ROOT / 'failure.json', dict(error=repr(error), timestamp=time.time()))
            current = json.loads((ROOT / 'current_job.json').read_text()) if (ROOT / 'current_job.json').exists() else {}
            atomic_json(ROOT / 'current_job.json', dict(current, status='failed', error=repr(error)))
            raise


if __name__ == '__main__':
    main()
