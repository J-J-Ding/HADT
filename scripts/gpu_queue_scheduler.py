"""Poll free GPUs and execute the frozen study queue without duplicate jobs.

Model/training implementations remain unchanged. Completion commits share a lock
with the exclusive manuscript publisher, providing a consistent audit snapshot.
"""
import argparse
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT / 'outputs/architecture_study'
sys.path.insert(0, str(PROJECT))
DESKTOP = {'Xorg', 'gnome-shell', 'snapd-desktop-integration'}
MIN_FREE_MIB = 12288


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False))
    temp.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def commit_record(path, value, publication_lock):
    with Path(publication_lock).open('a') as commit:
        fcntl.flock(commit, fcntl.LOCK_SH)
        atomic_json(path, value)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def query_gpus():
    def query(fields, application=False):
        option = '--query-compute-apps=' if application else '--query-gpu='
        result = subprocess.run(['nvidia-smi', option + fields, '--format=csv,noheader,nounits'],
                                check=True, capture_output=True, text=True, timeout=20)
        return list(csv.reader(result.stdout.strip().splitlines())) if result.stdout.strip() else []
    cards = [dict(index=int(row[0]), uuid=row[1].strip(), memory_used=float(row[2]),
                  memory_total=float(row[3]), utilization=float(row[4]))
             for row in query('index,uuid,memory.used,memory.total,utilization.gpu')]
    processes = [dict(pid=int(row[0]), uuid=row[1].strip(), name=row[2].strip(), memory_used=float(row[3]))
                 for row in query('pid,gpu_uuid,process_name,used_gpu_memory', True)]
    if not cards:
        raise RuntimeError('No GPU information returned; refusing allocation')
    return cards, processes


def idle_gpu(gpu, processes):
    if gpu['utilization'] > 10 or gpu['memory_used'] > 1024 or gpu['memory_total'] - gpu['memory_used'] < MIN_FREE_MIB:
        return False
    return not any(p['uuid'] == gpu['uuid'] and not (
        Path(p['name']).name in DESKTOP and p['memory_used'] <= 256) for p in processes)


def eligible_gpus(cards, processes, assigned, streak, observe=True):
    eligible = []
    present = set()
    for card in cards:
        index = card['index']
        present.add(index)
        if index not in range(4) or index in assigned or not idle_gpu(card, processes):
            streak[index] = 0
            continue
        if observe:
            streak[index] = streak.get(index, 0) + 1
        if streak.get(index, 0) >= 2:
            eligible.append(card)
    for index in set(streak) - present:
        streak[index] = 0
    return eligible


def job_key(job):
    return hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest()[:20]


def claim_job(root, job):
    path = Path(root) / 'job_locks' / (job_key(job) + '.lock')
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open('a')
    try:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        stream.close()
        return None
    return stream


def claim_gpu(root, index):
    """Reserve a GPU before spawn; the worker inherits this open description."""
    path = Path(root) / 'gpu_locks' / f'gpu{index}.lock'
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open('a')
    try:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        stream.close()
        return None
    return stream


def run_path(root, job, selection):
    stage = 'confirm' if job['dataset'] == 'ETTh1' and job['horizon'] == 96 and job['candidate_id'] in selection['shortlist'] else 'transfer'
    return Path(root) / stage / job['candidate_id'] / job['dataset'] / f"H{job['horizon']}" / f"seed{job['seed']}"


def pending_indices(root, queue, selection):
    return [i for i, job in enumerate(queue) if not (run_path(root, job, selection) / 'complete.json').exists()]


def alive(pid):
    try:
        status = Path(f'/proc/{pid}/stat').read_text().rsplit(') ', 1)[1].split()[0]
        return status != 'Z'
    except (FileNotFoundError, ProcessLookupError):
        return False


def owned_entrypoint(pid, name, project=PROJECT):
    try:
        args = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
        arguments = [os.fsdecode(arg) for arg in args[1:] if arg]
        while arguments and arguments[0] in ('-u', '-B', '-s', '-S', '-E', '-I'):
            arguments.pop(0)
        return (Path(f'/proc/{pid}').stat().st_uid == os.getuid() and bool(arguments) and
                Path(arguments[0]).name == name and Path(f'/proc/{pid}/cwd').resolve() == Path(project).resolve())
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return False


def adopt_workers(root, active):
    for folder in (Path(root) / 'parallel_workers').glob('*'):
        status = read(folder / 'status.json') if (folder / 'status.json').exists() else {}
        launch = read(folder / 'launch.json') if (folder / 'launch.json').exists() else {}
        value = launch if launch and (status.get('pid') != launch['pid'] or
                         status.get('started_unix_ns', 0) < launch.get('launched_unix_ns', 0)) else status
        if value.get('status') not in ('running', 'starting') or not alive(value['pid']) or not owned_entrypoint(value['pid'], Path(__file__).name):
            continue
        gpu = value['gpu_index']
        if gpu in active:
            if active[gpu]['pid'] != value['pid']:
                raise RuntimeError('Multiple live study workers recorded on one GPU')
            continue
        active[gpu] = dict(pid=value['pid'], job_index=value['job_index'], uuid=value['gpu_uuid'], process=None)


def handover_legacy(pid, current_file, project=PROJECT):
    """Pause only a verified study parent after its current job has committed.

    Existing export children finish while the parent is paused. No incomplete
    training is killed, and no foreign process is signaled.
    """
    if not alive(pid):
        return True
    if not owned_entrypoint(pid, 'run_architecture_study.py', project):
        raise RuntimeError('Legacy PID no longer identifies our study process')
    def finished():
        job = read(current_file)
        path = Path(job.get('source_dir', ''))
        if not path.is_absolute():
            path = Path(project) / path
        if not (Path(project).resolve() in path.resolve().parents and (path / 'complete.json').exists()):
            return False
        # Cached training does not update the frozen runner's current-job marker.
        # Its next queue item may therefore be doing test-only work in the parent.
        study_root = Path(project) / 'outputs/architecture_study'
        if all((study_root / name).exists() for name in ['evaluation_queue.json', 'selection.json', 'progress.json']):
            queue = read(study_root / 'evaluation_queue.json')
            cursor = int(read(study_root / 'progress.json')['completed_jobs'])
            if cursor < len(queue):
                upcoming = run_path(study_root, queue[cursor], read(study_root / 'selection.json'))
                if (upcoming / 'validation_complete.json').exists() and not (upcoming / 'complete.json').exists():
                    return False
        return True
    if not finished():
        return False
    os.kill(pid, signal.SIGSTOP)
    retired = False
    try:
        if not finished():
            return False
        deadline = time.monotonic() + 300
        while True:
            children_file = Path(f'/proc/{pid}/task/{pid}/children')
            children = [int(p) for p in children_file.read_text().split()] if children_file.exists() else []
            if not any(alive(child) for child in children):
                break
            if time.monotonic() > deadline:
                return False
            time.sleep(.25)
        os.kill(pid, signal.SIGTERM)
        os.kill(pid, signal.SIGCONT)
        deadline = time.monotonic() + 10
        while alive(pid) and time.monotonic() < deadline:
            time.sleep(.05)
        if alive(pid):
            raise RuntimeError('Legacy parent has not exited; refusing concurrent dispatch')
        retired = True
        return True
    finally:
        if not retired and alive(pid):
            os.kill(pid, signal.SIGCONT)


def run_one(index, gpu_index, gpu_uuid):
    cards, processes = query_gpus()
    assigned = next((g for g in cards if g['index'] == gpu_index and g['uuid'] == gpu_uuid), None)
    if assigned is None or (gpu_index != 3 and not idle_gpu(assigned, processes)) or assigned['memory_total'] - assigned['memory_used'] < MIN_FREE_MIB:
        return 75
    import torch
    from scripts import run_architecture_study as study
    from experiments.architecture_search import candidates
    if Path(sys.prefix).name != 'djjTF' or os.environ.get('CUDA_VISIBLE_DEVICES') != gpu_uuid or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Worker requires djjTF and exactly its assigned GPU UUID')
    torch.set_num_threads(4)
    protocol, selection = read(ROOT / 'protocol.json'), read(ROOT / 'selection.json')
    queue = read(ROOT / 'evaluation_queue.json')
    if study.source_digest() != protocol['source_hash'] or queue != study.evaluation_queue(selection['candidate_id']):
        raise RuntimeError('Frozen study source or queue changed')
    job = queue[index]
    claim = claim_job(ROOT, job)
    if claim is None:
        return 75
    folder = ROOT / 'parallel_workers' / job_key(job)
    meta = dict(pid=os.getpid(), job_index=index, job=job, gpu_index=gpu_index, gpu_uuid=gpu_uuid,
                cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'], python=sys.executable,
                device_name=torch.cuda.get_device_name(0), controller_sha256=digest(__file__),
                source_hash=protocol['source_hash'], status='running', started_utc=datetime.now(timezone.utc).isoformat(),
                started_unix_ns=time.time_ns())
    original_atomic = study.atomic_json
    def isolated_atomic(path, value):
        if Path(path).resolve() == (ROOT / 'current_job.json').resolve():
            return atomic_json(folder / 'current_job.json', value)
        if Path(path).name in ('validation_complete.json', 'complete.json'):
            return commit_record(path, value, ROOT / 'publication.lock')
        return original_atomic(path, value)
    study.atomic_json = isolated_atomic
    atomic_json(folder / 'status.json', meta)
    path = run_path(ROOT, job, selection)
    reused_training = (path / 'validation_complete.json').exists()
    try:
        stage = path.relative_to(ROOT).parts[0]
        candidate = next(c for c in candidates() if c['id'] == job['candidate_id'])
        record = study.train_job(stage, candidate, job['dataset'], job['horizon'], job['seed'], protocol['source_hash'])
        study.test_job(record, protocol['source_hash'])
        execution_path = path / 'execution.json'
        previous = read(execution_path) if execution_path.exists() else {'attempts': []}
        attempt = dict(meta, status='complete', reused_training=reused_training,
                       checkpoint_sha256=study.sha(path / 'best_checkpoint.pt'),
                       selection_sha256=study.sha(ROOT / 'selection.json'),
                       completed_utc=datetime.now(timezone.utc).isoformat())
        previous['attempts'].append(attempt)
        atomic_json(execution_path, previous)
        atomic_json(folder / 'status.json', attempt)
        return 0
    except Exception as error:
        atomic_json(folder / 'status.json', dict(meta, status='failed', error=repr(error), traceback=traceback.format_exc()))
        raise
    finally:
        claim.close()


def monitor(interval):
    from scripts import run_architecture_study as study
    from scripts.export_architecture_study import verify, records
    from experiments.architecture_search import select_by_validation
    protocol, selection = read(ROOT / 'protocol.json'), read(ROOT / 'selection.json')
    queue = read(ROOT / 'evaluation_queue.json')
    if study.source_digest() != protocol['source_hash'] or queue != study.evaluation_queue(selection['candidate_id']):
        raise RuntimeError('Frozen source/queue changed')
    if select_by_validation(selection['confirmation_records'], study.SEEDS)['candidate_id'] != selection['candidate_id']:
        raise RuntimeError('Frozen validation selection disagrees')
    study.verify_inputs(protocol['input_hashes'])
    verify(records('complete'))
    legacy_pid = int((ROOT / 'worker.pid').read_text()) if (ROOT / 'worker.pid').exists() else 0
    legacy = legacy_pid if owned_entrypoint(legacy_pid, 'run_architecture_study.py') and alive(legacy_pid) else None
    active, streak = {}, {}
    # Surviving children retain their per-job locks if a prior supervisor exits.
    adopt_workers(ROOT, active)
    study_lock = None
    publisher = None
    published_count = -1
    publishing_count = -1
    halt = None
    last_idle_observation = -float('inf')
    policy = dict(allowed_physical_gpu_indices=[0, 1, 2, 3], base_gpu_index=3,
                  poll_interval_seconds=interval, idle_observations=2, idle_max_utilization_percent=10,
                  idle_max_memory_mib=1024, minimum_free_memory_mib=MIN_FREE_MIB,
                  idle_requires_no_non_desktop_compute_process=True,
                  authorization='User requested periodic idle detection and parallel execution on available GPUs.',
                  original_protocol_sha256=digest(ROOT / 'protocol.json'),
                  frozen_queue_sha256=digest(ROOT / 'evaluation_queue.json'),
                  supervisor_source_sha256=digest(__file__), training_source_hash=protocol['source_hash'])
    atomic_json(ROOT / 'execution_policy.json', policy)
    atomic_json(ROOT / 'scheduler.pid.json', dict(pid=os.getpid()))
    while True:
        adopt_workers(ROOT, active)
        try:
            cards, processes = query_gpus()
        except Exception as error:
            streak.clear()
            atomic_json(ROOT / 'scheduler_state.json', dict(status='gpu_query_failed', error=repr(error), pid=os.getpid()))
            time.sleep(interval)
            continue
        for gpu, entry in list(active.items()):
            process = entry['process']
            code = process.poll() if process is not None else (None if alive(entry['pid']) else 0)
            if code is None:
                continue
            marker = ROOT / 'parallel_workers' / job_key(queue[entry['job_index']]) / 'status.json'
            result = read(marker) if marker.exists() else {}
            if code != 75 and (code != 0 or result.get('status') != 'complete'):
                halt = dict(reason='worker_failed', job=queue[entry['job_index']], worker=result, exit_code=code)
            del active[gpu]
        pending = pending_indices(ROOT, queue, selection)
        completed = len(queue) - len(pending)
        assigned = set(active) | ({3} if legacy else set())
        observe = time.monotonic() - last_idle_observation >= interval
        free = eligible_gpus(cards, processes, assigned, streak, observe)
        if observe:
            last_idle_observation = time.monotonic()
        if legacy and not alive(legacy):
            if (ROOT / 'failure.json').exists():
                halt = dict(reason='legacy_failed', error=read(ROOT / 'failure.json'))
            legacy = None
        if legacy and free and not halt:
            if handover_legacy(legacy, ROOT / 'current_job.json'):
                legacy = None
                pending = pending_indices(ROOT, queue, selection)
                completed = len(queue) - len(pending)
                atomic_json(ROOT / 'legacy_handover.json', dict(retired_pid=legacy_pid, status='complete', completed_jobs=completed))
        if not legacy and study_lock is None:
            study_lock = (ROOT / 'worker.lock').open('a')
            fcntl.flock(study_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            (ROOT / 'worker.pid').write_text(str(os.getpid()))
        if publisher is not None and publisher.poll() is not None:
            if publisher.returncode:
                halt = dict(reason='publication_failed', exit_code=publisher.returncode,
                            log=str(ROOT / 'scheduler_export.log'))
            else:
                published_count = publishing_count
            publisher = None
        if not legacy and completed != published_count and publisher is None and not halt:
            with (ROOT / 'scheduler_export.log').open('a') as log:
                publisher = subprocess.Popen([sys.executable, 'scripts/export_architecture_study.py'],
                                              cwd=PROJECT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''),
                                              stdout=log, stderr=subprocess.STDOUT)
            publishing_count = completed
        if not legacy and not halt:
            available = {g['index']: g for g in free}
            # Preserve the user's existing GPU3 allocation; additions require true idle cards.
            for card in cards:
                if card['index'] == 3 and 3 not in active and card['memory_total'] - card['memory_used'] >= MIN_FREE_MIB:
                    available[3] = card
            for gpu, card in sorted(available.items()):
                if gpu in active:
                    continue
                # A boundary handover or publication can take time; never dispatch
                # using an old idle observation without rechecking the device.
                try:
                    fresh_cards, fresh_processes = query_gpus()
                except Exception as error:
                    streak[gpu] = 0
                    print(f'GPU_REFRESH_FAILED {error!r}', flush=True)
                    continue
                card = next((g for g in fresh_cards if g['index'] == gpu), None)
                if card is None or card['memory_total'] - card['memory_used'] < MIN_FREE_MIB or (gpu != 3 and not idle_gpu(card, fresh_processes)):
                    streak[gpu] = 0
                    continue
                index = None
                for candidate_index in pending:
                    if (run_path(ROOT, queue[candidate_index], selection) / 'complete.json').exists():
                        continue
                    if any(e['job_index'] == candidate_index for e in active.values()):
                        continue
                    probe = claim_job(ROOT, queue[candidate_index])
                    if probe is not None:
                        probe.close()
                        index = candidate_index
                        break
                if index is None:
                    break
                gpu_reservation = claim_gpu(ROOT, gpu)
                if gpu_reservation is None:
                    continue
                env = dict(os.environ, CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=card['uuid'],
                           OMP_NUM_THREADS='4', MPLCONFIGDIR='/tmp/hadt_parallel_mpl')
                folder = ROOT / 'parallel_workers' / job_key(queue[index])
                folder.mkdir(parents=True, exist_ok=True)
                try:
                    launched_unix_ns = time.time_ns()
                    with (folder / 'worker.log').open('a') as log:
                        process = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()),
                                                    '--job-index', str(index), '--gpu-index', str(gpu), '--gpu-uuid', card['uuid'],
                                                    '--gpu-lock-fd', str(gpu_reservation.fileno())],
                                                   cwd=PROJECT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                                   pass_fds=(gpu_reservation.fileno(),))
                    atomic_json(folder / 'launch.json', dict(status='starting', pid=process.pid, gpu_index=gpu,
                                job_index=index, gpu_uuid=card['uuid'], launched_unix_ns=launched_unix_ns))
                finally:
                    gpu_reservation.close()
                active[gpu] = dict(pid=process.pid, job_index=index, uuid=card['uuid'], process=process)
                print(f'DISPATCH GPU{gpu} PID{process.pid} JOB{index + 1} {queue[index]}', flush=True)
        workers = [dict(pid=e['pid'], gpu_index=gpu, gpu_uuid=e['uuid'], job=queue[e['job_index']]) for gpu, e in active.items()]
        if legacy:
            workers.append(dict(pid=legacy, gpu_index=3, legacy=True, job=read(ROOT / 'current_job.json')))
        mode = 'attention_required' if halt else ('watching_legacy' if legacy else 'running_parallel')
        state = dict(status=mode, pid=os.getpid(), completed_jobs=completed, total_jobs=len(queue),
                     workers=workers, gpus=cards, eligible_spare_gpu_indices=[g['index'] for g in free],
                     idle_streaks=streak, last_check_utc=datetime.now(timezone.utc).isoformat(), interval_seconds=interval,
                     publication_running=publisher is not None, error=halt)
        atomic_json(ROOT / 'scheduler_state.json', state)
        if not legacy:
            atomic_json(ROOT / 'current_job.json', dict(status=mode, workers=workers, error=halt))
            atomic_json(ROOT / 'progress.json', dict(status=mode, completed_jobs=completed, total_jobs=len(queue)))
        print(f'POLL {completed}/{len(queue)} {mode} active={[w["gpu_index"] for w in workers]} spare={[g["index"] for g in free]}', flush=True)
        if not pending and not active and publisher is None and (not legacy or not alive(legacy)) and not halt:
            atomic_json(ROOT / 'scheduler_state.json', dict(state, status='complete'))
            atomic_json(ROOT / 'current_job.json', dict(status='complete'))
            atomic_json(ROOT / 'progress.json', dict(status='complete', completed_jobs=len(queue), total_jobs=len(queue)))
            return
        time.sleep(2 if legacy and free else interval)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--interval', type=int, default=30)
    parser.add_argument('--once', action='store_true', help='Read-only GPU availability check; never launches work')
    parser.add_argument('--job-index', type=int)
    parser.add_argument('--gpu-index', type=int)
    parser.add_argument('--gpu-uuid')
    parser.add_argument('--gpu-lock-fd', type=int)
    args = parser.parse_args()
    if args.once:
        cards, processes = query_gpus()
        print(json.dumps(dict(gpus=cards, idle_gpu_indices=[g['index'] for g in cards if idle_gpu(g, processes)]), indent=2))
        return
    if Path(sys.prefix).name != 'djjTF' or args.interval < 30:
        raise RuntimeError('Use djjTF with a polling interval of at least thirty seconds')
    if args.job_index is not None:
        if args.gpu_index not in range(4) or not args.gpu_uuid:
            raise RuntimeError('Worker needs an allowed physical GPU and UUID')
        if args.gpu_lock_fd is None:
            reservation = claim_gpu(ROOT, args.gpu_index)
            if reservation is None:
                raise SystemExit(75)
        else:
            reservation = os.fdopen(args.gpu_lock_fd, 'a')
            expected = (ROOT / 'gpu_locks' / f'gpu{args.gpu_index}.lock').stat()
            actual = os.fstat(reservation.fileno())
            if (actual.st_dev, actual.st_ino) != (expected.st_dev, expected.st_ino):
                raise RuntimeError('GPU reservation descriptor does not match assigned device')
        with reservation:
            raise SystemExit(run_one(args.job_index, args.gpu_index, args.gpu_uuid))
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / 'scheduler.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        monitor(args.interval)


if __name__ == '__main__':
    main()
