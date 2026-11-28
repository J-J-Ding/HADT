"""Drain issued jobs, then install an ETT-first scheduling barrier without lost training."""
import fcntl
import json
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.provenance import atomic_json, sha256

OUT = ROOT / 'outputs/paper_seven_datasets_20261126'
STATE = OUT / 'scheduling_priority.json'
DRIVER = ROOT / 'scripts/extend_paper_datasets.py'
ARCHIVE = OUT / 'protocol_history/ett_first_20261127'


def run_path(job):
    return OUT / job['dataset'] / job['name'] / f"H{job['horizon']}" / f"seed{job['seed']}"


def controller_processes():
    result = []
    for entry in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            if entry.stat().st_uid != os.getuid():
                continue
            argv = entry.read_bytes().split(b'\0')
            if len(argv) < 2:
                continue
            script = Path(argv[1].decode())
            if not script.is_absolute():
                script = (entry.parent / 'cwd').resolve() / script
            if script.resolve() == DRIVER:
                result.append((int(entry.parent.name), b'--worker' in argv))
        except (FileNotFoundError, PermissionError, ProcessLookupError, UnicodeDecodeError):
            continue
    return result


def main():
    os.chdir(ROOT)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    with (OUT / 'queue.lock').open('a') as lock:
        # Existing workers finish their current run; no further job can be claimed.
        fcntl.flock(lock, fcntl.LOCK_EX)
        queue = json.loads((OUT / 'queue.json').read_text())
        issued = queue['jobs'][:queue['next']]
        for source in (OUT / 'queue.json', OUT / 'protocol_v2.json', DRIVER):
            target = ARCHIVE / source.name
            if not target.exists():
                target.write_bytes(source.read_bytes())
        while True:
            remaining = [j for j in issued if not (run_path(j) / 'complete.json').exists()]
            atomic_json(STATE, dict(status='draining_current_jobs', pid=os.getpid(),
                priority=['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2'],
                deferred=['Traffic', 'Weather', 'Electricity'], current_remaining=remaining,
                training_settings_changed=False))
            if not remaining:
                break
            processes = controller_processes()
            worker_pids = {p for p, worker in processes if worker}
            failures = [json.loads(f.read_text()) for f in (OUT / 'workers').glob('gpu*.json')]
            if not worker_pids or any(s.get('status') == 'failed' for s in failures):
                atomic_json(STATE, dict(status='needs_attention', reason='A current worker stopped before its run completed.', remaining=remaining))
                raise RuntimeError('Current training failed before the scheduling transition')
            time.sleep(15)

        source = DRIVER.read_text()
        old_sort = "jobs.sort(key=lambda j:(0 if j['dataset'] in ('Traffic','Electricity') and j['name']=='timemixer' else 1, -j['horizon'],j['dataset'],j['seed'],j['name']))"
        new_sort = "jobs.sort(key=lambda j:(0 if j['dataset'].startswith('ETT') else 1, 0 if j['dataset'] in ('Traffic','Electricity') and j['name']=='timemixer' else 1, -j['horizon'],j['dataset'],j['seed'],j['name']))"
        anchor = "            if i>=len(q['jobs']):return\n            job=q['jobs'][i];q['next']+=1;atomic_json(root/'queue.json',q)"
        replacement = """            if i>=len(q['jobs']):return
            job=q['jobs'][i]
            # All ETT runs must finish before any new large-dataset run starts.
            if not job['dataset'].startswith('ETT'):
                remaining=[j for j in q['jobs'] if j['dataset'].startswith('ETT') and not (path_for(root,j)/'complete.json').exists()]
                if remaining:
                    atomic_json(status,dict(status='waiting_for_ett',remaining=len(remaining),pid=os.getpid()))
                    time.sleep(5)
                    continue
            q['next']+=1;atomic_json(root/'queue.json',q)"""
        assert source.count(old_sort) == source.count(anchor) == 1
        updated = source.replace(old_sort, new_sort).replace(anchor, replacement)
        compile(updated, str(DRIVER), 'exec')
        protocol = json.loads((OUT / 'protocol_v2.json').read_text())
        assert sha256(DRIVER) == protocol['driver']
        temp = DRIVER.with_suffix('.py.priority.tmp')
        temp.write_text(updated)
        temp.replace(DRIVER)
        protocol['driver'] = sha256(DRIVER)
        atomic_json(OUT / 'protocol_v2.json', protocol)

        # Every issued run is complete. Stop only this project's idle queue workers
        # and controller; the detached supervisor restarts the amended scheduler.
        processes = controller_processes()
        for pid, worker in sorted(processes, key=lambda item: item[1]):
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        atomic_json(STATE, dict(status='ett_first_installed', pid=os.getpid(),
            priority=['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2'],
            deferred=['Traffic', 'Weather', 'Electricity'],
            strict_completion_barrier=True, training_settings_changed=False,
            driver_sha256=protocol['driver'], preserved_runs=len(issued),
            previous_protocol=str(ARCHIVE / 'protocol_v2.json')))


if __name__ == '__main__':
    main()
