import json
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest


class GpuSchedulerTests(unittest.TestCase):
    def test_idle_requires_low_utilization_memory_and_no_compute_jobs(self):
        from scripts.gpu_queue_scheduler import idle_gpu
        gpu = dict(index=0, uuid='gpu0', memory_used=20, memory_total=32000, utilization=0)
        self.assertTrue(idle_gpu(gpu, []))
        self.assertFalse(idle_gpu(dict(gpu, utilization=90), []))
        self.assertFalse(idle_gpu(dict(gpu, memory_used=1400), []))
        self.assertFalse(idle_gpu(gpu, [dict(uuid='gpu0', pid=12, name='python', memory_used=100)]))
        self.assertTrue(idle_gpu(gpu, [dict(uuid='gpu0', pid=12, name='/usr/lib/snapd/snapd-desktop-integration', memory_used=10)]))

    def test_two_observations_and_assigned_cards_are_not_idle_candidates(self):
        from scripts.gpu_queue_scheduler import eligible_gpus
        cards = [dict(index=i, uuid=f'gpu{i}', memory_used=20, memory_total=32000, utilization=0) for i in range(4)]
        streak = {}
        self.assertEqual(eligible_gpus(cards, [], {3}, streak), [])
        self.assertEqual([x['index'] for x in eligible_gpus(cards, [], {3}, streak)], [0, 1, 2])
        busy = [dict(uuid='gpu1', pid=9, name='python', memory_used=10)]
        self.assertEqual([x['index'] for x in eligible_gpus(cards, busy, {3}, streak)], [0, 2])
        self.assertEqual([x['index'] for x in eligible_gpus(cards, [], {3}, streak)], [0, 2])
        empty_streak = {}
        eligible_gpus(cards, [], {3}, empty_streak)
        self.assertEqual(eligible_gpus(cards, [], {3}, empty_streak, observe=False), [])

    def test_publication_snapshot_blocks_new_completion_commits(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lock_path = root / 'publication.lock'
            target = root / 'complete.json'
            with lock_path.open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                process = subprocess.Popen([sys.executable, '-c',
                    'import sys; from pathlib import Path; from scripts.gpu_queue_scheduler import commit_record; '
                    'commit_record(Path(sys.argv[1]), {"complete":True}, Path(sys.argv[2]))', str(target), str(lock_path)])
                try:
                    time.sleep(.2)
                    self.assertIsNone(process.poll())
                    self.assertFalse(target.exists())
                finally:
                    fcntl.flock(lock, fcntl.LOCK_UN)
                    process.wait(timeout=10)
            self.assertEqual(json.loads(target.read_text()), {'complete': True})

    def test_claim_lock_is_exclusive_and_released(self):
        from scripts.gpu_queue_scheduler import claim_job
        with tempfile.TemporaryDirectory() as temporary:
            job = dict(dataset='ETTh1', horizon=96, candidate_id='linear', seed=2024)
            first = claim_job(Path(temporary), job)
            self.assertIsNotNone(first)
            self.assertIsNone(claim_job(Path(temporary), job))
            first.close()
            second = claim_job(Path(temporary), job)
            self.assertIsNotNone(second)
            second.close()

    def test_gpu_reservation_survives_parent_close_before_worker_metadata(self):
        from scripts.gpu_queue_scheduler import claim_gpu
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reservation = claim_gpu(root, 3)
            process = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'],
                                       pass_fds=(reservation.fileno(),))
            reservation.close()
            try:
                self.assertIsNone(claim_gpu(root, 3))
            finally:
                process.terminate()
                process.wait(timeout=5)
            released = claim_gpu(root, 3)
            self.assertIsNotNone(released)
            released.close()

    def test_polling_interval_below_thirty_seconds_is_rejected(self):
        script = Path(__file__).resolve().parents[1] / 'scripts/gpu_queue_scheduler.py'
        result = subprocess.run([sys.executable, str(script), '--interval', '5'], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('thirty seconds', result.stderr)

    def test_pending_jobs_use_frozen_stage_and_ignore_complete_jobs(self):
        from scripts.gpu_queue_scheduler import run_path, pending_indices
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selection = dict(shortlist=['mlp64_lr3e4', 'transformer3'])
            queue = [dict(dataset='ETTh1', horizon=96, candidate_id='mlp64_lr3e4', seed=2024),
                     dict(dataset='Traffic', horizon=720, candidate_id='linear', seed=2025)]
            first = run_path(root, queue[0], selection)
            self.assertIn('confirm', first.parts)
            first.mkdir(parents=True)
            (first / 'complete.json').write_text('{}')
            self.assertEqual(pending_indices(root, queue, selection), [1])

    def test_handover_waits_for_completion_then_retires_only_owned_parent(self):
        from scripts.gpu_queue_scheduler import handover_legacy
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            script = project / 'run_architecture_study.py'
            script.write_text('import time\ntime.sleep(120)\n')
            run = project / 'outputs/architecture_study/transfer/run'
            run.mkdir(parents=True)
            current = project / 'current.json'
            current.write_text(json.dumps(dict(source_dir=str(run), status='training')))
            process = subprocess.Popen([sys.executable, '-u', str(script)], cwd=project)
            try:
                time.sleep(.1)
                self.assertFalse(handover_legacy(process.pid, current, project))
                self.assertIsNone(process.poll())
                (run / 'complete.json').write_text('{}')
                # A stale current-job marker must not hide a cached-training,
                # unfinished test evaluation in the next queue item.
                study = project / 'outputs/architecture_study'
                queue = [dict(dataset='Fixture', horizon=5, candidate_id='linear', seed=s) for s in [2024, 2025]]
                (study / 'evaluation_queue.json').write_text(json.dumps(queue))
                (study / 'selection.json').write_text(json.dumps(dict(shortlist=[])))
                (study / 'progress.json').write_text(json.dumps(dict(completed_jobs=1)))
                upcoming = study / 'transfer/linear/Fixture/H5/seed2025'
                upcoming.mkdir(parents=True)
                (upcoming / 'validation_complete.json').write_text('{}')
                self.assertFalse(handover_legacy(process.pid, current, project))
                self.assertIsNone(process.poll())
                (upcoming / 'complete.json').write_text('{}')
                self.assertTrue(handover_legacy(process.pid, current, project))
                process.wait(timeout=5)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()


if __name__ == '__main__':
    unittest.main()
