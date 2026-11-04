import json
from pathlib import Path
import tempfile
import unittest


class TestTrafficProvenance(unittest.TestCase):
    def test_changed_data_and_completed_untracked_runs_are_rejected(self):
        from scripts.run_traffic_depth_comparison import check_dataset_fingerprint
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            check_dataset_fingerprint(run, 'original_digest')
            check_dataset_fingerprint(run, 'original_digest')
            with self.assertRaisesRegex(RuntimeError, 'Dataset fingerprint mismatch'):
                check_dataset_fingerprint(run, 'changed_digest')
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run/'metrics.json').write_text('{}')
            with self.assertRaisesRegex(RuntimeError, 'Missing dataset fingerprint'):
                check_dataset_fingerprint(run, 'current_digest')
