import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


class RunForecastTests(unittest.TestCase):
    def test_render_all_saved_variables_and_windows_and_refresh_changed_source(self):
        from scripts.plot_run_forecasts import render_run, write_gallery
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            history = np.arange(3 * 12 * 2, dtype=float).reshape(3, 12, 2)
            target = np.arange(3 * 5 * 2, dtype=float).reshape(3, 5, 2)
            quantiles = target[..., None] + np.array([-3, -2, -1, 0, 1, 2, 3])
            source = run / 'forecast_samples.npz'
            def save():
                np.savez_compressed(source, history=history, target=target, quantiles=quantiles,
                                    sample_indices=[0, 7, 14], channel_indices=[0, 9],
                                    feature_columns=['a', 'b / unsafe'])
            save()
            record = dict(source_dir=str(run), dataset='Fixture', horizon=5,
                          candidate_id='mlp64_lr3e4', seed=2024)
            result = render_run(record)
            output = run / 'figures' / 'predictions'
            self.assertTrue((output / 'overview.png').is_file())
            for channel in [0, 9]:
                self.assertTrue((output / f'channel_{channel:03d}.png').is_file())
                self.assertTrue((output / f'channel_{channel:03d}.pdf').is_file())
            metadata = json.loads((output / 'plot_complete.json').read_text())
            self.assertEqual(metadata['sample_indices'], [0, 7, 14])
            self.assertEqual(metadata['channel_indices'], [0, 9])
            missing = dict(record, source_dir=str(run / 'missing'), dataset='Pending')
            gallery = run / 'gallery.html'
            self.assertEqual(write_gallery([record, missing], gallery), 1)
            content = gallery.read_text()
            self.assertIn('data-dataset="Fixture"', content)
            self.assertIn('figures/predictions/index.html', content)
            self.assertNotIn('data-dataset="Pending"', content)
            before = (output / 'overview.png').stat().st_mtime_ns
            render_run(record)
            self.assertEqual(before, (output / 'overview.png').stat().st_mtime_ns)
            target[0, 0, 0] += 10
            save()
            render_run(record)
            after = json.loads((output / 'plot_complete.json').read_text())
            self.assertNotEqual(metadata['source_sha256'], after['source_sha256'])

    def test_rejects_malformed_arrays_instead_of_drawing_wrong_quantiles(self):
        from scripts.plot_run_forecasts import validate_arrays
        arrays = dict(history=np.zeros((3, 10, 1)), target=np.zeros((3, 5, 1)),
                      quantiles=np.zeros((3, 5, 1, 7)), sample_indices=np.array([0, 1, 2]),
                      channel_indices=np.array([7]), feature_columns=np.array(['x']))
        validate_arrays(arrays)
        arrays['quantiles'][0, 0, 0, 0] = 1
        with self.assertRaises(ValueError):
            validate_arrays(arrays)


if __name__ == '__main__':
    unittest.main()
