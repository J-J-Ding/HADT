import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import numpy as np

from experiments.architecture_search import candidates, build_candidate, select_by_validation
from scripts.run_paper_experiments import configuration


class ArchitectureSearchTests(unittest.TestCase):
    def test_saved_observation_audit_accounts_for_float32_roundtrip_and_rejects_corruption(self):
        from scripts.validate_architecture_study import assert_saved_observations
        mean = np.array([405.123456789], dtype=np.float64)
        scale = np.array([123.456789012], dtype=np.float64)
        raw = np.array([[0.], [100.], [400.]], dtype=np.float32)
        # Reproduce the actual dataset transform and saved original-unit representation.
        from data_provider.scaler import StandardScaler
        scaler = StandardScaler()
        scaler.mean_, scaler.scale_ = mean, scale
        saved = scaler.inverse_transform(scaler.transform(raw))
        assert_saved_observations(saved, raw, mean, scale)
        corrupted = saved.copy()
        corrupted[0, 0] += .001
        with self.assertRaises(AssertionError):
            assert_saved_observations(corrupted, raw, mean, scale)
        with self.assertRaises(AssertionError):
            assert_saved_observations(saved[::-1], raw, mean, scale)

    def test_candidates_order_shape_gradients_and_channel_independence(self):
        torch.set_num_threads(2)
        for candidate in candidates():
            with self.subTest(candidate=candidate['id']):
                cfg = configuration('ETTh1', 'hadt')
                cfg['data']['seq_len'] = 32
                cfg['model'].update(d_model=16, n_heads=4, d_ff=32,
                                    patch_len=8, stride=8, n_points=4)
                model = build_candidate(cfg, 5, candidate, small_test=True).eval()
                x = torch.randn(2, 32, 2, requires_grad=True)
                y = model(x)
                self.assertEqual(tuple(y.shape), (2, 5, 2, 7))
                self.assertTrue(torch.isfinite(y).all())
                self.assertTrue((y.diff(dim=-1) >= 0).all())
                torch.testing.assert_close(y[:, :, :1], model(x[:, :, :1]), atol=1e-5, rtol=1e-5)
                y.square().mean().backward()
                self.assertTrue(torch.isfinite(x.grad).all())
                for name, parameter in model.named_parameters():
                    if parameter.requires_grad:
                        self.assertIsNotNone(parameter.grad, name)
                        self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                model.zero_grad(set_to_none=True)
                constant = torch.ones_like(x).detach().requires_grad_(True)
                constant_output = model(constant)
                self.assertTrue(torch.isfinite(constant_output).all())
                self.assertTrue((constant_output.diff(dim=-1) >= 0).all())
                constant_output.square().mean().backward()
                self.assertTrue(torch.isfinite(constant.grad).all())
                for name, parameter in model.named_parameters():
                    if parameter.requires_grad:
                        self.assertIsNotNone(parameter.grad, name)
                        self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_selection_uses_complete_validation_groups_only(self):
        records = [dict(candidate_id=c, seed=s, best_val_pinball=v,
                        QuantileLoss_standardized=test)
                   for c, v, test in [('a', .2, 0.), ('b', .1, 99.)]
                   for s in [2024, 2025, 2026]]
        records.append(dict(candidate_id='incomplete', seed=2024, best_val_pinball=0.))
        self.assertEqual(select_by_validation(records, [2024, 2025, 2026])['candidate_id'], 'b')
        changed = copy.deepcopy(records)
        for r in changed:
            r['QuantileLoss_standardized'] = -123
        self.assertEqual(select_by_validation(changed, [2024, 2025, 2026])['candidate_id'], 'b')
        with self.assertRaises(ValueError):
            select_by_validation(records[-1:], [2024, 2025, 2026])

    def test_invalid_or_duplicate_validation_values_are_rejected(self):
        with self.assertRaises(ValueError):
            select_by_validation([dict(candidate_id='x', seed=2024, best_val_pinball=float('nan'))], [2024])
        record = dict(candidate_id='x', seed=2024, best_val_pinball=.1)
        with self.assertRaises(ValueError):
            select_by_validation([record, record], [2024])

    def test_cache_rejects_changed_configuration(self):
        from scripts.run_architecture_study import check_fingerprint
        with tempfile.TemporaryDirectory() as temporary:
            signature = dict(config=dict(epochs=8), seed=2024, code_sha256='old')
            marker = check_fingerprint(temporary, signature)
            (Path(temporary) / 'provenance.json').write_text(json.dumps(marker))
            self.assertEqual(check_fingerprint(temporary, signature), marker)
            with self.assertRaises(RuntimeError):
                check_fingerprint(temporary, dict(signature, code_sha256='new'))
            with self.assertRaises(RuntimeError):
                check_fingerprint(temporary, dict(signature, config=dict(epochs=30)))
            (Path(temporary) / 'provenance.json').unlink()
            (Path(temporary) / 'validation_complete.json').write_text('{}')
            with self.assertRaises(RuntimeError):
                check_fingerprint(temporary, signature)

    def test_matrix_has_all_missing_horizons_and_deduplicates_controls(self):
        from scripts.run_architecture_study import evaluation_queue
        jobs = evaluation_queue('linear')
        keys = {(j['dataset'], j['horizon'], j['candidate_id'], j['seed']) for j in jobs}
        self.assertEqual(len(keys), len(jobs))
        for horizon in [96, 192, 336, 720]:
            for dataset in ['ETTh1', 'ETTh2']:
                for seed in [2024, 2025, 2026]:
                    self.assertIn((dataset, horizon, 'linear', seed), keys)
        for dataset in ['ETTm1', 'ETTm2', 'Weather', 'Electricity', 'Traffic']:
            self.assertIn((dataset, 96, 'transformer3', 2024), keys)

    def test_dilated_convolution_matches_reference_values_and_gradients(self):
        from experiments.architecture_search import LinearConv1d
        for kernel, dilation in [(3, 1), (3, 2), (3, 4), (1, 1)]:
            reference = torch.nn.Conv1d(3, 5, kernel, padding=(kernel // 2) * dilation,
                                        dilation=dilation).double()
            actual = LinearConv1d(3, 5, kernel, padding=(kernel // 2) * dilation,
                                   dilation=dilation).double()
            actual.load_state_dict(reference.state_dict())
            x1 = torch.randn(2, 3, 11, dtype=torch.float64, requires_grad=True)
            x2 = x1.detach().clone().requires_grad_(True)
            y1, y2 = reference(x1), actual(x2)
            torch.testing.assert_close(y1, y2, atol=1e-12, rtol=1e-12)
            y1.square().sum().backward()
            y2.square().sum().backward()
            torch.testing.assert_close(x1.grad, x2.grad, atol=1e-12, rtol=1e-12)
            torch.testing.assert_close(reference.weight.grad, actual.weight.grad, atol=1e-12, rtol=1e-12)
            torch.testing.assert_close(reference.bias.grad, actual.bias.grad, atol=1e-12, rtol=1e-12)

    def test_study_freezes_configuration_and_dataset_bytes(self):
        from scripts.run_architecture_study import freeze_inputs, verify_inputs
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            data = directory / 'data.csv'
            data.write_text('date,x\n1,2\n')
            config = directory / 'config.yaml'
            config.write_text('epochs: 30\n')
            signature = freeze_inputs([data, config])
            verify_inputs(signature)
            config.write_text('epochs: 8\n')
            with self.assertRaises(RuntimeError):
                verify_inputs(signature)
            config.write_text('epochs: 30\n')
            data.write_text('date,x\n1,999\n')
            with self.assertRaises(RuntimeError):
                verify_inputs(signature)

    def test_export_rejects_record_identity_not_bound_to_provenance(self):
        from scripts.export_architecture_study import verify
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            (path / 'training_history.csv').write_text('epoch,val_pinball\n1,0.1\n')
            from scripts.run_architecture_study import check_fingerprint
            marker = check_fingerprint(path, dict(candidate=dict(id='real'), dataset='ETTh1',
                                                   horizon=96, seed=2024, config={},
                                                   code_sha256='c', data_sha256='d'))
            (path / 'provenance.json').write_text(json.dumps(marker))
            forged = dict(source_dir=str(path), best_epoch=1, best_val_pinball=.1,
                          candidate_id='forged', candidate=dict(id='forged'), dataset='ETTh1',
                          horizon=96, seed=2024, code_sha256='c', data_sha256='d')
            with self.assertRaises(ValueError):
                verify([forged])

    def test_confirmation_curves_render_with_installed_pandas_and_matplotlib(self):
        from scripts import export_architecture_study as exporter
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / 'training_history.csv').write_text('epoch,val_pinball\n1,0.2\n2,0.1\n')
            record = dict(candidate_id='mlp3', seed=2024, best_val_pinball=.1, source_dir=temporary)
            with patch.object(exporter, 'FIG', directory):
                exporter.plots([], [record], None, [])
            self.assertGreater((directory / 'architecture_learning_curves.pdf').stat().st_size, 1000)


if __name__ == '__main__':
    unittest.main()
