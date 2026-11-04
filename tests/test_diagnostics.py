import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from models.hadt_nq import HADTNQ


class TestDiagnostics(unittest.TestCase):
    def test_attention_and_interventions(self):
        torch.manual_seed(2)
        model = HADTNQ(seq_len=32, pred_len=8, patch_len=8, stride=4,
                       d_model=16, n_heads=2, e_layers=1, d_ff=32,
                       dropout=0, max_horizon=8, horizon_chunk_size=3).eval()
        x = torch.randn(2, 32, 3)
        with torch.no_grad():
            model.decoder.offset_projection.bias.fill_(0.5)
            baseline, aux = model(x, return_aux=True)
            self.assertIn('attention', aux)
            self.assertEqual(tuple(aux['attention'].shape), (2, 3, 8, 2, model.n_points))
            torch.testing.assert_close(aux['attention'].sum(-1), torch.ones(2, 3, 8, 2))
            for name in ('no_offsets', 'no_encoder', 'no_revin'):
                output = model(x, intervention=name)
                self.assertEqual(output.shape, baseline.shape)
                self.assertTrue(torch.isfinite(output).all())
                self.assertTrue((output[..., 1:] >= output[..., :-1]).all())
                self.assertFalse(torch.allclose(output, baseline))
            torch.testing.assert_close(model(x), baseline)

    def test_artifacts_original_scale(self):
        from engine.visualization import save_forecast_figures
        from data_provider.scaler import StandardScaler
        scaler = StandardScaler().fit(np.array([[10., 100.], [14., 108.]]))
        x = np.zeros((3, 8, 2))
        y = np.zeros((3, 4, 2))
        q = np.broadcast_to(np.arange(-3, 4), (3, 4, 2, 7)).copy()
        with tempfile.TemporaryDirectory() as tmp:
            save_forecast_figures(x, y, q, scaler, ['a', 'b'], [0, 5, 10], tmp)
            saved = np.load(Path(tmp) / 'forecast_samples.npz')
            np.testing.assert_allclose(saved['target'][0, 0], [12., 104.])
            self.assertEqual(len(list(Path(tmp).glob('forecast_*.png'))), 2)

    def test_complete_run_figures(self):
        from types import SimpleNamespace
        import pandas as pd
        from engine.visualization import create_run_figures
        from data_provider.scaler import StandardScaler
        from torch.utils.data import DataLoader, TensorDataset
        model = HADTNQ(seq_len=32, pred_len=8, patch_len=8, stride=4,
                       d_model=16, n_heads=2, e_layers=1, d_ff=32,
                       dropout=0, max_horizon=8).eval()
        ds = TensorDataset(torch.randn(3,32,1), torch.randn(3,8,1))
        bundle = SimpleNamespace(test_loader=DataLoader(ds),
                                 scaler=StandardScaler().fit(np.array([[1.],[3.]])),
                                 feature_columns=['OT'])
        with tempfile.TemporaryDirectory() as tmp:
            pd.DataFrame(dict(epoch=[1,2],train_pinball=[.3,.2],val_pinball=[.2,.25])).to_csv(
                Path(tmp)/'training_history.csv', index=False)
            pd.DataFrame(dict(horizon_step=range(1,9), PICP50=[.5]*8,
                              PICP80=[.8]*8, PICP90=[.9]*8)).to_csv(Path(tmp)/'horizon_metrics.csv',index=False)
            create_run_figures(model, bundle, torch.device('cpu'), tmp,
                               dict(PICP50=.5, PICP80=.8, PICP90=.9))
            for name in ['calibration','training_history','intermediate_steps']:
                self.assertTrue((Path(tmp)/'figures'/f'{name}.png').exists())

    def test_high_dimensional_figures_use_fixed_channels_and_correct_scaler(self):
        from types import SimpleNamespace
        import pandas as pd
        from engine.visualization import create_run_figures
        from data_provider.scaler import StandardScaler
        from torch.utils.data import DataLoader, TensorDataset
        model = HADTNQ(seq_len=32, pred_len=8, patch_len=8, stride=4,
                       d_model=16, n_heads=2, e_layers=1, d_ff=32,
                       dropout=0, max_horizon=8).eval()
        scaler = StandardScaler().fit(np.stack([np.arange(10),np.arange(10)+2]))
        ds = TensorDataset(torch.zeros(3,32,10),torch.zeros(3,8,10))
        bundle = SimpleNamespace(test_loader=DataLoader(ds),scaler=scaler,
                                 feature_columns=[str(i) for i in range(10)])
        original_mean = scaler.mean_.copy()
        with tempfile.TemporaryDirectory() as tmp:
            pd.DataFrame(dict(epoch=[1],train_pinball=[.3],val_pinball=[.2])).to_csv(
                Path(tmp)/'training_history.csv', index=False)
            create_run_figures(model,bundle,torch.device('cpu'),tmp,
                               dict(PICP50=.5,PICP80=.8,PICP90=.9),max_channels=3)
            saved=np.load(Path(tmp)/'figures/forecast_samples.npz')
            np.testing.assert_array_equal(saved['channel_indices'],[0,4,9])
            np.testing.assert_allclose(saved['target'][0,0],[1,5,10])
            self.assertEqual(saved['quantiles'].shape,(3,8,3,7))
            self.assertEqual(len(list((Path(tmp)/'figures').glob('forecast_*.png'))),3)
            np.testing.assert_array_equal(scaler.mean_,original_mean)


if __name__ == '__main__':
    unittest.main()
