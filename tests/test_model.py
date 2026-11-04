import unittest
import torch

from models.hadt_nq import HADTNQ
from losses.quantile_loss import multi_quantile_pinball_loss


class TestHADTNQ(unittest.TestCase):
    def _model(self, pred_len=96):
        return HADTNQ(
            seq_len=96,
            pred_len=pred_len,
            patch_len=16,
            stride=8,
            d_model=64,
            n_heads=4,
            e_layers=2,
            d_ff=128,
            dropout=0.0,
            n_points="auto",
            max_horizon=720,
            horizon_chunk_size=32,
            gradient_checkpointing=True,
        )

    def test_shape_and_monotonicity(self):
        model = self._model(96).eval()
        x = torch.randn(2, 96, 7)
        q = model(x)
        self.assertEqual(tuple(q.shape), (2, 96, 7, 7))
        self.assertTrue(torch.all(q[..., 1:] > q[..., :-1]))

    def test_offset_gradient(self):
        model = self._model(96).train()
        x = torch.randn(2, 96, 7)
        y = torch.randn(2, 96, 7)
        q = model(x)
        loss = multi_quantile_pinball_loss(q, y)
        loss.backward()
        grad = model.decoder.offset_projection.weight.grad
        self.assertIsNotNone(grad)
        self.assertGreater(float(grad.abs().sum()), 0.0)

    def test_horizon_720(self):
        model = self._model(720).eval()
        x = torch.randn(1, 96, 3)
        with torch.no_grad():
            q = model(x)
        self.assertEqual(tuple(q.shape), (1, 720, 3, 7))


if __name__ == "__main__":
    unittest.main()
