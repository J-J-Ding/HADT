import json
import logging
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import pandas as pd
import torch
from engine.trainer import Trainer
from engine.checkpoint import load_checkpoint


class TestTrainingArtifacts(unittest.TestCase):
    def test_checkpoint_tracks_validation_best_and_scaler(self):
        # Changing checkpoint selection to last epoch must fail this test.
        model = torch.nn.Linear(2, 7)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        config = dict(training=dict(channel_sample_size=2, gradient_clip=1, epochs=3,
                                   early_stopping_patience=3),
                      model=dict(quantiles=[.05,.1,.25,.5,.75,.9,.95]),
                      evaluation=dict(channel_chunk_size=2),
                      data_scaler=dict(mean=[10,20], scale=[2,3]), feature_columns=['a','b'])
        class TinyModel(torch.nn.Module):
            def __init__(self, linear):
                super().__init__()
                self.linear = linear
            def forward(self, x):
                return self.linear(x).unsqueeze(2).expand(-1, -1, 2, -1)
        wrapped = TinyModel(model)
        with tempfile.TemporaryDirectory() as tmp:
            trainer = Trainer(wrapped, optimizer, scheduler, torch.device('cpu'), config, tmp,
                              logging.getLogger('artifact_test'))
            with patch('engine.trainer.validation_pinball_loss', side_effect=[.2,.1,.3]):
                path, val, _ = trainer.fit([(torch.randn(2,3,2),torch.randn(2,3,2))], [])
            ckpt = load_checkpoint(path, wrapped)
            self.assertEqual(ckpt['epoch'], 2)
            self.assertEqual(val, .1)
            self.assertEqual(ckpt['scaler'], config['data_scaler'])
            self.assertEqual(len(pd.read_csv(Path(tmp)/'training_history.csv')), 3)
            self.assertIn('scheduler_state', ckpt)
