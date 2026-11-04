import unittest


class TestPaperEvidence(unittest.TestCase):
    def record(self,seed,value):
        return dict(dataset='ETTh1',method='hadt',horizon=96,seed=seed,
                    QuantileLoss_standardized=value,PICP90=.9,PINAW90=.2,
                    NormalizedWinkler90=.3,CrossingRate=0.,best_val_pinball=value)

    def test_complete_seed_groups_only_and_sample_standard_deviation(self):
        from scripts.export_paper_evidence import aggregate_complete_groups
        records=[self.record(2024,1.),self.record(2025,2.)]
        self.assertEqual(len(aggregate_complete_groups(records)),0)
        frame=aggregate_complete_groups(records+[self.record(2026,3.)])
        self.assertEqual(frame.iloc[0]['n'],3)
        self.assertAlmostEqual(frame.iloc[0]['QuantileLoss_standardized_mean'],2.)
        self.assertAlmostEqual(frame.iloc[0]['QuantileLoss_standardized_std'],1.)

    def test_conflicting_duplicate_measurements_are_rejected(self):
        from scripts.export_paper_evidence import unique_records
        with self.assertRaisesRegex(ValueError,'Conflicting'):
            unique_records([self.record(2024,1.),self.record(2024,2.)])

    def test_unconstrained_head_interval_scores_are_not_presented_as_proper(self):
        from scripts.export_paper_evidence import aggregate_complete_groups
        records=[dict(self.record(seed,1.),method='independent',CrossingRate=.01)
                 for seed in [2024,2025,2026]]
        result=aggregate_complete_groups(records)
        self.assertNotIn('NormalizedWinkler90_mean',result.columns)
        self.assertIn('CrossingRate_mean',result.columns)

    def test_missing_historical_outputs_fall_back_to_fresh_training(self):
        import os,tempfile
        from scripts.run_paper_experiments import import_pilot
        original=os.getcwd()
        with tempfile.TemporaryDirectory() as tmp:
            try:
                os.chdir(tmp)
                self.assertIsNone(import_pilot('ETTh1','hadt',96,2024,{},'unused'))
            finally:os.chdir(original)
