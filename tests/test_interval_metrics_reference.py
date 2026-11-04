"""Hand-calculated scoring example independent of any fitted model."""
import unittest
import numpy as np
from data_provider.scaler import StandardScaler
from metrics.interval_metrics import IntervalMetricAccumulator


class TestIntervalMetricReference(unittest.TestCase):
    def test_two_observation_reference_scores(self):
        scaler=StandardScaler().fit(np.array([[-1.],[1.]]))
        q=np.broadcast_to(np.arange(-3,4,dtype=float),(2,1,1,7)).copy()
        target=np.array([0.,4.]).reshape(2,1,1)
        acc=IntervalMetricAccumulator(scaler)
        acc.update(q,target,np.array([0]),[.05,.1,.25,.5,.75,.9,.95])
        result=acc.summary()
        # Quantile-loss sums are 1.2 and 9.2 over seven levels.
        self.assertAlmostEqual(result['QuantileLoss_standardized'],10.4/14)
        self.assertEqual(result['PICP50'],.5)
        self.assertEqual(result['PICP80'],.5)
        self.assertEqual(result['PICP90'],.5)
        self.assertEqual(result['MPIW90'],6.)
        self.assertEqual(result['PINAW90'],3.)
        # 90% interval scores: width6 and width6+20*miss1, averaged=16.
        self.assertAlmostEqual(result['Winkler90'],16.)
        self.assertAlmostEqual(result['NormalizedWinkler90'],8.)
        self.assertEqual(result['CrossingRate'],0.)
        self.assertEqual(acc.horizon_frame().iloc[0]['PICP90'],.5)
