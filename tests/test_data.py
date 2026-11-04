import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

from data_provider.csv_dataset import (
    read_multivariate_csv,
    TimeSeriesWindowDataset,
)
from data_provider.scaler import StandardScaler


class TestDataPipeline(unittest.TestCase):
    def test_csv_and_chronological_windows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "toy.csv"
            n = 100
            frame = pd.DataFrame({
                "date": pd.date_range("2020-01-01", periods=n, freq="h"),
                "a": np.arange(n, dtype=float),
                "b": np.arange(n, dtype=float) * 2,
            })
            frame.to_csv(path, index=False)

            values, columns = read_multivariate_csv(path, "date")
            self.assertEqual(values.shape, (100, 2))
            self.assertEqual(columns, ["a", "b"])

            train_end, val_end = 60, 80
            bounds = {
                "train": (0, train_end),
                "val": (train_end, val_end),
                "test": (val_end, n),
            }
            scaler = StandardScaler().fit(values[:train_end])
            scaled = scaler.transform(values)

            train_ds = TimeSeriesWindowDataset(
                scaled, 10, 5, "train", bounds
            )
            val_ds = TimeSeriesWindowDataset(
                scaled, 10, 5, "val", bounds
            )
            test_ds = TimeSeriesWindowDataset(
                scaled, 10, 5, "test", bounds
            )

            self.assertEqual(train_ds.target_starts[0], 10)
            self.assertEqual(train_ds.target_starts[-1] + 5, train_end)
            self.assertEqual(val_ds.target_starts[0], train_end)
            self.assertEqual(val_ds.target_starts[-1] + 5, val_end)
            self.assertEqual(test_ds.target_starts[0], val_end)
            self.assertEqual(test_ds.target_starts[-1] + 5, n)

            x, y = val_ds[0]
            self.assertEqual(tuple(x.shape), (10, 2))
            self.assertEqual(tuple(y.shape), (5, 2))


if __name__ == "__main__":
    unittest.main()
