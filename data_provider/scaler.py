import numpy as np


class StandardScaler:
    """Small NumPy scaler fitted only on the training rows."""

    def __init__(self):
        self.mean_ = None
        self.scale_ = None
        self.data_min_ = None
        self.data_max_ = None
        self.data_range_ = None

    def fit(self, data):
        data = np.asarray(data, dtype=np.float64)
        if data.ndim != 2:
            raise ValueError(f"Scaler expects [T,C], got {data.shape}")
        if not np.isfinite(data).all():
            raise ValueError("Non-finite values found in training data.")
        self.mean_ = data.mean(axis=0)
        std = data.std(axis=0)
        self.scale_ = np.where(std < 1e-8, 1.0, std)
        self.data_min_ = data.min(axis=0)
        self.data_max_ = data.max(axis=0)
        self.data_range_ = self.data_max_ - self.data_min_
        self.data_range_ = np.where(self.data_range_ < 1e-8, 1.0, self.data_range_)
        return self

    def transform(self, data):
        self._check_fitted()
        data = np.asarray(data, dtype=np.float32)
        return ((data - self.mean_.astype(np.float32)) / self.scale_.astype(np.float32)).astype(np.float32)

    def inverse_transform(self, data):
        self._check_fitted()
        return np.asarray(data) * self.scale_ + self.mean_

    def state_dict(self):
        self._check_fitted()
        return {
            "mean": self.mean_.tolist(),
            "scale": self.scale_.tolist(),
            "data_min": self.data_min_.tolist(),
            "data_max": self.data_max_.tolist(),
            "data_range": self.data_range_.tolist(),
        }

    def _check_fitted(self):
        if self.mean_ is None:
            raise RuntimeError("Scaler has not been fitted.")
