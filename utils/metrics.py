import numpy as np
import pandas as pd


class IntervalMetricAccumulator:
    """Accumulates pooled marginal interval metrics and horizon-wise PICP."""

    INTERVALS = {
        "50": (2, 4, 0.50),
        "80": (1, 5, 0.80),
        "90": (0, 6, 0.90),
    }

    def __init__(self, scaler):
        self.scaler = scaler
        self.total_elements = 0
        self.pinball_sum = 0.0
        self.pinball_count = 0
        self.crossing_count = 0
        self.crossing_elements = 0
        self.stats = {}
        self.horizon_covered = {}
        self.horizon_count = None

        for name in self.INTERVALS:
            self.stats[name] = {
                "covered": 0,
                "width": 0.0,
                "normalized_width": 0.0,
                "winkler": 0.0,
                "normalized_winkler": 0.0,
                "count": 0,
            }
            self.horizon_covered[name] = None

    def update(self, pred_std, target_std, channel_indices, quantiles):
        """
        pred_std: [B,H,C_chunk,7], model output in global standardized scale
        target_std: [B,H,C_chunk]
        """
        pred_std = np.asarray(pred_std, dtype=np.float64)
        target_std = np.asarray(target_std, dtype=np.float64)
        channel_indices = np.asarray(channel_indices, dtype=np.int64)

        mean = self.scaler.mean_[channel_indices][None, None, :, None]
        scale = self.scaler.scale_[channel_indices][None, None, :, None]
        ranges = self.scaler.data_range_[channel_indices][None, None, :]

        pred_raw = pred_std * scale + mean
        target_raw = target_std * scale[..., 0] + mean[..., 0]

        # Quantile loss is computed in standardized units to avoid
        # domination by variables with larger physical units.
        tau = np.asarray(quantiles, dtype=np.float64).reshape(1, 1, 1, -1)
        error = target_std[..., None] - pred_std
        pinball = np.maximum(tau * error, (tau - 1.0) * error)
        self.pinball_sum += float(pinball.sum())
        self.pinball_count += int(pinball.size)

        crossing = np.any(pred_raw[..., 1:] < pred_raw[..., :-1], axis=-1)
        self.crossing_count += int(crossing.sum())
        self.crossing_elements += int(crossing.size)

        batch_size, horizon, channels = target_raw.shape
        if self.horizon_count is None:
            self.horizon_count = np.zeros(horizon, dtype=np.int64)
            for name in self.INTERVALS:
                self.horizon_covered[name] = np.zeros(horizon, dtype=np.int64)
        self.horizon_count += batch_size * channels

        for name, (lower_idx, upper_idx, nominal) in self.INTERVALS.items():
            lower = pred_raw[..., lower_idx]
            upper = pred_raw[..., upper_idx]
            covered = (target_raw >= lower) & (target_raw <= upper)
            width = upper - lower
            alpha = 1.0 - nominal

            winkler = (
                width
                + (2.0 / alpha) * (lower - target_raw) * (target_raw < lower)
                + (2.0 / alpha) * (target_raw - upper) * (target_raw > upper)
            )
            normalized_width = width / ranges
            normalized_winkler = winkler / ranges

            stat = self.stats[name]
            stat["covered"] += int(covered.sum())
            stat["width"] += float(width.sum())
            stat["normalized_width"] += float(normalized_width.sum())
            stat["winkler"] += float(winkler.sum())
            stat["normalized_winkler"] += float(normalized_winkler.sum())
            stat["count"] += int(covered.size)

            self.horizon_covered[name] += covered.sum(axis=(0, 2))

    def summary(self):
        result = {
            "QuantileLoss_standardized": (
                self.pinball_sum / max(self.pinball_count, 1)
            ),
            "CrossingRate": (
                self.crossing_count / max(self.crossing_elements, 1)
            ),
        }
        for name, stat in self.stats.items():
            count = max(stat["count"], 1)
            result[f"PICP{name}"] = stat["covered"] / count
            result[f"MPIW{name}"] = stat["width"] / count
            result[f"PINAW{name}"] = stat["normalized_width"] / count
            result[f"Winkler{name}"] = stat["winkler"] / count
            result[f"NormalizedWinkler{name}"] = (
                stat["normalized_winkler"] / count
            )
        return result

    def horizon_frame(self):
        if self.horizon_count is None:
            return pd.DataFrame()
        rows = []
        for h in range(len(self.horizon_count)):
            row = {"horizon_step": h + 1}
            denom = max(int(self.horizon_count[h]), 1)
            for name in self.INTERVALS:
                row[f"PICP{name}"] = (
                    int(self.horizon_covered[name][h]) / denom
                )
            rows.append(row)
        return pd.DataFrame(rows)
