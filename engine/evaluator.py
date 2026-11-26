import json
from tqdm.auto import tqdm
from pathlib import Path
import numpy as np
import torch

from utils.metrics import IntervalMetricAccumulator
from utils.channel_policy import channel_slices


@torch.no_grad()
def evaluate_test(
    model,
    loader,
    device,
    scaler,
    channel_chunk_size,
    quantiles,
    output_dir,
    save_horizon_metrics=True,
    intervention=None,
):
    model.eval()
    accumulator = IntervalMetricAccumulator(scaler)

    point = getattr(model, "output_mode", "quantile") == "point"
    point_squared, point_absolute, count = 0.0, 0.0, 0

    for batch_index, (x, y) in enumerate(tqdm(loader, desc=f"Test {intervention or 'full'}", dynamic_ncols=True, mininterval=2)):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        n_channels = x.shape[-1]

        for chunk in channel_slices(model, n_channels, channel_chunk_size):
            channel_ids = np.arange(chunk.start, chunk.stop)
            q = model(x[:, :, chunk], intervention=intervention)
            target = y[:, :, chunk]
            if not torch.isfinite(q).all():
                raise RuntimeError("Non-finite test predictions")
            if point:
                error = (q.double()-target.double()).cpu().numpy()
                point_squared += float((error**2).sum())
                point_absolute += float(np.abs(error).sum())
                count += error.size
            else:
                accumulator.update(q.detach().float().cpu().numpy(),
                                   target.detach().float().cpu().numpy(), channel_ids, quantiles)

    if point:
        if not count:
            raise ValueError("Test loader is empty")
        summary = dict(MSE_standardized=point_squared/count, MAE_standardized=point_absolute/count)
    else:
        if not accumulator.pinball_count:
            raise ValueError("Test loader is empty")
        summary = accumulator.summary()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    if save_horizon_metrics and not point:
        accumulator.horizon_frame().to_csv(
            output_dir / "horizon_metrics.csv", index=False
        )

    return summary
