import json
from tqdm.auto import tqdm
from pathlib import Path
import numpy as np
import torch

from metrics.interval_metrics import IntervalMetricAccumulator


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

    for batch_index, (x, y) in enumerate(tqdm(loader, desc=f"Test {intervention or 'full'}", dynamic_ncols=True, mininterval=2)):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        n_channels = x.shape[-1]

        for start in range(0, n_channels, channel_chunk_size):
            end = min(start + channel_chunk_size, n_channels)
            channel_ids = np.arange(start, end)

            q = model(x[:, :, start:end], intervention=intervention)
            target = y[:, :, start:end]

            accumulator.update(
                q.detach().float().cpu().numpy(),
                target.detach().float().cpu().numpy(),
                channel_ids,
                quantiles,
            )

    summary = accumulator.summary()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    if save_horizon_metrics:
        accumulator.horizon_frame().to_csv(
            output_dir / "horizon_metrics.csv", index=False
        )

    return summary
