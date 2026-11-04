import torch


DEFAULT_QUANTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)


def multi_quantile_pinball_loss(prediction, target, quantiles=DEFAULT_QUANTILES):
    """
    prediction: [B,H,C,Q]
    target:     [B,H,C]
    """
    if prediction.ndim != 4 or target.ndim != 3:
        raise ValueError(
            f"Expected prediction [B,H,C,Q] and target [B,H,C], "
            f"got {prediction.shape} and {target.shape}."
        )
    if prediction.shape[:3] != target.shape:
        raise ValueError("Prediction and target dimensions do not match.")
    if prediction.shape[-1] != len(quantiles):
        raise ValueError("Quantile dimension does not match quantile list.")

    tau = prediction.new_tensor(quantiles).view(1, 1, 1, -1)
    error = target.unsqueeze(-1) - prediction
    loss = torch.maximum(tau * error, (tau - 1.0) * error)
    return loss.mean()
