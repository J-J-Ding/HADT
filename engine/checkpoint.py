from pathlib import Path
import torch


def save_checkpoint(path, model, optimizer, epoch, best_val_loss, config, scheduler=None, amp_scaler=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": int(epoch),
            "best_val_loss": float(best_val_loss),
            "config": config,
            "scheduler_state": scheduler.state_dict() if scheduler else None,
            "amp_scaler_state": amp_scaler.state_dict() if amp_scaler else None,
            "scaler": config.get("data_scaler"),
            "feature_columns": config.get("feature_columns"),
        },
        path,
    )


def load_checkpoint(path, model, map_location="cpu"):
    checkpoint = torch.load(
        path, map_location=map_location, weights_only=False
    )
    model.load_state_dict(checkpoint["model_state"])
    return checkpoint
