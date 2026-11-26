from pathlib import Path
import torch


def save_checkpoint(path, model, optimizer, epoch, best_val_loss, config, scheduler=None, amp_scaler=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 2,
            "model_name": getattr(model, "model_name", "hadt_nq"),
            "variant": getattr(model, "variant", "quantile"),
            "output_mode": getattr(model, "output_mode", "quantile"),
            "n_vars": getattr(model, "n_vars", None),
            "native_config": getattr(model, "native_config", None),
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
    expected = getattr(model, "model_name", "hadt_nq")
    if "model_name" not in checkpoint:
        if expected != "hadt_nq":
            raise ValueError("Legacy checkpoints may only load into the explicit historical HADT path")
    elif (checkpoint["model_name"] != expected or
          checkpoint["output_mode"] != getattr(model,"output_mode","quantile") or
          checkpoint.get("variant") != getattr(model,"variant","quantile") or
          checkpoint.get("n_vars") != getattr(model,"n_vars",None) or
          checkpoint.get("native_config") != getattr(model,"native_config",None)):
        raise ValueError("Checkpoint model/output/variant/channel metadata mismatch")
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return checkpoint
