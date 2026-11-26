import csv
import math
from tqdm.auto import tqdm
import time
from pathlib import Path
import torch

from utils.losses import forecasting_loss
from utils.channel_policy import resolve_policy, channel_slices
from engine.checkpoint import save_checkpoint


def _channel_indices(n_channels, sample_size, device):
    if sample_size is None or n_channels <= int(sample_size):
        return torch.arange(n_channels, device=device)
    return torch.randperm(n_channels, device=device)[:int(sample_size)]


@torch.no_grad()
def validation_pinball_loss(
    model, loader, device, channel_chunk_size, quantiles
):
    model.eval()
    total_loss = 0.0
    total_elements = 0

    for x, y in tqdm(loader, desc="Validation", dynamic_ncols=True, leave=False, mininterval=2):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        channels = x.shape[-1]

        for chunk in channel_slices(model, channels, channel_chunk_size):
            q = model(x[:, :, chunk])
            target = y[:, :, chunk]
            loss = forecasting_loss(q, target, quantiles)
            elements = target.numel() * (len(quantiles) if q.ndim == 4 else 1)
            total_loss += float(loss.item()) * elements
            total_elements += elements

    if not total_elements:
        raise ValueError("Validation loader is empty")
    return total_loss / total_elements


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        device,
        config,
        run_dir,
        logger,
        train_objective=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config
        self.run_dir = Path(run_dir)
        self.logger = logger
        self.train_objective = train_objective or forecasting_loss

        train_cfg = config["training"]
        policy = resolve_policy(model, config)
        self.channel_sample_size = policy["train_sample_size"]
        self.gradient_clip = float(train_cfg["gradient_clip"])
        self.epochs = int(train_cfg["epochs"])
        self.patience = int(train_cfg["early_stopping_patience"])
        self.quantiles = tuple(config["model"]["quantiles"])
        self.loss_name = "mse" if getattr(model,"output_mode","quantile")=="point" else "pinball"
        self.val_chunk_size = policy["evaluation_chunk_size"]
        self.amp_enabled = bool(train_cfg.get("amp", False)) and device.type == "cuda"

        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled)

    def fit(self, train_loader, val_loader):
        best_val = math.inf
        bad_epochs = 0
        best_path = self.run_dir / "best_checkpoint.pt"
        start_time = time.time()

        history = []
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            running_loss = 0.0
            running_elements = 0

            progress = tqdm(train_loader, desc=f"Train {epoch}/{self.epochs}", dynamic_ncols=True, mininterval=2)
            for x, y in progress:
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                indices = _channel_indices(
                    x.shape[-1], self.channel_sample_size, self.device
                )
                x = x.index_select(dim=2, index=indices)
                y = y.index_select(dim=2, index=indices)

                self.optimizer.zero_grad(set_to_none=True)

                if self.amp_enabled:
                    with torch.autocast(
                        device_type="cuda", dtype=torch.float16
                    ):
                        prediction = self.model(x)
                        loss = self.train_objective(
                            prediction, y, self.quantiles
                        )
                    if not torch.isfinite(loss):
                        raise RuntimeError("Non-finite training loss")
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.gradient_clip
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    prediction = self.model(x)
                    loss = self.train_objective(
                        prediction, y, self.quantiles
                    )
                    if not torch.isfinite(loss):
                        raise RuntimeError("Non-finite training loss")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.gradient_clip
                    )
                    self.optimizer.step()

                elements = y.numel() * (len(self.quantiles) if prediction.ndim == 4 else 1)
                running_loss += float(loss.detach().item()) * elements
                running_elements += elements
                progress.set_postfix({self.loss_name: f"{running_loss / running_elements:.5f}"}, refresh=False)

            if not running_elements:
                raise ValueError("Training loader is empty")
            train_loss = running_loss / running_elements
            val_loss = validation_pinball_loss(
                self.model,
                val_loader,
                self.device,
                self.val_chunk_size,
                self.quantiles,
            )

            if not math.isfinite(train_loss) or not math.isfinite(val_loss):
                raise RuntimeError("Non-finite loss; refusing to select a checkpoint.")
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]["lr"]

            self.logger.info(
                f"Epoch {epoch:03d}/{self.epochs} | "
                f"train_{self.loss_name}={train_loss:.6f} | "
                f"val_{self.loss_name}={val_loss:.6f} | lr={current_lr:.2e}"
            )

            if val_loss < best_val - 1e-8:
                best_val = val_loss
                bad_epochs = 0
                save_checkpoint(
                    best_path,
                    self.model,
                    self.optimizer,
                    epoch,
                    best_val,
                    self.config,
                    scheduler=self.scheduler, amp_scaler=self.scaler,
                )
                self.logger.info(f"Saved best checkpoint: {best_path}")
            else:
                bad_epochs += 1

            history.append(dict(epoch=epoch, train_pinball=train_loss,
                                val_pinball=val_loss, learning_rate=current_lr,
                                best_val=best_val, elapsed_seconds=time.time()-start_time))
            if getattr(self.model,"output_mode","quantile") == "point":
                history[-1]["train_mse"] = history[-1].pop("train_pinball")
                history[-1]["val_mse"] = history[-1].pop("val_pinball")
            with open(self.run_dir / "training_history.csv", "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(history[0]))
                writer.writeheader()
                writer.writerows(history)
            if bad_epochs >= self.patience:
                self.logger.info(
                    f"Early stopping at epoch {epoch}; "
                    f"best validation loss={best_val:.6f}"
                )
                break

        elapsed = time.time() - start_time
        self.logger.info(
            f"Training finished in {elapsed:.1f}s; best val={best_val:.6f}"
        )
        if not best_path.exists():
            raise RuntimeError("Training ended without producing a checkpoint.")
        return best_path, best_val, elapsed
