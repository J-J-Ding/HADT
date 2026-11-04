from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from data_provider.csv_dataset import (
    TimeSeriesWindowDataset,
    read_multivariate_csv,
)
from data_provider.scaler import StandardScaler


@dataclass
class DataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    scaler: StandardScaler
    feature_columns: list
    n_vars: int
    n_rows: int
    split_bounds: dict


def build_data_bundle(config, dataset_name, pred_len):
    data_cfg = config["data"]
    runtime_cfg = config["runtime"]
    train_cfg = config["training"]
    eval_cfg = config["evaluation"]

    files = data_cfg["files"]
    if dataset_name not in files:
        raise KeyError(f"No CSV mapping configured for dataset '{dataset_name}'.")

    csv_path = Path(data_cfg["root_path"]) / files[dataset_name]
    raw_values, feature_columns = read_multivariate_csv(
        csv_path, time_column=data_cfg["time_column"]
    )

    if data_cfg.get("feature_mode", "M") != "M":
        raise NotImplementedError(
            "This independent project currently implements multivariate mode M only."
        )

    n_rows, n_vars = raw_values.shape
    ratio_group = "ETT" if dataset_name.upper().startswith("ETT") else "other"
    train_ratio, val_ratio, test_ratio = data_cfg["split_ratios"][ratio_group]
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-8:
        raise ValueError("Split ratios must sum to 1.")

    train_end = int(n_rows * train_ratio)
    val_end = train_end + int(n_rows * val_ratio)
    split_bounds = {
        "train": (0, train_end),
        "val": (train_end, val_end),
        "test": (val_end, n_rows),
    }

    scaler = StandardScaler().fit(raw_values[:train_end])
    scaled_values = scaler.transform(raw_values)

    seq_len = int(data_cfg["seq_len"])
    train_ds = TimeSeriesWindowDataset(
        scaled_values, seq_len, pred_len, "train", split_bounds
    )
    val_ds = TimeSeriesWindowDataset(
        scaled_values, seq_len, pred_len, "val", split_bounds
    )
    test_ds = TimeSeriesWindowDataset(
        scaled_values, seq_len, pred_len, "test", split_bounds
    )

    num_workers = int(runtime_cfg.get("num_workers", 0))
    pin_memory = bool(runtime_cfg.get("pin_memory", True)) and torch.cuda.is_available()

    common = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
    }

    train_loader = DataLoader(
        train_ds,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        drop_last=False,
        **common,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(eval_cfg["batch_size"]),
        shuffle=False,
        drop_last=False,
        **common,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=int(eval_cfg["batch_size"]),
        shuffle=False,
        drop_last=False,
        **common,
    )

    return DataBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        scaler=scaler,
        feature_columns=feature_columns,
        n_vars=n_vars,
        n_rows=n_rows,
        split_bounds=split_bounds,
    )
