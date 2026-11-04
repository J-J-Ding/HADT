from copy import deepcopy
from pathlib import Path
import yaml


EXPECTED_QUANTILES = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    validate_config(config)
    return config


def clone_config(config):
    return deepcopy(config)


def validate_config(config):
    data = config["data"]
    model = config["model"]
    training = config["training"]
    experiment = config["experiment"]

    l = int(data["seq_len"])
    p = int(model["patch_len"])
    s = int(model["stride"])
    d = int(model["d_model"])
    heads = int(model["n_heads"])

    if p <= 0 or s <= 0 or p > l:
        raise ValueError("Require 0 < patch_len <= seq_len and stride > 0.")
    if d % heads != 0:
        raise ValueError("d_model must be divisible by n_heads.")
    if model["quantiles"] != EXPECTED_QUANTILES:
        raise ValueError(f"Quantiles must equal {EXPECTED_QUANTILES}.")
    if int(model["max_horizon"]) < max(experiment["horizons"]):
        raise ValueError("max_horizon must cover all configured horizons.")
    if training["batch_size"] <= 0:
        raise ValueError("batch_size must be positive.")

    ratios_cfg = data["split_ratios"]
    for group in ("ETT", "other"):
        ratios = ratios_cfg[group]
        if len(ratios) != 3 or abs(sum(ratios) - 1.0) > 1e-8:
            raise ValueError(f"split_ratios[{group}] must contain three values summing to 1.")
