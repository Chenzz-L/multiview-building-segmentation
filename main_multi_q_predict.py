#!/conda/env python
"""One-pass fixed-prefix multi-Q SAIL prediction entry point."""

from __future__ import annotations

import argparse
import os
import sys
from types import SimpleNamespace

import torch
from lightning.pytorch.trainer import Trainer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data import build_DataLoader, build_Dataset
from models.fusion_model_interface_dinov3_multi_q import (
    FusionMInterface_DINOv3_MultiQ,
)
from yacs_config.default_config import load_yaml


def load_checkpoint(model, path: str, strict: bool = True) -> None:
    if not path:
        return
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint.get("state_dict", checkpoint)
    missing, unexpected = model.load_state_dict(state, strict=strict)
    print(
        f"[PredictInit] loaded checkpoint: {path} | strict={strict} | "
        f"missing={len(missing)} unexpected={len(unexpected)}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-cfg", "--cfg_name", required=True)
    args = parser.parse_args()
    cfg = load_yaml("./cfgs/" + args.cfg_name + ".yaml")
    model_keys = (
        "model", "callback", "predict", "metric", "feature_save_dir",
        "predict_tdom_only", "predict_tail_q_values",
    )
    model = FusionMInterface_DINOv3_MultiQ(
        **{key: value for key, value in cfg.items() if key in model_keys}
    )
    load_checkpoint(
        model,
        cfg.get("predict_ckpt_path"),
        strict=bool(cfg.get("predict_strict_load", True)),
    )
    dataset = build_Dataset(SimpleNamespace(**cfg["PREDICT_DATASET_CFG"]))
    loader, _ = build_DataLoader(
        dataset, SimpleNamespace(**cfg["PREDICT_DATALOADER_CFG"])
    )
    Trainer(**cfg["trainer"], profiler="simple").predict(model, loader)


if __name__ == "__main__":
    main()
