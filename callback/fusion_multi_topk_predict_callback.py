"""Prediction writer for isolated multi-TopK SAIL inference."""

import os
from typing import Sequence

import torch
from lightning import LightningModule, Trainer
from osgeo import gdal

from sampler.normal_predict_sampler import normal_stitch
from utils.raster_io import fastly_save_image
from .fusion_predict_callback import FusionWriter


class MultiTopKFusionWriter(FusionWriter):
    """Stitch one full probability/binary raster for every requested K."""

    def __init__(self, *args, topk_values, **kwargs):
        super().__init__(*args, **kwargs)
        self.topk_values = tuple(sorted({int(value) for value in topk_values}))
        self.topk_outputs = {
            value: self._make_dir(
                os.path.join(self.output_dir, f"top{value}", "xyz"),
                {
                    "output_predict_dir": "xyz_predict",
                    "output_prob_dir": "xyz_prob",
                },
            )
            for value in self.topk_values
        }

    def write_on_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        predictions: Sequence,
        batch_indices: Sequence,
    ) -> None:
        files = [
            name for name in os.listdir(self.ref_xyz_dir)
            if os.path.isfile(os.path.join(self.ref_xyz_dir, name))
        ]
        assert len(files) == 1, (
            f"ref_xyz_dir must contain exactly one file, found {len(files)}"
        )
        ref_filename = files[0]
        ref_file = os.path.join(self.ref_xyz_dir, ref_filename)
        offsets = [item["offset"] for item in predictions]
        patch_sizes = [item["patch_size"] for item in predictions]
        height, width = patch_sizes[0]
        full_width = max(x + width for x, _ in offsets)
        full_height = max(y + height for _, y in offsets)

        for topk in self.topk_values:
            key = str(topk)
            probabilities = [
                item["pred_by_topk"][key] for item in predictions
            ]
            out_class = probabilities[0].shape[1]
            stitched = normal_stitch(
                probabilities,
                offsets,
                full_width,
                full_height,
                out_class,
            )
            output = self.topk_outputs[topk]
            if self.save_prob:
                fastly_save_image(
                    stitched.squeeze().detach().cpu().numpy(),
                    os.path.join(output["output_prob_dir"], ref_filename),
                    ref_file,
                    gdal_type=gdal.GDT_Float32,
                )
            binary = (stitched.squeeze() > self.patch_threshold).to(torch.uint8)
            fastly_save_image(
                binary.detach().cpu().numpy(),
                os.path.join(output["output_predict_dir"], ref_filename),
                ref_file,
                gdal_type=gdal.GDT_Byte,
            )
