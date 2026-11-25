import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lightning.pytorch.callbacks import BasePredictionWriter
from lightning.pytorch.utilities import rank_zero_only
import torch
from osgeo import gdal
import csv
from utils.raster_io import fastly_save_image
class AbstractWriter(BasePredictionWriter):

    def __init__(self, output_dir, ref_dir, num_class, write_interval = 'batch_and_epoch'):
        super().__init__(write_interval)
        self.output_dir = output_dir
        self.ref_dir = ref_dir
        self.num_class = num_class
        self.skip_create_stat = False

    @rank_zero_only
    def _make_dir(self, output_dir, out_name_dict):
        out_dir = {}
        os.makedirs(output_dir, exist_ok=True)  # exist_ok=True 表示如果目录已存在，不会报错

        for key ,value in out_name_dict.items():
            path = os.path.join(output_dir, value)
            out_dir[key] = path
            if not os.path.exists(path):
                out_dir[key] = path
                os.makedirs(path, exist_ok=True)
        return out_dir

    def _write2disk(self, data : torch.Tensor, metrics, out_file, ref_file, stat_file, file_name, gdal_type = gdal.GDT_Byte):
        if data.ndim == 4:
            data = data.squeeze(0)
        fastly_save_image(data.detach().cpu().numpy(), out_file, ref_file, gdal_type = gdal_type)

        with open(stat_file, 'a') as f:
            writer = csv.writer(f)
            if not self.skip_create_stat:
                title_name = [' image name',] + list(metrics.keys())
                writer.writerow(title_name)
                self.skip_create_stat = True

            for key, value in metrics.items():
                if isinstance(value, torch.Tensor):
                    value = value.detach().cpu().item()
                    metrics[key] = value
            writer.writerow([file_name,] + list(metrics.values()))

    def _write_stat_for_end(self, metrics, out_file):
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().item()
                metrics[key] = value
        with open(out_file, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(metrics.keys())
            writer.writerow(metrics.values())
