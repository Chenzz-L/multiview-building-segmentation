import os
import sys
from typing import Any, Optional, Sequence
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from lightning import LightningModule, Trainer
import torch
import tifffile
from osgeo import gdal
from sampler.normal_predict_sampler import normal_sampler,normal_stitch,streaming_stitch_step
from .predict_callback import AbstractWriter
from utils.raster_io import fastly_save_image, OPTIONS
class FusionWriter(AbstractWriter):

    def __init__(
        self,
        output_dir,
        ref_img_dir,
        ref_xyz_dir,
        num_class,
        write_interval,
        save_prob=False,
        patch_only=False,
        patch_threshold=0.5,
    ):
        if patch_only and write_interval == "epoch":
            write_interval = "batch"
        super().__init__(output_dir, [ref_img_dir, ref_xyz_dir], num_class, write_interval)
        self.output_dir = output_dir
        self.ref_img_dir = ref_img_dir
        self.ref_xyz_dir = ref_xyz_dir
        self.num_class = num_class
        self.save_prob = save_prob
        self.patch_only = patch_only
        self.patch_threshold = patch_threshold

        # self.img_output = self._make_dir(os.path.join(output_dir,'img'), {
        #     'output_predict_dir' :'img_predict',
        #     'output_shape_dir_origin' : 'img_shape_origin',
        #     'output_eval_dir' : 'img_evaluation',
        #     'output_stat_dir' : 'img_statistics'})
        self.xyz_output = self._make_dir(os.path.join(output_dir,'xyz'), {
            'output_predict_dir' :'xyz_predict',
            'output_prob_dir' :'xyz_prob',
            'output_patch_predict_dir' :'xyz_patch_predict',
            'output_patch_prob_dir' :'xyz_patch_prob',
            'output_shape_dir_origin' : 'xyz_shape_origin',
            'output_eval_dir' : 'xyz_evaluation',
            'output_stat_dir' : 'xyz_statistics'
        })

    def _write_patch_prediction(self, prediction: Any) -> None:
        prob = prediction["pred"].squeeze().detach().cpu()
        name = prediction.get("xyz_name", None)
        if name is None:
            xoff, yoff = prediction["offset"]
            name = f"patch_{xoff}_{yoff}"
        stem = os.path.splitext(os.path.basename(str(name)))[0]

        if self.save_prob:
            prob_file = os.path.join(
                self.xyz_output['output_patch_prob_dir'],
                f"{stem}.tif",
            )
            tifffile.imwrite(
                prob_file,
                prob.numpy().astype("float32"),
            )

        pred_file = os.path.join(
            self.xyz_output['output_patch_predict_dir'],
            f"{stem}.tif",
        )
        pred = (prob > self.patch_threshold).to(torch.uint8)
        tifffile.imwrite(pred_file, pred.numpy())

    def write_on_batch_end(
        self,
        trainer: "Trainer",
        pl_module: "LightningModule",
        prediction: Any,
        batch_indices: Optional[Sequence[int]],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        if not self.patch_only:
            return
        if prediction is None:
            return
        self._write_patch_prediction(prediction)

    # def write_on_batch_end(
    #     self,
    #     trainer: "pl.Trainer",
    #     pl_module: "pl.LightningModule",
    #     prediction: Any,
    #     batch_indices: Optional[Sequence[int]],
    #     batch: Any,
    #     batch_idx: int,
    #     dataloader_idx: int):
    #     # optionally, you can also save `batch_indices` to get the information about the data index
    #     # from your prediction data
    #     img_predictions, xyz_prediction, img_metrics, xyz_metrics = prediction

    #     # save for image branch
    #     for img_prediction, metrics, file_name in zip(img_predictions, img_metrics, batch['name'][0:-1]):
    #         ref_file = os.path.join(self.ref_img_dir, file_name)
    #         out_file = os.path.join(self.img_output['output_predict_dir'], file_name)
    #         stat_file = os.path.join(self.img_output['output_stat_dir'], os.path.basename(file_name).split('.')[0]+'.csv')
    #         self._write2disk(img_prediction, metrics, out_file, ref_file, stat_file )
    #     # save for xyz branch
    #     file_name = batch['name'][-1]
    #     ref_file = os.path.join(self.ref_xyz_dir, file_name)
    #     out_file = os.path.join(self.xyz_output['output_predict_dir'], file_name)
    #     stat_file = os.path.join(self.xyz_output['output_stat_dir'], os.path.basename(file_name).split('.')[0]+'.csv')
    #     # stat_file = os.path.join(self.xyz_output['output_stat_dir'], 'all_statistics'+'.csv')
    #     self._write2disk(xyz_prediction, xyz_metrics, out_file, ref_file, stat_file)

    # def write_on_epoch_end(
    #     self,
    #     trainer: "pl.Trainer",
    #     pl_module: "pl.LightningModule",
    #     predictions: Sequence[Any],
    #     batch_indices: Sequence[Any],
    #     ) -> None:
    #         img_predict_metrics = pl_module.img_predict_metrics
    #         self._write_stat_for_end(img_predict_metrics,os.path.join(self.img_output['output_stat_dir'], 'global_statistics.csv'))
    #         xyz_predict_metrics = pl_module.xyz_predict_metrics
    #         self._write_stat_for_end(xyz_predict_metrics,os.path.join(self.xyz_output['output_stat_dir'], 'global_statistics.csv'))

    # def _write2disk(self, data : torch.Tensor, metrics, out_file, ref_file, stat_file, gdal_type = gdal.GDT_Byte):
    #     if data.ndim == 4:
    #         data = data.squeeze(0)
    #     fastly_save_image(data.detach().cpu().numpy(), out_file, ref_file, gdal_type = gdal_type)

    #     for key, value in metrics.items():
    #         if isinstance(value, torch.Tensor):
    #             value = value.detach().cpu().item()
    #             metrics[key] = value
    #     with open(stat_file, 'w') as f:
    #         writer=csv.writer(f)
    #         writer.writerow(metrics.keys())
    #         writer.writerow(metrics.values())

    # def _write_stat_for_end(self, metrics, out_file):
    #     for key, value in metrics.items():
    #         if isinstance(value, torch.Tensor):
    #             value = value.detach().cpu().item()
    #             metrics[key] = value
    #     with open(out_file, 'w') as f:
    #         writer=csv.writer(f)
    #         writer.writerow(metrics.keys())
    #         writer.writerow(metrics.values())

    # def write_on_batch_end(
    #     self,
    #     trainer: "pl.Trainer",
    #     pl_module: "pl.LightningModule",
    #     prediction: Any,
    #     batch_indices: Optional[Sequence[int]],
    #     batch: Any,
    #     batch_idx: int,
    #     dataloader_idx: int):
    #     # optionally, you can also save `batch_indices` to get the information about the data index
    #     # from your prediction data
    #     img_predictions, xyz_prediction, img_evals, xyz_eval, img_metrics, xyz_metrics = prediction
    #     stat_file = os.path.join(self.img_output['output_stat_dir'], 'each_statistics'+'.csv')
    #     # save for image branch
    #     for img_prediction, img_eval, metrics, file_name in zip(img_predictions, img_evals,img_metrics, batch['name'][0:-1]):
    #         ref_file = os.path.join(self.ref_img_dir, file_name)
    #         out_file = os.path.join(self.img_output['output_predict_dir'], file_name)
    #         evaluation_file = os.path.join(self.img_output['output_eval_dir'], file_name)
    #         self._write2disk(img_prediction, metrics, out_file, ref_file, stat_file, file_name )
    #         fastly_save_image(image_data = img_eval.detach().cpu().numpy(), out_file = evaluation_file, ref_file = ref_file, options = OPTIONS)
    #     # save for xyz branch
    #     file_name = batch['name'][-1]
    #     ref_file = os.path.join(self.ref_xyz_dir, file_name)
    #     out_file = os.path.join(self.xyz_output['output_predict_dir'], file_name)
    #     stat_file = os.path.join(self.xyz_output['output_stat_dir'], 'each_statistics'+'.csv')
    #     evaluation_file = os.path.join(self.xyz_output['output_eval_dir'], file_name)
    #     self._write2disk(xyz_prediction, xyz_metrics, out_file, ref_file, stat_file, file_name)
    #     fastly_save_image(image_data = xyz_eval.detach().cpu().numpy(), out_file = evaluation_file, ref_file = ref_file, options = OPTIONS)
    # def write_on_epoch_end(
    #     self,
    #     trainer: "pl.Trainer",
    #     pl_module: "pl.LightningModule",
    #     predictions: Sequence[Any],
    #     batch_indices: Sequence[Any],
    #     ) -> None:
    #         img_predict_metrics = pl_module.img_predict_metrics.compute()
    #         self._write_stat_for_end(img_predict_metrics, os.path.join(self.img_output['output_stat_dir'], 'global_statistics.csv'))
    #         xyz_predict_metrics = pl_module.xyz_predict_metrics.compute()
    #         self._write_stat_for_end(xyz_predict_metrics, os.path.join(self.xyz_output['output_stat_dir'], 'global_statistics.csv'))
    def write_on_epoch_end(
        self,
        trainer: "Trainer",
        pl_module: "LightningModule",
        predictions: Sequence[Any],
        batch_indices: Sequence[Any],
    ) -> None:
        if self.patch_only:
            return

        files = [f for f in os.listdir(self.ref_xyz_dir) if os.path.isfile(os.path.join(self.ref_xyz_dir, f))]
        assert len(files) == 1, f"ref_xyz_dir must contain exactly one file, found {len(files)}"
        ref_filename = files[0]

        ref_file = os.path.join(self.ref_xyz_dir, ref_filename)
        out_file = os.path.join(self.xyz_output['output_predict_dir'], ref_filename)
        prob_list = []
        offsets = []
        patch_sizes = []
        
        # 🔹接收 predict_step 返回的内容
        for o in predictions:
            prob_list.append(o["pred"])
            offsets.append(o["offset"])
            patch_sizes.append(o["patch_size"])

        # 假设 patch_size 格式为 (H, W)
        H, W = patch_sizes[0]

        # 🔹根据 offset 计算整体拼接尺寸
        max_x = max([x + W for (x, y) in offsets])
        max_y = max([y + H for (x, y) in offsets])

        full_H, full_W = max_y, max_x  # 得到完整图大小

        out_class = prob_list[0].shape[1]
        
        # 🔹拼接概率图
        stitched_prob = normal_stitch(prob_list, offsets, full_W, full_H, out_class)

        # 🔹得到最终分类图
        if self.save_prob:
            prob_file = os.path.join(
                self.xyz_output['output_prob_dir'],
                ref_filename,
            )
            fastly_save_image(
                stitched_prob.squeeze().detach().cpu().numpy(),
                prob_file,
                ref_file,
                gdal_type=gdal.GDT_Float32,
            )

        class_prediction = (stitched_prob.squeeze() > 0.5).long()  

        fastly_save_image(
            class_prediction.detach().cpu().numpy(),
            out_file,
            ref_file,
            gdal_type=gdal.GDT_Byte,
        )
