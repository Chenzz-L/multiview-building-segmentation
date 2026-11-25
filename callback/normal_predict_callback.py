import os
import sys
import torch
import numpy as np
from osgeo import gdal
from typing import Any, Optional, Sequence
import networkx as nx
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from .predict_callback import AbstractWriter
from utils.raster_io import OPTIONS, fastly_save_image

class NormalWriter(AbstractWriter):

    def __init__(self, output_dir, ref_dir, num_class, write_interval):
        super().__init__(output_dir, ref_dir, num_class, write_interval)
        self.output_dir = output_dir
        self.ref_dir = ref_dir
        self.num_class = num_class
        self.output = self._make_dir(output_dir, {
            'output_predict_dir' :'predict',
            # 'output_bimap_dir' :'binary',
            'output_shape_dir_origin' : 'shape_origin',
            # 'output_shape_dir_refined' : 'shape_refined',
            'output_eval_dir' : 'evaluation',
            'output_stat_dir' : 'statistics'})

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
    #     prediction, metrics = prediction
    #     file_name = batch['name'][0]
    #     out_file = os.path.join(self.output['output_predict_dir'], file_name)
    #     ref_file = os.path.join(self.ref_dir, file_name)
    #     if prediction.ndim == 4:
    #         prediction = prediction.squeeze(0)
    #     if isinstance(prediction,torch.Tensor):
    #         prediction = prediction.detach().cpu().numpy()
    #     fastly_save_image(prediction, out_file, ref_file, gdal_type = gdal.GDT_Byte)

    #     for key, value in metrics.items():
    #         if isinstance(value,torch.Tensor):
    #             value = value.detach().cpu().item()
    #             metrics[key] = value

    #     with open(os.path.join(self.output['output_stat_dir'], os.path.basename(file_name).split('.')[0]+'.csv'), 'w') as f:
    #         writer=csv.writer(f)
    #         writer.writerow(metrics.keys())
    #         writer.writerow(metrics.values())
# 语义分割
    def write_on_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        prediction: Any,
        batch_indices: Optional[Sequence[int]],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int):
        # optionally, you can also save `batch_indices` to get the information about the data index
        # from your prediction data
        prediction, eval_image, metrics = prediction
        file_name = batch['name']
        out_file = os.path.join(self.output['output_predict_dir'], file_name)
        ref_file = os.path.join(self.ref_dir, file_name)
        stat_file = os.path.join(self.output['output_stat_dir'], 'each_statistics'+'.csv')

        self._write2disk(prediction, metrics, out_file, ref_file, stat_file, file_name)
        eval_image = eval_image.squeeze()
        evaluation_file = os.path.join(self.output['output_eval_dir'], file_name)
        fastly_save_image(image_data = eval_image.detach().cpu().numpy(), out_file = evaluation_file, ref_file = ref_file, options = OPTIONS)

    def write_on_epoch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        predictions: Sequence[Any],
        batch_indices: Sequence[Any],
        ) -> None:
            predict_metrics = pl_module.predict_metrics.compute()
            self._write_stat_for_end(predict_metrics, os.path.join(self.output['output_stat_dir'], 'global_statistics.csv'))

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
    #     file_name = batch[3][0]
    #     out_file = os.path.join(self.output['output_predict_dir'], file_name)
    #     ref_file = os.path.join(self.ref_dir, file_name)
    #     stat_file = os.path.join(self.output['output_stat_dir'], 'each_statistics'+'.csv')

    #     fastly_save_image(prediction.detach().cpu().numpy(), out_file, ref_file, gdal_type = gdal.GDT_Byte)
    # -----------------------------------------------------------------------------------------------------------------------------------
    

# 实例分割 (极速版)
    # def write_on_batch_end(
    #     self,
    #     trainer,
    #     pl_module,
    #     prediction,
    #     batch_indices,
    #     batch,
    #     batch_idx,
    #     dataloader_idx=0
    # ):
    #     """
    #     prediction: List[Dict]
    #         {
    #             "mask": BoolTensor [h, w],
    #             "score": float,
    #             "label": int,
    #             "offset": (x0, y0)
    #         }
    #     """
    #     image = batch[0]
    #     H, W = image.shape[-2:]
    #     file_name = batch[3][0]

    #     ref_file = os.path.join(self.ref_dir, file_name)
    #     out_file = os.path.join(self.output["output_predict_dir"], file_name)

    #     if len(prediction) == 0:
    #         inst_map = np.zeros((H, W), dtype=np.uint16)
    #     else:
    #         # 1. 预处理：计算每个实例的 bbox 和面积，减少后续重复计算
    #         for inst in prediction:
    #             m = inst["mask"]
    #             y, x = torch.where(m)
    #             if len(y) == 0:
    #                 inst["valid"] = False
    #                 continue
    #             inst["valid"] = True
    #             ymin, xmin, ymax, xmax = y.min().item(), x.min().item(), y.max().item(), x.max().item()
    #             # 裁剪 mask 减少内存并统一逻辑
    #             inst["mask"] = m[ymin:ymax+1, xmin:xmax+1]
    #             inst["area"] = len(y)
    #             # 全图 bbox: [ymin, xmin, ymax, xmax] (左闭右开)
    #             inst["bbox"] = [ymin + inst["offset"][1], 
    #                             xmin + inst["offset"][0],
    #                             ymax + 1 + inst["offset"][1], 
    #                             xmax + 1 + inst["offset"][0]]

    #         # 过滤掉空的 mask
    #         valid_instances = [inst for inst in prediction if inst.get("valid", False)]

    #         if len(valid_instances) == 0:
    #             inst_map = np.zeros((H, W), dtype=np.uint16)
    #         else:
    #             # 2. 执行极速合并 (结果与原本 logic 一致)
    #             merged_instances = merge_instances_fast(
    #                 valid_instances,
    #                 iou_thr=0.1,    # ★ 跨 patch 必须低
    #                 overlap_thr=0.6
    #             )

    #             # 3. 映射到全图 instance map (内存安全：只在最后一步使用全图矩阵)
    #             inst_map = np.zeros((H, W), dtype=np.uint16)

    #             # 按 score 排序，确保 ID 分配有确定性（可选）
    #             merged_instances = sorted(merged_instances, key=lambda x: x.score, reverse=True)

    #             for idx, m_inst in enumerate(merged_instances):
    #                 inst_id = idx + 1
    #                 y0, x0, y1, x1 = m_inst.bbox
    #                 # 裁剪出全图对应区域
    #                 patch = inst_map[y0:y1, x0:x1]
    #                 m = m_inst.mask.cpu().numpy()
    #                 # 只在 mask 为 True 的地方打上 ID
    #                 patch[m] = inst_id

    #     # 4. 写文件
    #     fastly_save_image(
    #         inst_map,
    #         out_file,
    #         ref_file=ref_file,
    #         gdal_type=gdal.GDT_UInt16
    #     )

    # 实例分割 (原始慢速版，备查)
    # def write_on_batch_end(
    #     self,
    #     trainer,
    #     pl_module,
    #     prediction,
    #     batch_indices,
    #     batch,
    #     batch_idx,
    #     dataloader_idx=0
    # ):
    #     """
    #     prediction: List[Dict]
    #         {
    #             "mask": BoolTensor [h, w],
    #             "score": float,
    #             "label": int,
    #             "offset": (x0, y0)
    #         }
    #     """

    #     image = batch[0]
    #     H, W = image.shape[-2:]
    #     file_name = batch[3][0]

    #     ref_file = os.path.join(self.ref_dir, file_name)
    #     out_file = os.path.join(
    #         self.output["output_predict_dir"],
    #         file_name
    #     )

    #     # ---------- 映射到全图 mask ----------
    #     global_instances = []

    #     for inst in prediction:
    #         # ---------- 过滤背景 ----------
    #         # if inst["label"] == 0:  # 若 background 是最后一类
    #         #     continue

    #         mask = inst["mask"]
    #         h, w = mask.shape
    #         x0, y0 = inst["offset"]

    #         # 边界保护（非常重要，防止越界）
    #         x1 = min(x0 + w, W)
    #         y1 = min(y0 + h, H)
    #         mask = mask[:y1 - y0, :x1 - x0]
    #         gmask = torch.zeros((H, W), dtype=torch.bool)
    #         gmask[y0:y1, x0:x1] = mask.cpu()

    #         global_instances.append({
    #             "mask": gmask,
    #             "score": inst["score"],
    #             "label": inst["label"]
    #         })

    #     if len(global_instances) == 0:
    #         inst_map = np.zeros((H, W), dtype=np.uint16)
    #     else:
    #         # ---------- 按 score 排序（降低 merge 复杂度） ----------
    #         global_instances = sorted(
    #             global_instances,
    #             key=lambda x: x["score"],
    #             reverse=True
    #         )

    #         # ---------- merge（关键参数） ----------
    #         merged_instances = merge_instances(
    #             global_instances,
    #             iou_thr=0.1    # ★ 跨 patch 必须低
    #         )

    #         # ---------- 生成 instance map ----------
    #         inst_map = np.zeros((H, W), dtype=np.uint16)

    #         for idx, inst in enumerate(merged_instances):
    #             inst_id = idx + 1
    #             inst_map[inst["mask"].numpy()] = inst_id

    #     # ---------- 写文件 ----------
    #     fastly_save_image(
    #         inst_map,
    #         out_file,
    #         ref_file=ref_file,
    #         gdal_type=gdal.GDT_UInt16
    #     )
# ---------------------------------------------------------------------------------------------------------------------------------

# ------------------------保存特征---------------------------------------------------------------------------------------------------------
    # def write_on_batch_end(
    #     self,
    #     trainer,
    #     pl_module,
    #     prediction,
    #     batch_indices,
    #     batch,
    #     batch_idx,
    #     dataloader_idx=0
    # ):
    #     # image = batch[0]
    #     # H, W = image.shape[-2:]
    #     # file_name = batch[3][0]
    #     # file_name=file_name.split('.')[0]
    #     # save_path = os.path.join(
    #     #     self.output["output_predict_dir"],
    #     #     f"{file_name}.pt"
    #     # )

    #     # # prediction 就是整图 neck 特征 dict，直接保存
    #     # torch.save(prediction, save_path)

    # #     print(f"Saved full neck features to {save_path}")
    #         pass

    # def write_on_epoch_end(
    #     self,
    #     trainer: "pl.Trainer",
    #     pl_module: "pl.LightningModule",
    #     predictions: Sequence[Any],
    #     batch_indices: Sequence[Any],
    #     ) -> None:
    #         predict_metrics = pl_module.predict_metrics.compute()
    #         self._write_stat_for_end(predict_metrics, os.path.join(self.output['output_stat_dir'], 'global_statistics.csv'))
    # def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
    #         # 这里什么都不用做，因为我们在 Model 里打印/保存指标了
    #         pass
        
        
        
def mask_iou(mask1, mask2):
    inter = (mask1 & mask2).sum().float()
    union = (mask1 | mask2).sum().float()
    if union == 0:
        return 0.0
    return (inter / union).item()


def overlap_ratio(mask1, mask2):
    inter = (mask1 & mask2).sum().float()
    return (inter / min(mask1.sum(), mask2.sum()).clamp(min=1)).item()

def merge_instances(instances, iou_thr=0.1, overlap_thr=0.6):
    """原始慢速版合并 (用于结果对比)"""
    if len(instances) == 0:
        return []
    instances = sorted(instances, key=lambda x: x["score"], reverse=True)
    merged = []
    for inst in instances:
        merged_flag = False
        for m in merged:
            if inst["label"] != m["label"]:
                continue
            iou = mask_iou(inst["mask"], m["mask"])
            ov  = overlap_ratio(inst["mask"], m["mask"])
            if iou > iou_thr or ov > overlap_thr:
                m["mask"] |= inst["mask"]
                m["score"] = max(m["score"], inst["score"])
                merged_flag = True
                break
        if not merged_flag:
            merged.append(inst)
    return merged

class MergedInstance:
    """内部辅助类：管理合并后的实例信息，支持局部扩展"""
    def __init__(self, inst):
        self.label = inst["label"]
        self.score = inst["score"]
        self.bbox = list(inst["bbox"]) # [ymin, xmin, ymax, xmax]
        self.area = inst["area"]
        # 存储局部 mask 以及相对于全图的 offset
        self.mask = inst["mask"].clone().cpu()
        self.offset = inst["offset"]

    def calculate_overlap_with(self, other_inst):
        """计算当前合并实例与新实例的 IoU 和 Overlap"""
        b1, b2 = self.bbox, other_inst["bbox"]
        # 1. BBox 快速相交判断
        iy0, ix0 = max(b1[0], b2[0]), max(b1[1], b2[1])
        iy1, ix1 = min(b1[2], b2[2]), min(b1[3], b2[3])
        if iy1 <= iy0 or ix1 <= ix0:
            return 0.0, 0.0

        # 2. 提取公共区域的 mask 手段
        # self
        sy0, sx0 = self.bbox[0], self.bbox[1]
        sub_self = self.mask[int(iy0-sy0):int(iy1-sy0), int(ix0-sx0):int(ix1-sx0)]
        # other
        oy0, ox0 = other_inst["bbox"][0], other_inst["bbox"][1]
        sub_other = other_inst["mask"][int(iy0-oy0):int(iy1-oy0), int(ix0-ox0):int(ix1-ox0)]
        
        # 3. 计算交集
        inter = (sub_self & sub_other.cpu()).sum().float().item()
        if inter == 0: return 0.0, 0.0
        
        union = self.area + other_inst["area"] - inter
        return inter / union, inter / min(self.area, other_inst["area"])

    def merge(self, other_inst):
        """将新实例合并进来，必要时扩充局部 mask 边界"""
        self.score = max(self.score, other_inst["score"])
        b1, b2 = self.bbox, other_inst["bbox"]
        
        # 计算合并后的新边界
        new_bbox = [min(b1[0], b2[0]), min(b1[1], b2[1]),
                    max(b1[2], b2[2]), max(b1[3], b2[3])]
        
        if new_bbox != self.bbox:
            # 扩充局部 Mask
            H_new, W_new = int(new_bbox[2] - new_bbox[0]), int(new_bbox[3] - new_bbox[1])
            new_mask = torch.zeros((H_new, W_new), dtype=torch.bool)
            # 贴入旧数据
            dy, dx = int(self.bbox[0] - new_bbox[0]), int(self.bbox[1] - new_bbox[1])
            new_mask[dy:dy+self.mask.shape[0], dx:dx+self.mask.shape[1]] = self.mask
            self.mask = new_mask
            self.bbox = new_bbox
            self.offset = (new_bbox[1], new_bbox[0])

        # 贴入新实例数据
        dy, dx = int(other_inst["bbox"][0] - self.bbox[0]), int(other_inst["bbox"][1] - self.bbox[1])
        self.mask[dy:dy+other_inst["mask"].shape[0], dx:dx+other_inst["mask"].shape[1]] |= other_inst["mask"].cpu()
        
        # 更新面积
        self.area = self.mask.sum().item()

def merge_instances_fast(instances, iou_thr=0.1, overlap_thr=0.6):
    """极速版合并：基于 BBox 筛选和局部精确 IoU 计算"""
    if len(instances) == 0:
        return []

    # 按 score 排序 (保证确定性)
    instances = sorted(instances, key=lambda x: x["score"], reverse=True)
    merged = []

    for inst in instances:
        found_match = False
        # 遍历已合并的建筑
        for m in merged:
            if inst["label"] != m.label:
                continue
            
            # 内部执行 BBox 快速判断和局部像素计算
            iou, ov = m.calculate_overlap_with(inst)
            
            if iou > iou_thr or ov > overlap_thr:
                m.merge(inst)
                found_match = True
                break
        
        if not found_match:
            merged.append(MergedInstance(inst))

    return merged
