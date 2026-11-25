import os
import sys
import torch
import torch.nn.functional as F

# 引用你的基类
from .model_interface import MInterface
from metric.instance_evaluation import InstanceSegEvaluator

# 如果你保留了 instance_nms 工具函数，可以直接用，或者用 torchvision.ops.nms
# SOLOv2 内部已经做了 Matrix NMS，通常不需要再做 heavy NMS，除非做 Patch Stitching

class NormalMInterface_DINOv3_SOLOv2(MInterface):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.val_evaluator = None
        self.predict_coco_evaluator = None
    def forward(self, x, targets=None):
        return self.model(x, targets=targets) 
    def training_step(self, batch, batch_idx):
        x, dsm, targets = batch[0], batch[1], batch[2]
        current_batch_size = x.shape[0]
        gt_instances_list = [t["instances"] for t in targets]
        loss_dict = self([x, dsm], targets=gt_instances_list)
        total_loss = sum(loss_dict.values())
        self.log("train_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=current_batch_size)
        for k, v in loss_dict.items():
            self.log(f"train_{k}", v, on_step=False, on_epoch=True, logger=True, batch_size=current_batch_size)
                
        return total_loss

    def validation_step(self, batch, batch_idx):
        x, dsm, targets = batch[0], batch[1], batch[2]
        current_batch_size = x.shape[0]
        preds_instances = self([x, dsm])
        gt_instances_list = [t["instances"] for t in targets]
        p_sem, t_sem = self.batch_to_semantic(preds_instances, gt_instances_list, x.shape[-2:])
        metrics = self.val_metrics(p_sem, t_sem) 
        self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=current_batch_size)
        return {
            "pred": preds_instances,
            "target": targets,
            "image_size": x.shape[-2:]
        }
    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        if self.val_coco_evaluator is None:
            return
        raw_preds = outputs["pred"]
        raw_targets = outputs["target"]
        image_size = outputs["image_size"]
        converted_preds = []
        for inst in raw_preds:
            converted_preds.append({
                "masks": inst.pred_masks.cpu(),
                "scores": inst.scores.cpu(),
                "labels": inst.pred_classes.cpu(),
            })
        converted_targets = []
        for t in raw_targets:
            img_id = t["image_id"]
            gt_inst = t["instances"]
            if hasattr(gt_inst.gt_masks, 'tensor'):
                gt_masks = gt_inst.gt_masks.tensor
            else:
                gt_masks = gt_inst.gt_masks
            converted_targets.append({
                "image_id": img_id,
                "masks": gt_masks.cpu(),
                "labels": gt_inst.gt_classes.cpu(),
            })
        self.val_coco_evaluator.process(
            converted_preds,
            converted_targets,
            image_size=image_size
        )

    def on_validation_epoch_start(self):
        self.val_coco_evaluator = InstanceSegEvaluator()
        
    def on_validation_epoch_end(self):
        if self.val_coco_evaluator is not None:
            coco_metrics = self.val_coco_evaluator.evaluate()
            self.log_dict(
                {f"coco_{k}": v for k, v in coco_metrics.items()},
                on_epoch=True, prog_bar=True, logger=True, sync_dist=False, rank_zero_only=True,
            )

    def predict_step(self, batch, batch_idx):
        """
        大图切片推理逻辑
        Return: List[Dict]: 所有 patch 的实例（带全局坐标 offset）
        """
        image, dsm = batch[0], batch[1]   # [1, C, H, W]
        _, _, H, W = image.shape

        crop_size = 1024 # 根据显存调整
        stride = 800     # 建议有一定重叠

        all_instances = []

        # 1. 滑动窗口
        for y0 in range(0, H, stride):
            for x0 in range(0, W, stride):
                # 边界处理
                yy0 = min(y0, H - crop_size); yy0 = max(yy0, 0)
                xx0 = min(x0, W - crop_size); xx0 = max(xx0, 0)
                yy1 = yy0 + crop_size
                xx1 = xx0 + crop_size

                img_patch = image[:, :, yy0:yy1, xx0:xx1]
                dsm_patch = dsm[:, :, yy0:yy1, xx0:xx1]

                # 2. 推理
                with torch.no_grad():
                    # SOLOv2 直接返回 List[Instances]
                    # List 长度为 1 (batch_size=1)
                    results = self([img_patch, dsm_patch])
                    patch_res = results[0] # Instances 对象

                # 3. 解码并添加 Offset
                # patch_res 包含 .pred_masks, .scores, .pred_classes
                if len(patch_res) == 0:
                    continue
                
                # 转为 CPU 并处理
                masks = patch_res.pred_masks.cpu() # [N, H, W] BoolTensor
                scores = patch_res.scores.cpu()
                labels = patch_res.pred_classes.cpu()

                for i in range(len(scores)):
                    mask = masks[i]
                    # 只有 Mask 有效才保留
                    if mask.sum() == 0: continue
                    
                    all_instances.append({
                        "mask": mask,           # 局部 Mask (1024x1024)
                        "score": scores[i].item(),
                        "label": labels[i].item(),
                        "offset": (xx0, yy0)    # 关键：记录偏移量
                    })

        return all_instances

    # =====================================================================
    #  辅助工具函数
    # =====================================================================
    
    def batch_to_semantic(self, preds_instances, targets, img_size):
        """
        将 SOLOv2 的 Instances 转换为语义分割 Mask (B, H, W)
        用于计算 Dice/IoU 监控训练
        """
        # 1. 处理 GT
        gt_list = []
        for t in targets:
            # Detectron2 Instances -> BitMasks -> Tensor
            if hasattr(t, 'gt_masks'):
                mask = t.gt_masks.tensor
                if mask.shape[0] > 0:
                    mask = mask.any(dim=0).long()
                else:
                    mask = torch.zeros(img_size, dtype=torch.long, device=self.device)
            else:
                mask = torch.zeros(img_size, dtype=torch.long, device=self.device)
            gt_list.append(mask)
        t_sem = torch.stack(gt_list)

        # 2. 处理 Preds
        pred_list = []
        for p in preds_instances:
            # p 是 Instances 对象
            if len(p) > 0:
                mask = p.pred_masks # [N, H, W]
                mask = mask.any(dim=0).long()
            else:
                mask = torch.zeros(img_size, dtype=torch.long, device=self.device)
            pred_list.append(mask)
        p_sem = torch.stack(pred_list)

        return p_sem, t_sem
