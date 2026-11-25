import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time
import torch
import math

from sampler.normal_predict_sampler import normal_sampler, normal_stitch, normal_sampler_with_dsm
from .model_interface import MInterface
from metric.tool import cal_eval_image
from loss.losses import SetCriterion, HungarianMatcher
from metric.instance_evaluation import InstanceSegEvaluator
import torch.nn.functional as F
from utils.featurestitcheravg import FeatureStitcherAvg
import numpy as np
import cv2


class NormalMInterface_DINOv3_mask2former(MInterface):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.val_evaluator = None
        matcher = HungarianMatcher(
            cost_class=2.0,
            cost_mask=5.0,
            cost_dice=5.0,
            # cost_height=1.0,
            num_points=12544,   # 例如 112*112，可根据 mask 分辨率调
        )
        num_decoder_layers=6
        base_weight_dict = {
            'loss_ce': 2.0,
            'loss_mask': 5.0,
            'loss_dice': 5.0,
            # 'loss_center_count': 2.0,  # 新增
        }

        weight_dict = base_weight_dict.copy()

        for i in range(num_decoder_layers - 1):
            for k, v in base_weight_dict.items():
                weight_dict[f"{k}_{i}"] = v
        self.loss = SetCriterion(
            num_classes=1,      # 建筑物单类
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=0.1,
            losses=["labels", "masks"],  # 新增 center_count
            num_points=12544,
            oversample_ratio=3.0,
            importance_sample_ratio=0.75,
        )
        self.predict_coco_evaluator = None
    def training_step(self, batch, batch_idx):
        x, dsm, y = batch[0], batch[1], batch[2]
            
        # 1. 前向传播
        pred = self([x, dsm])
            
        # 2. 计算 Loss
        loss_dict = self.loss(pred, y)
        total_loss = 0.0
        for k, v in loss_dict.items():
            if k in self.loss.weight_dict:
                total_loss += v * self.loss.weight_dict[k]

            # 正常日志记录
        self.log("train_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        for k, v in loss_dict.items():
            self.log(f"train_{k}", v, on_step=False, on_epoch=True, logger=True)
                
        return total_loss


    def validation_step(self, batch, batch_idx):
        x, dsm, y = batch[0], batch[1], batch[2]
        pred = self([x, dsm])
        
        loss_dict = self.loss(pred, y)
        total_loss = 0.0
        for k, v in loss_dict.items():
            if k in self.loss.weight_dict:
                total_loss += v * self.loss.weight_dict[k]
        
        self.log("val_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        p_sem, t_sem = self.batch_to_semantic(pred, y, x.shape[-2:])
        metrics=self.val_metrics(p_sem, t_sem) 
        self.log_dict(metrics, on_step=True,on_epoch=True, prog_bar=True, logger=True) # 自动 log 所有指标
        
        return{
            "pred": pred,
            "target": y,
            "image_size": x.shape[-2:]
        }

    
    
    # 只保存实例分割结果
    # def predict_step(self, batch, batch_idx):
    #     """
    #     Return:
    #         List[Dict]: 所有 patch 的实例（带 offset）
    #     """
    #     image, dsm = batch[0], batch[1]   # [1, C, H, W]
    #     _, _, H, W = image.shape

    #     crop_size = 512
    #     stride = 360   # ★ 强烈建议 ≤ crop/2

    #     all_instances = []

    #     for y0 in range(0, H, stride):
    #         for x0 in range(0, W, stride):
    #             # ---- 对齐 patch ----
    #             yy0 = max(0, min(y0, H - crop_size))
    #             xx0 = max(0, min(x0, W - crop_size))
    #             yy1 = yy0 + crop_size
    #             xx1 = xx0 + crop_size

    #             img_patch = image[:, :, yy0:yy1, xx0:xx1]
    #             dsm_patch = dsm[:, :, yy0:yy1, xx0:xx1]

    #             with torch.no_grad():
    #                 outputs = self([img_patch, dsm_patch])
    #                 patch_instances = self.decode_instances(
    #                     outputs,
    #                     score_thr=0.75,
    #                     patch_size=crop_size
    #                 )

    #             # ★ 只对“当前 patch”做 NMS（非常重要）
    #             patch_instances = instance_nms(
    #                 patch_instances,
    #                 iou_thr=0.5,
    #                 topk=10
    #             )

    #             for inst in patch_instances:
    #                 inst["offset"] = (xx0, yy0)
    #                 all_instances.append(inst)

    #     return all_instances

    
    
    # ------------保存特征 (流式拼图版本，极大节省内存) ------------
    def predict_step(self, batch, batch_idx):
        import os
        import torch
        image, dsm = batch[0], batch[1]
        _, _, H, W = image.shape

        crop_size = 512
        stride = 360
        
        strides_dict = {
            "res2": 4,
            "res3": 8,
            "res4": 16,
            "res5": 32,
        }

        # 1. 预先获取通道数以初始化容器
        with torch.no_grad():
            dummy_out = self([image[:, :, :crop_size, :crop_size], 
                               dsm[:, :, :crop_size, :crop_size] if dsm is not None else None])
            
        full_feats = {}
        count_maps = {}
        for level, s in strides_dict.items():
            C = dummy_out[level].shape[1]
            Hs, Ws = (H + s - 1) // s, (W + s - 1) // s
            # 设置为 CPU 上的 half 精度，这是内存最省的方案
            full_feats[level] = torch.zeros((C, Hs, Ws), dtype=torch.float16, device="cpu")
            count_maps[level] = torch.zeros((1, Hs, Ws), dtype=torch.float16, device="cpu")
            
        del dummy_out
        torch.cuda.empty_cache()

        # 2. 循环处理并“流式”累加
        for y0 in range(0, H, stride):
            for x0 in range(0, W, stride):
                yy0, xx0 = min(max(y0, 0), H - crop_size), min(max(x0, 0), W - crop_size)
                
                img_patch = image[:, :, yy0:yy0+crop_size, xx0:xx0+crop_size]
                dsm_patch = dsm[:, :, yy0:yy0+crop_size, xx0:xx0+crop_size] if dsm is not None else None

                with torch.no_grad():
                    outputs = self([img_patch, dsm_patch])

                for level, feat in outputs.items():
                    s = strides_dict[level]
                    _, _, h, w = feat.shape
                    fx0, fy0 = xx0 // s, yy0 // s
                    vh, vw = min(h, full_feats[level].shape[1]-fy0), min(w, full_feats[level].shape[2]-fx0)
                    
                    if vh <= 0 or vw <= 0: continue
                    
                    # 取出 patch，转 half 存入 CPU 容器，随后该 patch 显存就会被回收
                    full_feats[level][:, fy0:fy0+vh, fx0:fx0+vw] += feat[0, :, :vh, :vw].detach().half().cpu()
                    count_maps[level][:, fy0:fy0+vh, fx0:fx0+vw] += 1.0

        # 3. 计算加权平均
        for level in full_feats:
            full_feats[level] /= count_maps[level].clamp(min=1.0)
            
        # 🚀 在底层收集之前，自己保存，彻底阻断 PL 内存泄露
        root_dir = self.trainer.default_root_dir
        predict_dir = os.path.join(root_dir, "predict")
        os.makedirs(predict_dir, exist_ok=True)
        file_name = batch[3][0].split('.')[0]
        save_path = os.path.join(predict_dir, f"{file_name}.pt")
        
        torch.save(full_feats, save_path)
        print(f"✅ Saved full neck features to: {save_path}")

        # 清除本函数的临时变量
        del full_feats
        del count_maps

        # 4. 手动显存/内容清理提示
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        return None
    
        
    def on_validation_batch_end(
        self,
        outputs,
        batch,
        batch_idx,
        dataloader_idx=0
    ):
        # 所有 rank 都累积
        self.val_coco_evaluator.process(
            outputs["pred"],
            outputs["target"],
            image_size=outputs["image_size"]
        )
    def on_validation_epoch_start(self):
        self.val_coco_evaluator = InstanceSegEvaluator()
        
    def on_validation_epoch_end(self):
        coco_metrics = self.val_coco_evaluator.evaluate()

        # 关键：所有 rank 都 log，但只在 rank0 写 logger
        self.log_dict(
            {f"coco_{k}": v for k, v in coco_metrics.items()},
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=False,
            rank_zero_only=True,
        )

        
    # def on_predict_epoch_end(self):
    #         if not self.trainer.is_global_zero:
    #             return

    #         print("\n[Predict] Calculating COCO Metrics...")
            
    #         if self.predict_coco_evaluator is not None:
    #             coco_metrics = self.predict_coco_evaluator.evaluate()
                
    #             # 打印
    #             print("-" * 50)
    #             print("Predict Set Evaluation Results:")
    #             for k, v in coco_metrics.items():
    #                 print(f"{k:<30}: {v:.4f}")
    #             print("-" * 50)

    #             # --- [新增] 保存到 CSV ---
    #             import csv
    #             # 假设你想存到 log 目录或者指定目录
    #             save_dir = self.trainer.default_root_dir 
    #             csv_path = os.path.join(save_dir, 'coco_metrics.csv')
                
    #             try:
    #                 with open(csv_path, 'w', newline='') as f:
    #                     writer = csv.writer(f)
    #                     writer.writerow(['Metric', 'Value'])
    #                     for k, v in coco_metrics.items():
    #                         writer.writerow([k, v])
    #                 print(f"COCO Metrics saved to: {csv_path}")
    #             except Exception as e:
    #                 print(f"Failed to save CSV: {e}")

    #         self.predict_coco_evaluator = None
        
    def batch_to_semantic(self, outputs, targets, image_size):
        """
        辅助函数：将 Mask2Former 的 Batch 输出转换为语义分割 Tensor
        Return:
            pred_tensor: (B, H, W) 0/1 张量
            target_tensor: (B, H, W) 0/1 张量
        """
        sem_preds = []
        sem_targets = []
        
        # 遍历 Batch 中的每一张图
        for i in range(len(targets)):
            # --- A. 处理预测 (Pred) ---
            # 1. 取出 logits 和 masks
            p_logits = outputs['pred_logits'][i] # [N, 2]
            p_masks = outputs['pred_masks'][i]   # [N, h, w]
            
            # 2. 筛选 (置信度 > 0.5)
            scores = p_logits.softmax(dim=-1)[:, 0] # 取出 building 类概率
            keep = scores > 0.5
            
            if keep.sum() > 0:
                valid_masks = p_masks[keep]
                # 上采样到原图大小 (512x512)
                valid_masks = torch.nn.functional.interpolate(
                    valid_masks.unsqueeze(1), 
                    size=image_size, 
                    mode='bilinear', align_corners=False
                ).squeeze(1)
                # Sigmoid + 合并 (Any)
                sem_p = (valid_masks.sigmoid() > 0.5).any(dim=0).long()
            else:
                sem_p = torch.zeros(image_size, dtype=torch.long, device=p_masks.device)
            
            sem_preds.append(sem_p)

            # --- B. 处理真值 (Target) ---
            # targets[i]['masks'] 是 [N, H, W]
            t_masks = targets[i]['masks']
            if t_masks.shape[0] > 0:
                sem_t = (t_masks.sum(dim=0) > 0).long()
            else:
                sem_t = torch.zeros(image_size, dtype=torch.long, device=p_masks.device)
            
            sem_targets.append(sem_t)

        # 堆叠回 Batch 维度
        return torch.stack(sem_preds), torch.stack(sem_targets)

    
    def decode_instances(self, outputs, score_thr=0.3, patch_size=1024):
        results = []

        pred_logits = outputs["pred_logits"][0]   # [Q, C+1]
        pred_masks  = outputs["pred_masks"][0]    # [Q, h, w]

        probs = pred_logits.softmax(-1)
        scores, labels = probs[:, :-1].max(-1)

        keep = scores > score_thr
        scores = scores[keep]
        labels = labels[keep]
        masks  = pred_masks[keep]

        if masks.numel() == 0:
            return results

        # 上采样到 patch 大小
        masks = torch.nn.functional.interpolate(
            masks.unsqueeze(1),
            size=(patch_size, patch_size),
            mode="bilinear",
            align_corners=False
        ).squeeze(1)

        for i in range(masks.shape[0]):
            # ★ 用 logit threshold（Mask2Former 正确姿势）
            mask = masks[i] > 0

            # 小碎片过滤 (根据遥感地图分辨率，建筑物通常大于一定像素，建议提高)
            if mask.sum() < 400: # 大约 20x20 像素
                continue

            border = touches_border(mask, margin=16)

            results.append({
                "mask": mask,
                "score": scores[i].item() * (0.5 if border else 1.0),
                "label": labels[i].item(),
            })


        return results

    

def mask_iou(mask1, mask2):
    inter = (mask1 & mask2).sum().float()
    union = (mask1 | mask2).sum().float()
    if union == 0:
        return 0.0
    return (inter / union).item()


def overlap_ratio(mask1, mask2):
    inter = (mask1 & mask2).sum().float()
    return (inter / min(mask1.sum(), mask2.sum()).clamp(min=1)).item()



def instance_nms(instances, iou_thr=0.5, topk=20):
    """
    instances: List[Dict(mask, score, label)]
    """
    if len(instances) == 0:
        return []

    instances = sorted(instances, key=lambda x: x["score"], reverse=True)

    keep = []
    for inst in instances:
        discard = False
        for k in keep:
            if inst["label"] != k["label"]:
                continue
            if mask_iou(inst["mask"], k["mask"]) > iou_thr:
                discard = True
                break
        if not discard:
            keep.append(inst)
        if len(keep) >= topk:
            break

    return keep


def touches_border(mask, margin=16):
    h, w = mask.shape
    return (
        mask[:margin, :].any() or
        mask[-margin:, :].any() or
        mask[:, :margin].any() or
        mask[:, -margin:].any()
    )
    
    
def stitch_neck_features_dict(
    all_neck_feats,
    image_shape,
    strides_dict,
):
    """
    Args:
        all_neck_feats: List[{
            "offset": (x0, y0),
            "feat": Dict[str, Tensor[C, h, w]]
        }]
        image_shape: (H, W) of original image
        strides_dict: Dict[str, int], e.g. {"res2":4, ...}

    Returns:
        full_neck_feats: Dict[str, Tensor[C, H_s, W_s]]
    """

    H, W = image_shape

    # ===============================
    # 1. 初始化 full feature + count
    # ===============================
    full_feats = {}
    count_maps = {}

    for level, stride in strides_dict.items():
        # 找一个样本确定 C
        sample_feat = all_neck_feats[0]["feat"][level]
        C = sample_feat.shape[0]

        Hs = (H + stride - 1) // stride
        Ws = (W + stride - 1) // stride

        full_feats[level] = torch.zeros((C, Hs, Ws), dtype=sample_feat.dtype)
        count_maps[level] = torch.zeros(
            (1, Hs, Ws),
            dtype=sample_feat.dtype,   # ⭐ 跟 feature 一致（fp16）
        )

    # ===============================
    # 2. 累加所有 patch 特征
    # ===============================
    for item in all_neck_feats:
        x0, y0 = item["offset"]
        feats = item["feat"]

        for level, feat in feats.items():
            stride = strides_dict[level]

            _, h, w = feat.shape

            # 原图像素 → 特征图坐标
            fx0 = x0 // stride
            fy0 = y0 // stride
            fx1 = fx0 + w
            fy1 = fy0 + h

            # 安全裁剪（防止越界）
            fx1 = min(fx1, full_feats[level].shape[2])
            fy1 = min(fy1, full_feats[level].shape[1])

            valid_w = fx1 - fx0
            valid_h = fy1 - fy0

            if valid_w <= 0 or valid_h <= 0:
                continue

            full_feats[level][:, fy0:fy1, fx0:fx1] += feat[:, :valid_h, :valid_w]
            count_maps[level][:, fy0:fy1, fx0:fx1] += 1.0

    # ===============================
    # 3. 重叠区域取平均
    # ===============================
    for level in full_feats:
        count = count_maps[level]
        count[count == 0] = 1.0  # 防止除零
        full_feats[level] = full_feats[level] / count

    return full_feats
