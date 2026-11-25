# import numpy as np
# import torch
# from pycocotools.coco import COCO
# from pycocotools import mask as mask_utils
# from pycocotools.cocoeval import COCOeval


# class InstanceSegEvaluator:
#     """
#     Lightning-friendly COCO Instance Segmentation Evaluator
#     - No Detectron2 dependency
#     - No dataset_name needed
#     """

#     def __init__(self, iou_type="segm"):
#         assert iou_type == "segm"
#         self.iou_type = iou_type
#         self.reset()

#     def reset(self):
#         self.gt_images = []
#         self.gt_annotations = []
#         self.pred_annotations = []

#         self._img_id = 0
#         self._ann_id = 0

#     @torch.no_grad()
#     def process(self, outputs, targets, image_size):
#         """
#         Args:
#             outputs: model outputs (Mask2Former style)
#             targets: list of target dicts
#             image_size: (H, W)
#         """
#         H, W = image_size

#         for i in range(len(targets)):
#             img_id = int(targets[i]["image_id"])

#             # ---------- GT ----------
#             self.gt_images.append({
#                 "id": img_id,
#                 "height": H,
#                 "width": W,
#             })

#             gt_masks = targets[i]["masks"]  # [N, H, W]
#             for m in gt_masks:
#                 rle = mask_utils.encode(
#                     np.asfortranarray(m.cpu().numpy().astype(np.uint8))
#                 )
#                 rle["counts"] = rle["counts"].decode("utf-8")

#                 self.gt_annotations.append({
#                     "id": self._ann_id,
#                     "image_id": img_id,
#                     "category_id": 1,
#                     "segmentation": rle,
#                     "area": float(mask_utils.area(rle)),
#                     "bbox": mask_utils.toBbox(rle).tolist(),
#                     "iscrowd": 0,
#                 })
#                 self._ann_id += 1

#             # ---------- Predictions ----------
#             p_logits = outputs["pred_logits"][i]  # [Q, 2]
#             p_masks = outputs["pred_masks"][i]    # [Q, h, w]

#             scores = p_logits.softmax(dim=-1)[:, 0]
#             keep = scores > 0.5
#             if keep.sum() == 0:
#                 continue

#             scores = scores[keep]
#             masks = p_masks[keep]

#             masks = torch.nn.functional.interpolate(
#                 masks.unsqueeze(1),
#                 size=(H, W),
#                 mode="bilinear",
#                 align_corners=False
#             ).squeeze(1).sigmoid() > 0.5

#             for s, m in zip(scores, masks):
#                 rle = mask_utils.encode(
#                     np.asfortranarray(m.cpu().numpy().astype(np.uint8))
#                 )
#                 rle["counts"] = rle["counts"].decode("utf-8")

#                 self.pred_annotations.append({
#                     "image_id": img_id,
#                     "category_id": 1,
#                     "segmentation": rle,
#                     "score": float(s),
#                 })

#     def evaluate(self):
#         if len(self.pred_annotations) == 0:
#             return {
#                 "AP": 0.0,
#                 "AP50": 0.0,
#                 "AP75": 0.0,
#                 "AR": 0.0,
#             }

#         coco_gt = COCO()
#         coco_gt.dataset = {
#             "images": self.gt_images,
#             "annotations": self.gt_annotations,
#             "categories": [{"id": 1, "name": "building"}],
#         }
#         coco_gt.createIndex()

#         coco_dt = coco_gt.loadRes(self.pred_annotations)

#         coco_eval = COCOeval(coco_gt, coco_dt, self.iou_type)
#         coco_eval.evaluate()
#         coco_eval.accumulate()
#         coco_eval.summarize()

#         return {
#             "AP": coco_eval.stats[0] * 100,
#             "AP50": coco_eval.stats[1] * 100,
#             "AP75": coco_eval.stats[2] * 100,
#             "AR": coco_eval.stats[8] * 100,
#         }


import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools import mask as mask_utils
from pycocotools.cocoeval import COCOeval


class InstanceSegEvaluator:
    """
    Lightning-friendly COCO Instance Segmentation Evaluator
    Supports both:
    1. Standard/SOLOv2 format: List[Dict] -> [{'masks':, 'scores':}, ...]
    2. Mask2Former format: Dict -> {'pred_logits':, 'pred_masks':}
    """

    def __init__(self, iou_type="segm"):
        assert iou_type == "segm"
        self.iou_type = iou_type
        self.reset()

    def reset(self):
        self.gt_images = []
        self.gt_annotations = []
        self.pred_annotations = []

        self._img_id = 0
        self._ann_id = 0
        self._registered_img_ids = set() # 防止 image_id 重复添加

    @torch.no_grad()
    def process(self, outputs, targets, image_size):
        """
        Args:
            outputs: 
                - Case A (SOLOv2): List[Dict] -> [{'masks': [N,H,W], 'scores': [N]}, ...]
                - Case B (Mask2Former): Dict -> {'pred_logits': [B,Q,C], 'pred_masks': ...}
            targets: List[Dict] -> [{'image_id': int, 'masks': [N,H,W]}, ...]
            image_size: (H, W)
        """
        H, W = image_size

        # 遍历 Batch
        for i in range(len(targets)):
            img_id = int(targets[i]["image_id"])

            # -----------------------------
            # 1. 处理 Ground Truth (GT)
            # -----------------------------
            # 注册 Image (如果 batch 内有重复图或跨 batch，需去重)
            if img_id not in self._registered_img_ids:
                self.gt_images.append({
                    "id": img_id,
                    "height": H,
                    "width": W,
                })
                self._registered_img_ids.add(img_id)

            gt_masks = targets[i]["masks"]  # [N, H, W]
            
            # 转 numpy
            if isinstance(gt_masks, torch.Tensor):
                gt_masks = gt_masks.cpu().numpy()

            for m in gt_masks:
                # 编码 RLE
                rle = mask_utils.encode(
                    np.asfortranarray(m.astype(np.uint8))
                )
                rle["counts"] = rle["counts"].decode("utf-8")

                self.gt_annotations.append({
                    "id": self._ann_id,
                    "image_id": img_id,
                    "category_id": 1,
                    "segmentation": rle,
                    "area": float(mask_utils.area(rle)),
                    "bbox": mask_utils.toBbox(rle).tolist(),
                    "iscrowd": 0,
                })
                self._ann_id += 1

            # -----------------------------
            # 2. 处理 Predictions (核心修改)
            # -----------------------------
            
            # === 分支 A: 新格式 (SOLOv2 / Standard) ===
            if isinstance(outputs, list):
                pred = outputs[i]
                masks = pred["masks"]   # [N, H, W]
                scores = pred["scores"] # [N]
                # labels = pred["labels"] # 只有单类的话可以不用，或者用来设置 category_id

            # === 分支 B: 旧格式 (Mask2Former) ===
            elif isinstance(outputs, dict) and "pred_logits" in outputs:
                p_logits = outputs["pred_logits"][i]  # [Q, 2]
                p_masks = outputs["pred_masks"][i]    # [Q, h, w]

                scores = p_logits.softmax(dim=-1)[:, 0]
                keep = scores > 0.5
                if keep.sum() == 0:
                    continue

                scores = scores[keep]
                masks_raw = p_masks[keep]

                # 上采样 + Sigmoid
                masks = torch.nn.functional.interpolate(
                    masks_raw.unsqueeze(1),
                    size=(H, W),
                    mode="bilinear",
                    align_corners=False
                ).squeeze(1).sigmoid() > 0.5
            else:
                raise TypeError(f"Unknown output format: {type(outputs)}")

            # --- 统一后处理 (转 Numpy -> RLE) ---
            if isinstance(masks, torch.Tensor):
                masks = masks.cpu().numpy()
            if isinstance(scores, torch.Tensor):
                scores = scores.cpu().numpy()

            if len(scores) == 0:
                continue

            for s, m in zip(scores, masks):
                # 编码
                rle = mask_utils.encode(
                    np.asfortranarray(m.astype(np.uint8))
                )
                rle["counts"] = rle["counts"].decode("utf-8")

                self.pred_annotations.append({
                    "image_id": img_id,
                    "category_id": 1, # 单类 Building
                    "segmentation": rle,
                    "score": float(s),
                })

    def evaluate(self):
        if len(self.pred_annotations) == 0:
            print("No predictions found for evaluation.")
            return {
                "AP": 0.0, "AP50": 0.0, "AP75": 0.0, "AR": 0.0,
            }

        # 构建 COCO GT 对象
        coco_gt = COCO()
        coco_gt.dataset = {
            "images": self.gt_images,
            "annotations": self.gt_annotations,
            "categories": [{"id": 1, "name": "building"}],
        }
        coco_gt.createIndex()

        # 构建 COCO DT (Result) 对象
        # pycocotools 可能会有些 noisy output，这里可以加 try-except
        try:
            coco_dt = coco_gt.loadRes(self.pred_annotations)
        except Exception as e:
            print(f"Error loading results into COCO: {e}")
            return {"AP": 0.0}

        coco_eval = COCOeval(coco_gt, coco_dt, self.iou_type)
        
        # 将 image_ids 限制在验证集出现的 ID 中 (防止 id 错乱)
        img_ids = sorted([img['id'] for img in self.gt_images])
        coco_eval.params.imgIds = img_ids
        
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        return {
            "AP": coco_eval.stats[0] * 100,
            "AP50": coco_eval.stats[1] * 100,
            "AP75": coco_eval.stats[2] * 100,
            "AR": coco_eval.stats[8] * 100,
        }
