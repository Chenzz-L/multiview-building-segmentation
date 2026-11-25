# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import torch
from pycocotools.coco import COCO
import torch.nn.functional as F
from data.transforms.augmentation import seg_augmentation, build_agument_pipeline
from detectron2.structures import Instances, Boxes, BitMasks
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from data.datasets.dataset_Base import BaseDataset
from data.datasets.dataste_support import regist_dataset
import albumentations as A
from torchvision.ops import masks_to_boxes
__all__ = ['Dataset_RGB_DSM_COCO_Simple_solov2_partdata']


@regist_dataset()
class Dataset_RGB_DSM_COCO_Simple_solov2_partdata(BaseDataset):
    def __init__(
        self,
        data_dir,
        json_name="train.json",
        augmentation=True,
        standardization=True,
        dsm_none_value=-99999,
        geom_pipeline=None,
        rgb_pipeline=None,
        split_k=5,  # <--- [新增] 默认1表示全量，设为5表示只用1/5
        **kwargs
    ):
        super().__init__(data_dir, augmentation, standardization, **kwargs)

        self.image_dir = os.path.join(data_dir, "image")
        self.dsm_dir = os.path.join(data_dir, "dsm")

        self.coco = COCO(os.path.join(data_dir, json_name))
        
        # 1. 加载所有 ID 并排序（保证多卡顺序一致）
        self.ids = sorted(self.coco.imgs.keys())

        # 2. [核心修改] 备份全量 ID 并计算固定长度
        self.split_k = split_k
        self.all_ids = list(self.ids)  # 深度拷贝备份
        
        if self.split_k > 1:
            # 向下取整，计算每个分片的最小固定长度
            # 作用：保证 Epoch 0, 1, 2 的长度完全一致，避免 DDP 报错
            self.fixed_length = len(self.all_ids) // self.split_k
            
            # 初始化第一个分片
            self.resample_views(seed=0)
        else:
            self.fixed_length = len(self.all_ids)

        self.mean = np.array([0.430, 0.411, 0.296], np.float32).reshape(3, 1, 1)
        self.std = np.array([0.213, 0.156, 0.143], np.float32).reshape(3, 1, 1)

        # 几何增强
        self.geom_aug = (
            A.Compose(
                build_agument_pipeline(geom_pipeline).transforms,
                additional_targets={"dsm": "mask"}
            )
            if augmentation and geom_pipeline is not None
            else None
        )

        # 颜色增强
        self.rgb_aug = (
            A.Compose(build_agument_pipeline(rgb_pipeline).transforms)
            if augmentation and rgb_pipeline is not None
            else None
        )

        self.dsm_none_value = dsm_none_value

    def resample_views(self, seed: int):
        """
        根据 Epoch (seed) 动态切换数据分片，并强制截断到固定长度
        """
        if self.split_k <= 1:
            return

        # 1. 计算偏移量
        offset = seed % self.split_k
        
        # 2. 步长切片 (Strided Slicing)
        # 假设 ids 是 [1, 2, 3, 4, 5, 6...]
        # offset 0 -> [1, 6...] (跳过中间高重叠的)
        subset = self.all_ids[offset::self.split_k]
        
        # 3. [关键] 截断到固定长度，防止 IndexError
        if len(subset) > self.fixed_length:
            subset = subset[:self.fixed_length]
            
        # 4. 更新当前生效的 ids
        self.ids = subset

    def __len__(self):
        # 始终返回固定长度
        return len(self.ids)

    def __getitem__(self, idx):
        # idx 范围自动适配当前的 self.ids
        img_id = self.ids[idx]
        
        img_info = self.coco.loadImgs(img_id)[0]
        file_name = img_info["file_name"]
        
        # 初始尺寸
        image_h, image_w = img_info["height"], img_info["width"]

        # ---------- load RGB ----------
        image = self.open(
            os.path.join(self.image_dir, file_name),
            channel_sum=3
        )  # (3, H, W)

        # ---------- load DSM ----------
        dsm = self.open(
            os.path.join(self.dsm_dir, file_name),
            channel_sum=1,
            none_value=self.dsm_none_value
        )[0]  # (H, W)

        # ---------- load COCO masks ----------
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        if len(anns) > 0:
            masks = np.stack(
                [self.coco.annToMask(a) for a in anns],
                axis=0
            ).astype(np.uint8)  # (N, H, W)
        else:
            masks = np.zeros((0, image.shape[1], image.shape[2]), dtype=np.uint8)

        # ---------- geometric augmentation ----------
        if self.geom_aug is not None:
            image_hwc = image.transpose(1, 2, 0)
            
            # Albumentations 处理
            if masks.shape[0] > 0:
                auged = self.geom_aug(
                    image=image_hwc,
                    dsm=dsm,
                    masks=list(masks) # list input needed
                )
                if len(auged["masks"]) > 0:
                    masks = np.stack(auged["masks"], axis=0)
                else:
                    masks = np.zeros((0, image.shape[1], image.shape[2]), dtype=np.uint8)
            else:
                auged = self.geom_aug(image=image_hwc, dsm=dsm)
                masks = np.zeros((0, image.shape[1], image.shape[2]), dtype=np.uint8)

            image = auged["image"].transpose(2, 0, 1)
            dsm = auged["dsm"][None, :, :]
            
            # 更新尺寸 (Augmentation 可能会 Crop/Resize)
            image_h, image_w = image.shape[1], image.shape[2]
        else:
            dsm = dsm[None, :, :]

        # ---------- RGB-only augmentation ----------
        if self.rgb_aug is not None:
            image = self.rgb_aug(
                image=image.transpose(1, 2, 0)
            )["image"].transpose(2, 0, 1)

        # ---------- normalize RGB ----------
        if self.standardization:
            image = (image.astype(np.float32) - self.mean) / self.std

        img_tensor = torch.from_numpy(image).float()
        dsm_tensor = torch.from_numpy(dsm).float()

        # ---------- 构建 Target ----------
        target_dict = self._build_target(masks, (image_h, image_w), img_id)

        return img_tensor, dsm_tensor, target_dict

    def _build_target(self, masks, img_size, img_id):
        """
        构建 Detectron2 格式的 Target，并处理空 Mask 异常
        """
        h, w = img_size
        target_instances = Instances((h, w))

        if masks.ndim == 3 and masks.shape[0] > 0:
            masks_tensor = torch.from_numpy(masks)
            
            valid_inds = masks_tensor.sum(dim=(1, 2)) > 0
            masks_tensor = masks_tensor[valid_inds]
            if masks_tensor.shape[0] == 0:
                target_instances.gt_masks = BitMasks(torch.zeros((0, h, w), dtype=torch.bool))
                target_instances.gt_boxes = Boxes(torch.zeros((0, 4)))
                target_instances.gt_classes = torch.zeros(0, dtype=torch.long)
            else:
                boxes = masks_to_boxes(masks_tensor)
                keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
                boxes = boxes[keep]
                masks_tensor = masks_tensor[keep]
                
                target_instances.gt_masks = BitMasks(masks_tensor)
                target_instances.gt_boxes = Boxes(boxes)
                target_instances.gt_classes = torch.zeros(len(boxes), dtype=torch.long)
                
        else:
            target_instances.gt_masks = BitMasks(torch.zeros((0, h, w), dtype=torch.bool))
            target_instances.gt_boxes = Boxes(torch.zeros((0, 4)))
            target_instances.gt_classes = torch.zeros(0, dtype=torch.long)

        return {
            "image_id": img_id,
            "instances": target_instances
        }

    @staticmethod
    def collate_fn(batch):
        imgs, dsms, targets = zip(*batch)
        imgs = torch.stack(imgs, dim=0)
        dsms = torch.stack(dsms, dim=0)
        
        stride = 32
        h, w = imgs.shape[2], imgs.shape[3]
        if h % stride != 0 or w % stride != 0:
            new_h = (h + stride - 1) // stride * stride
            new_w = (w + stride - 1) // stride * stride
            pad_h = new_h - h
            pad_w = new_w - w
            imgs = F.pad(imgs, (0, pad_w, 0, pad_h), value=0)
            dsms = F.pad(dsms, (0, pad_w, 0, pad_h), value=0) 
            
        return imgs, dsms, list(targets)
