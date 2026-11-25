# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import torch
from pycocotools.coco import COCO

from data.transforms.augmentation import seg_augmentation, build_agument_pipeline

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from data.datasets.dataset_Base import BaseDataset
from data.datasets.dataste_support import regist_dataset
import albumentations as A
__all__ = ['Inference_dataset_RGB_DSM_COCO_Simple']


@regist_dataset()
class Inference_dataset_RGB_DSM_COCO_Simple(BaseDataset):
    def __init__(
        self,
        data_dir,
        json_name="train.json",
        augmentation=True,
        standardization=True,
        dsm_none_value=-99999,
        geom_pipeline=None,
        rgb_pipeline=None,
        **kwargs
    ):
        super().__init__(data_dir, augmentation, standardization, **kwargs)

        self.image_dir = os.path.join(data_dir, "image")
        self.dsm_dir = os.path.join(data_dir, "dsm")

        self.coco = COCO(os.path.join(data_dir, json_name))
        self.ids = sorted(self.coco.imgs.keys())

        self.mean = np.array([0.430, 0.411, 0.296], np.float32).reshape(3, 1, 1)
        self.std = np.array([0.213, 0.156, 0.143], np.float32).reshape(3, 1, 1)

        self.geom_aug = (
            A.Compose(
                build_agument_pipeline(geom_pipeline).transforms,
                additional_targets={"dsm": "mask"}
            )
            if augmentation and geom_pipeline is not None
            else None
        )

        self.rgb_aug = (
            A.Compose(build_agument_pipeline(rgb_pipeline).transforms)
            if augmentation and rgb_pipeline is not None
            else None
        )

        self.dsm_none_value = dsm_none_value

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        file_name = self.coco.loadImgs(img_id)[0]["file_name"]

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

            if masks.shape[0] > 0:
                auged = self.geom_aug(
                    image=image_hwc,
                    dsm=dsm,
                    masks=list(masks)
                )
                masks = np.stack(auged["masks"], axis=0)
            else:
                auged = self.geom_aug(
                    image=image_hwc,
                    dsm=dsm
                )
                masks = np.zeros(
                    (0, image.shape[1], image.shape[2]),
                    dtype=np.uint8
                )

            image = auged["image"].transpose(2, 0, 1)
            dsm = auged["dsm"][None, :, :]
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
        target = self._build_target(masks, img_id)

        return img_tensor, dsm_tensor, target, file_name

    def _build_target(self, masks, img_id):
        if masks.ndim == 3:
            masks_tensor = torch.from_numpy(masks).float()
            labels_tensor = torch.zeros(masks.shape[0], dtype=torch.long)
        else:
            masks_tensor = torch.zeros((0, 0, 0), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.long)

        return {
            "masks": masks_tensor,
            "labels": labels_tensor,
            "image_id": img_id,
        }

    @staticmethod
    def collate_fn(batch):
        imgs, dsms, targets, file_name = zip(*batch)
        return (
            torch.stack(imgs, dim=0),
            torch.stack(dsms, dim=0),
            list(targets),
            file_name,
        )
