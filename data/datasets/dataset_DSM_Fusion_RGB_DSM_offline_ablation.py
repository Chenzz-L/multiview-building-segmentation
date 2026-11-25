"""Certified full-pool max / K-means21 inputs for independent fusion ablations."""
import copy
import json
from pathlib import Path

import torch

from data.datasets.dataset_Base import BaseDataset
from data.datasets.dataste_support import regist_dataset
from data.datasets.dataset_DSM_Fusion_RGB_DSM_sail_memory import (
    FEATURE_SHAPES, SCALES, _contains_geometric_augmentation, _load_torch,
)
from data.transforms.augmentation import seg_augmentation

__all__ = ["Dataset_DSM_Fusion_RGB_DSM_OfflineAblation"]


@regist_dataset()
class Dataset_DSM_Fusion_RGB_DSM_OfflineAblation(BaseDataset):
    """Read offline tensors only; exclude padding, never exclude zero pixels.

    Optional explicit patch lists match a historical cohort without consulting
    anchor libraries or silently dropping image-free samples.
    """
    def __init__(self, augmentation, feature_data_dir, xyz_data_dir, aggregation,
                 indices_name="indices", standardization=False, clip=None,
                 data_pipeline=None, xyz_dsm_none_value=-99999,
                 patch_list_path=None, expected_groups=21):
        if aggregation not in ("max", "cluster21"):
            raise ValueError("aggregation must be max or cluster21")
        if _contains_geometric_augmentation(data_pipeline):
            raise ValueError("Offline projected features require synchronized geometric transforms")
        super().__init__(xyz_data_dir, augmentation, standardization, clip,
                         copy.deepcopy(data_pipeline), indices_name)
        self.feature_data_dir, self.xyz_data_dir = Path(feature_data_dir), Path(xyz_data_dir)
        self.aggregation = aggregation
        self.view_count = 1 if aggregation == "max" else int(expected_groups)
        if aggregation == "cluster21" and self.view_count != 21:
            raise ValueError("Certified export uses 21 slots")
        self.xyz_dsm_none_value = xyz_dsm_none_value
        self.tile_names = [Path(n).stem for n in self.ids]
        if len(self.tile_names) != len(set(self.tile_names)):
            raise ValueError("Duplicate CSV patches")
        if patch_list_path:
            keep = json.loads(Path(patch_list_path).read_text())["tiles"]
            if len(keep) != len(set(keep)) or set(keep)-set(self.tile_names):
                raise ValueError("Explicit cohort is not a unique subset of CSV")
            wanted = set(keep)
            indices = [i for i, n in enumerate(self.tile_names) if n in wanted]
            self.ids = [self.ids[i] for i in indices]
            self.tile_names = [self.tile_names[i] for i in indices]
        self.tile_keys = list(self.tile_names)
        root = self.feature_data_dir.parent.parent
        completed = json.loads((root/"DATASET_COMPLETE.json").read_text())
        if not completed["verified"] or completed["rule"]["zero_mask"]:
            raise ValueError("Expected completed unmasked offline dataset")
        if self.feature_data_dir.parent.name != aggregation:
            raise ValueError("Aggregation and feature folder disagree")
        self.receipts = {}
        split = self.feature_data_dir.name
        for name in self.tile_names:
            receipt = json.loads((root/"reports"/split/(name+".json")).read_text())
            path = self.feature_data_dir/(name+".pt")
            size = receipt["max_bytes" if aggregation == "max" else "cluster_bytes"]
            if not receipt["readback_verified"] or path.stat().st_size != size:
                raise ValueError("Offline file not verified: " + str(path))
            self.receipts[name] = receipt
        print(f"[OfflineAblation] mode={aggregation} patches={len(self)} slots={self.view_count}; "
              "real zeros included; no anchor selection or raw-feature reads")

    def __len__(self):
        return len(self.tile_names)

    def _load_features(self, name):
        data = _load_torch(self.feature_data_dir/(name+".pt"))
        receipt = self.receipts[name]
        if data["_fingerprint"] != receipt["fingerprint"] or data["_zero_mask"]:
            raise ValueError("Offline identity/zero-mask mismatch: " + name)
        if data["_num_views"] != receipt["num_views"]:
            raise ValueError("Raw source count mismatch: " + name)
        if self.aggregation == "max":
            views = [{s: data[s] for s in SCALES}]
            exists = torch.tensor([bool(data["_image_available"])])
            for s in SCALES:
                if tuple(views[0][s].shape) != FEATURE_SHAPES[s]:
                    raise ValueError("Incorrect max shape: " + name)
        else:
            groups = data["clusters"]
            exists = data["cluster_valid"].bool()
            sizes = data["cluster_sizes"].long()
            if tuple(exists.shape) != (21,) or not torch.equal(exists, sizes > 0):
                raise ValueError("Incorrect padding flags: " + name)
            if int(sizes.sum()) != receipt["num_views"]:
                raise ValueError("Not every source view is accounted for: " + name)
            for s in SCALES:
                if tuple(groups[s].shape) != (21,) + FEATURE_SHAPES[s]:
                    raise ValueError("Incorrect grouped shape: " + name)
            views = [{s: groups[s][i] for s in SCALES} for i in range(21)]
        return views, exists

    def __getitem__(self, index):
        name = self.tile_names[index]
        image = self.open(str(self.xyz_data_dir/"image"/(name+".tif")), channel_sum=3)
        label = self.open(str(self.xyz_data_dir/"label"/(name+".tif")))
        dsm = self.open(str(self.xyz_data_dir/"dsm"/(name+".tif")), none_value=self.xyz_dsm_none_value)
        if self.augmentation and self.data_pipeline is not None:
            image, label, dsm = seg_augmentation(image=image, mask=[label, dsm], aug=self.data_pipeline)
        views, exists = self._load_features(name)
        return (views, exists, torch.from_numpy(image).float(), torch.from_numpy(dsm).float(),
                torch.from_numpy(label).long(), name)
