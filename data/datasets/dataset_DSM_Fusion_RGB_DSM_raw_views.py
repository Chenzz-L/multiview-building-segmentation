"""Anchor-free raw-view sampling for the flat TDOM-query fusion baseline."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import random

import torch

from data.datasets.dataset_Base import BaseDataset
from data.datasets.dataste_support import regist_dataset
from data.datasets.dataset_DSM_Fusion_RGB_DSM_sail_memory import (
    FEATURE_SHAPES, SCALES, _contains_geometric_augmentation, _load_torch, sail_tile_key,
)
from data.transforms.augmentation import seg_augmentation

__all__ = ['Dataset_DSM_Fusion_RGB_DSM_RawViews', 'raw_view_fusion_collate']


@regist_dataset()
class Dataset_DSM_Fusion_RGB_DSM_RawViews(BaseDataset):
    """One unordered view collection: no anchors, exclusion lists or R cache.

    Missing index keys and unreadable selected files are errors. Explicit []
    pools keep their raster sample with invalid padded views; nothing is silently
    filtered according to K5 presence. Optional patch lists freeze a common cohort.
    """
    def __init__(self, augmentation, raw_feat_data_dir, xyz_data_dir,
                 view_index_path, indices_name='indices', standardization=False,
                 xyz_dsm_none_value=-99999, clip=None, data_pipeline=None,
                 view_count=21, view_count_choices=None, view_sampling='random',
                 seed=42, patch_list_path=None, raw_feat_data_roots=None):
        if _contains_geometric_augmentation(data_pipeline):
            raise ValueError('Raw projected features require synchronized geometry; use non-geometric preprocessing')
        super().__init__(xyz_data_dir, augmentation, standardization, clip,
                         copy.deepcopy(data_pipeline), indices_name)
        if view_sampling not in ('random', 'fixed', 'first'):
            raise ValueError('view_sampling must be random, fixed or first')
        if int(view_count) < 1:
            raise ValueError('view_count must be positive')
        self.view_count_choices = None
        if view_count_choices is not None:
            if view_sampling != 'random':
                raise ValueError('view_count_choices requires random sampling')
            self.view_count_choices = tuple(sorted(set(map(int, view_count_choices))))
            if not self.view_count_choices or self.view_count_choices[0] < 1:
                raise ValueError('view_count_choices must be positive')
        self.view_count = max(int(view_count), max(self.view_count_choices or (0,)))
        self.view_sampling, self.seed = view_sampling, int(seed)
        self.raw_feat_data_dir = Path(raw_feat_data_dir)
        self.raw_roots = [self.raw_feat_data_dir] + [Path(p) for p in (raw_feat_data_roots or [])]
        self.xyz_data_dir = Path(xyz_data_dir)
        self.xyz_dsm_none_value = xyz_dsm_none_value
        self.tile_names = [Path(name).stem for name in self.ids]
        self.tile_keys = [sail_tile_key(name) for name in self.tile_names]
        if len(self.tile_keys) != len(set(self.tile_keys)):
            raise ValueError('Duplicate requested patch identities')
        if patch_list_path:
            payload = json.loads(Path(patch_list_path).read_text(encoding='utf-8-sig'))
            keys = payload['tiles']
            if not isinstance(keys, list) or len(keys) != len(set(keys)):
                raise ValueError('Patch list must contain a unique tiles list')
            unknown = set(keys) - set(self.tile_keys)
            if unknown:
                raise ValueError(f'Patch list includes patches absent from CSV: {sorted(unknown)[:3]}')
            wanted = set(keys)
            keep = [i for i, key in enumerate(self.tile_keys) if key in wanted]
            self.ids = [self.ids[i] for i in keep]
            self.tile_names = [self.tile_names[i] for i in keep]
            self.tile_keys = [self.tile_keys[i] for i in keep]
        payload = json.loads(Path(view_index_path).read_text(encoding='utf-8-sig'))
        stored = payload.get('views', payload)
        missing = set(self.tile_keys) - set(stored)
        if missing:
            raise ValueError(f'Raw view index has unknown/missing requested patches: {sorted(missing)[:3]}')
        self.raw_index = {}
        for key in self.tile_keys:
            values = stored[key]
            if not isinstance(values, list) or any(not isinstance(p, str) or not p for p in values):
                raise ValueError(f'Invalid raw view list: {key}')
            names = [Path(p).name for p in values]
            if len(names) != len(set(names)):
                raise ValueError(f'Duplicate raw view identities: {key}')
            if any(not name.endswith('.pt') or sail_tile_key(name) != key for name in names):
                raise ValueError(f'Raw view belongs to a different patch: {key}')
            self.raw_index[key] = sorted(values, key=lambda p: Path(p).name)
        print(f'[RawViewsDataset] patches={len(self.tile_keys)} padded_views={self.view_count} '
              f'sampling={view_sampling} count_choices={self.view_count_choices}; no anchor selection')

    def __len__(self):
        return len(self.tile_keys)

    def _resolve_raw_path(self, value):
        path = Path(value)
        if path.is_absolute():
            return path
        for root in self.raw_roots:
            candidate = root / path
            if candidate.is_file():
                return candidate
        return self.raw_feat_data_dir / path

    def _select_view_paths(self, key):
        paths = self.raw_index[key]
        count = random.choice(self.view_count_choices) if self.view_count_choices else self.view_count
        if self.view_sampling == 'first':
            return paths[:count]
        if self.view_sampling == 'fixed':
            digest = hashlib.blake2b(f'{self.seed}:{key}'.encode(), digest_size=8).digest()
            return random.Random(int.from_bytes(digest, 'little')).sample(paths, min(count, len(paths)))
        return random.sample(paths, min(count, len(paths)))

    def _load_views(self, paths):
        views, exists, dtype = [], [], torch.float16
        for name in paths:
            path = self._resolve_raw_path(name)
            try:
                data = _load_torch(path)
                if not isinstance(data, dict) or any(s not in data for s in SCALES):
                    raise ValueError('Missing FPN scale')
                view = {s: data[s] for s in SCALES}
                if any(not isinstance(v, torch.Tensor) or tuple(v.shape) != FEATURE_SHAPES[s] for s, v in view.items()):
                    raise ValueError('Unexpected raw FPN tensor shape')
                dtype = view['res2'].dtype
                views.append(view)
                exists.append(True)
            except Exception as exc:
                raise RuntimeError(f'Failed to load raw view: {path}') from exc
        while len(views) < self.view_count:
            views.append({s: torch.zeros(shape, dtype=dtype) for s, shape in FEATURE_SHAPES.items()})
            exists.append(False)
        return views, torch.tensor(exists, dtype=torch.bool)

    def __getitem__(self, index):
        name, key = self.tile_names[index], self.tile_keys[index]
        image = self.open(str(self.xyz_data_dir / 'image' / (name + '.tif')), channel_sum=3)
        label = self.open(str(self.xyz_data_dir / 'label' / (name + '.tif')))
        dsm = self.open(str(self.xyz_data_dir / 'dsm' / (name + '.tif')), none_value=self.xyz_dsm_none_value)
        if self.augmentation and self.data_pipeline is not None:
            image, label, dsm = seg_augmentation(image=image, mask=[label, dsm], aug=self.data_pipeline)
        views, exists = self._load_views(self._select_view_paths(key))
        return views, exists, torch.from_numpy(image).float(), torch.from_numpy(dsm).float(), torch.from_numpy(label).long(), name


def raw_view_fusion_collate(batch):
    """Legacy flat-view six-field interface, including names for prediction."""
    if not batch:
        raise ValueError('Cannot collate an empty raw-view batch')
    count = len(batch[0][0])
    if not count or any(len(sample[0]) != count for sample in batch):
        raise ValueError('Raw view collections must have the same positive padded width')
    views = [{s: torch.stack([sample[0][i][s] for sample in batch]) for s in SCALES} for i in range(count)]
    return (views, torch.stack([sample[1] for sample in batch]).bool(),
        torch.stack([sample[2] for sample in batch]), torch.stack([sample[3] for sample in batch]),
        torch.stack([sample[4] for sample in batch]), [sample[5] for sample in batch])
