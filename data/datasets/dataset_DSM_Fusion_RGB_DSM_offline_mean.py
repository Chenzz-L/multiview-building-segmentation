"""Load the existing all-view mean, with the same patch filter as Concat SAIL."""
from pathlib import Path
import torch
from data.datasets.dataste_support import regist_dataset
from data.datasets.dataset_DSM_Fusion_RGB_DSM_sail_memory import (
    Dataset_DSM_Fusion_RGB_DSM_SAILMemory,SCALES,FEATURE_SHAPES,_load_torch,
)

__all__=['Dataset_DSM_Fusion_RGB_DSM_OfflineMeanMatched',
         'Dataset_DSM_Fusion_RGB_DSM_OfflineMeanNameMatched']


@regist_dataset()
class Dataset_DSM_Fusion_RGB_DSM_OfflineMeanMatched(Dataset_DSM_Fusion_RGB_DSM_SAILMemory):
    def __init__(self,augmentation,mean_feature_data_dir,anchor_data_dir,xyz_data_dir,
                 indices_name='indices',standardization=False,xyz_dsm_none_value=-99999,
                 clip=None,data_pipeline=None,strict=True,missing_anchor_policy='drop',seed=42,
                 disable_tails=True):
        if not disable_tails:raise ValueError('Offline mean dataset cannot enable raw tails')
        self.mean_feature_data_dir=Path(mean_feature_data_dir)
        if not self.mean_feature_data_dir.is_dir():raise FileNotFoundError(self.mean_feature_data_dir)
        super().__init__(augmentation=augmentation,anchor_data_dir=anchor_data_dir,
            raw_feat_data_dir=mean_feature_data_dir,xyz_data_dir=xyz_data_dir,
            indices_name=indices_name,standardization=standardization,xyz_dsm_none_value=xyz_dsm_none_value,
            clip=clip,data_pipeline=data_pipeline,num_anchors=1,tail_count=1,tail_sampling='fixed',
            view_index_path=None,auto_build_index=False,strict=strict,
            missing_anchor_policy=missing_anchor_policy,disable_tails=True,seed=seed)
        self.mean_paths={name:self._find_mean(name) for name in self.tile_names
                         if self._anchor_path(name).is_file()}
        missing=[name for name,path in self.mean_paths.items() if path is None]
        if missing and self.strict:
            raise FileNotFoundError(f'{len(missing)} matched patches lack mean files in {self.mean_feature_data_dir}: {missing[:5]}')

    def _find_mean(self,name):
        for path in (self.mean_feature_data_dir/(name+'.pt'),
                     self.mean_feature_data_dir/name.split('_')[0]/(name+'.pt')):
            if path.is_file():return path
        return None

    def _load_anchors(self,name):
        # The K5 directory defines only the matched split/fallback. Never load
        # its feature tensors. One memory slot holds the entire offline mean.
        if not self._anchor_path(name).is_file():
            if self.missing_anchor_policy=='error':raise FileNotFoundError(self._anchor_path(name))
            return [self._zeros(torch.float16)],[False],[]
        path=self.mean_paths.get(name)
        if path is None:
            if self.strict:raise FileNotFoundError(f'Missing mean feature for {name}')
            return [self._zeros(torch.float16)],[False],[]
        data=_load_torch(path)
        if not isinstance(data,dict) or any(s not in data for s in SCALES):raise ValueError(f'Invalid mean file: {path}')
        feature={s:data[s] for s in SCALES}
        for scale,value in feature.items():
            if not torch.is_tensor(value) or tuple(value.shape)!=FEATURE_SHAPES[scale]:
                raise ValueError(f'Unexpected {scale} shape in {path}')
            if not torch.isfinite(value).all():raise ValueError(f'Nonfinite {scale} feature in {path}')
        return [feature],[True],[]


@regist_dataset()
class Dataset_DSM_Fusion_RGB_DSM_OfflineMeanNameMatched(Dataset_DSM_Fusion_RGB_DSM_SAILMemory):
    """Read only the offline mean, matching P0's name-index patch filtering.

    The original K5 names identify the same eligible patches as P0. Neither
    anchor tensors nor tail tensors are read, and all candidate views enter
    the precomputed mean, including their zero-valued mapping samples.
    """
    def __init__(self, augmentation, mean_feature_data_dir, anchor_name_index_path,
                 raw_feat_data_dir, view_index_path, xyz_data_dir, indices_name='indices',
                 standardization=False, xyz_dsm_none_value=-99999, clip=None,
                 data_pipeline=None, strict=True, missing_anchor_policy='drop', seed=42):
        self.mean_feature_data_dir = Path(mean_feature_data_dir)
        if not self.mean_feature_data_dir.is_dir():
            raise FileNotFoundError(self.mean_feature_data_dir)
        super().__init__(augmentation=augmentation, anchor_data_dir=None,
            anchor_name_index_path=anchor_name_index_path, raw_feat_data_dir=raw_feat_data_dir,
            xyz_data_dir=xyz_data_dir, indices_name=indices_name, standardization=standardization,
            xyz_dsm_none_value=xyz_dsm_none_value, clip=clip, data_pipeline=data_pipeline,
            num_anchors=5, tail_count=1, tail_sampling='fixed', view_index_path=view_index_path,
            auto_build_index=False, strict=strict, missing_anchor_policy=missing_anchor_policy,
            disable_tails=True, seed=seed)
        self.matched_num_anchors = 5
        self.num_anchors = 1
        self.mean_keys = dict(zip(self.tile_names, self.tile_keys))
        self.mean_paths = {name: Dataset_DSM_Fusion_RGB_DSM_OfflineMeanMatched._find_mean(self, name)
                           for name in self.tile_names}
        missing = [name for name, path in self.mean_paths.items() if path is None]
        if missing and self.strict:
            raise FileNotFoundError(f'{len(missing)} matched patches lack mean files: {missing[:5]}')

    def _load_anchors(self, name):
        key = self.mean_keys[name]
        if not self.anchor_name_index[key]:
            return [self._zeros(torch.float16)], [False], []
        path = self.mean_paths[name]
        if path is None:
            if self.strict:
                raise FileNotFoundError('Missing full-pool mean for ' + name)
            return [self._zeros(torch.float16)], [False], []
        data = _load_torch(path)
        if (not isinstance(data, dict) or any(s not in data for s in SCALES)
                or data.get('_num_views') != len(self.raw_index[key])):
            raise ValueError('Mean does not represent this full candidate pool: ' + str(path))
        feature = {s: data[s] for s in SCALES}
        for scale, value in feature.items():
            if (not torch.is_tensor(value) or tuple(value.shape) != FEATURE_SHAPES[scale]
                    or not torch.isfinite(value).all()):
                raise ValueError(f'Invalid {scale} mean feature in {path}')
        return [feature], [True], []
