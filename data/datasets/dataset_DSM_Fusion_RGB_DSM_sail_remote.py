"""Optional validation-only read-through cache for a local feature bridge."""
import json
import os
from pathlib import Path,PurePosixPath
import tempfile
import time
import urllib.request

from filelock import FileLock
import torch
from data.datasets.dataste_support import regist_dataset
from data.datasets.dataset_DSM_Fusion_RGB_DSM_sail_memory import (
    Dataset_DSM_Fusion_RGB_DSM_SAILMemory,SCALES,FEATURE_SHAPES,_load_torch,
)

__all__=['Dataset_DSM_Fusion_RGB_DSM_SAILMemoryRemote']


@regist_dataset()
class Dataset_DSM_Fusion_RGB_DSM_SAILMemoryRemote(Dataset_DSM_Fusion_RGB_DSM_SAILMemory):
    def __init__(self,augmentation,anchor_data_dir,raw_feat_data_dir,xyz_data_dir,
                 indices_name='indices',standardization=False,xyz_dsm_none_value=-99999,
                 clip=None,data_pipeline=None,num_anchors=3,tail_count=8,
                 tail_count_choices=None,tail_sampling='random',tail_mean_data_dir=None,
                 view_index_path=None,auto_build_index=True,xyz_num_length=3,strict=True,
                 missing_anchor_policy='error',return_all_tail_paths=False,disable_tails=False,
                 synchronized_d4_augmentation=False,seed=42,remote_feature_base_url=None,
                 remote_feature_regions=None,remote_feature_cache_dir=None,remote_token_file=None,
                 remote_timeout=60,remote_retries=3):
        super().__init__(augmentation=augmentation,anchor_data_dir=anchor_data_dir,
            raw_feat_data_dir=raw_feat_data_dir,xyz_data_dir=xyz_data_dir,indices_name=indices_name,
            standardization=standardization,xyz_dsm_none_value=xyz_dsm_none_value,clip=clip,
            data_pipeline=data_pipeline,num_anchors=num_anchors,tail_count=tail_count,
            tail_count_choices=tail_count_choices,tail_sampling=tail_sampling,
            tail_mean_data_dir=tail_mean_data_dir,view_index_path=view_index_path,
            auto_build_index=auto_build_index,xyz_num_length=xyz_num_length,strict=strict,
            missing_anchor_policy=missing_anchor_policy,return_all_tail_paths=return_all_tail_paths,
            disable_tails=disable_tails,synchronized_d4_augmentation=synchronized_d4_augmentation,seed=seed)
        if return_all_tail_paths:raise ValueError('Remote validation dataset returns loaded fixed-Q tensors only')
        self.remote_url=str(remote_feature_base_url).rstrip('/')
        self.remote_regions={str(x) for x in (remote_feature_regions or [])}
        self.remote_cache=Path(remote_feature_cache_dir)
        self.remote_token_file=Path(remote_token_file)
        self.remote_timeout=float(remote_timeout);self.remote_retries=int(remote_retries)
        if not self.remote_url.startswith('http://127.0.0.1:'):
            raise ValueError('Bridge must be addressed through the loopback SSH tunnel')
        if not self.remote_token_file.is_file():raise FileNotFoundError(self.remote_token_file)
        self.remote_cache.mkdir(parents=True,exist_ok=True)
        self._validated=set()
        self.remote_download_count=0;self.remote_download_bytes=0

    def _remote_file(self,name):
        relative=PurePosixPath(str(name).replace('\\','/'))
        if relative.is_absolute() or len(relative.parts)!=2 or '..' in relative.parts or relative.suffix!='.pt':
            raise ValueError('Expected region/file.pt relative identity')
        target=self.remote_cache.joinpath(*relative.parts)
        target.parent.mkdir(parents=True,exist_ok=True)
        metadata=target.with_suffix('.pt.meta.json')
        with FileLock(str(target)+'.lock',timeout=self.remote_timeout*self.remote_retries+30):
            if target.is_file() and str(relative) in self._validated:return target
            token=self.remote_token_file.read_text(encoding='utf-8').strip()
            headers={'Authorization':'Bearer '+token}
            url=self.remote_url+'/'+str(relative)
            opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
            remote=None
            last_error=None
            for attempt in range(self.remote_retries):
                temporary=None
                try:
                    with opener.open(urllib.request.Request(url,headers=headers,method='HEAD'),timeout=self.remote_timeout) as response:
                        remote=dict(size=int(response.headers['Content-Length']),mtime_ns=response.headers['X-File-MTime-Ns'])
                    if target.is_file() and metadata.is_file():
                        prior=json.loads(metadata.read_text())
                        if prior==remote and target.stat().st_size==remote['size']:
                            self._validated.add(str(relative));return target
                    with opener.open(urllib.request.Request(url,headers=headers),timeout=self.remote_timeout) as response:
                        assert int(response.headers['Content-Length'])==remote['size']
                        assert response.headers['X-File-MTime-Ns']==remote['mtime_ns']
                        with tempfile.NamedTemporaryFile('wb',delete=False,dir=target.parent,suffix='.partial') as stream:
                            temporary=Path(stream.name);total=0
                            while True:
                                block=response.read(1024*1024)
                                if not block:break
                                stream.write(block);total+=len(block)
                    if total!=remote['size']:raise IOError('Incomplete remote feature transfer')
                    data=_load_torch(temporary)
                    if not isinstance(data,dict) or any(s not in data for s in SCALES):raise ValueError('Missing FPN scale')
                    for s in SCALES:
                        if tuple(data[s].shape)!=FEATURE_SHAPES[s]:raise ValueError('Unexpected feature shape: '+s)
                    del data
                    os.replace(temporary,target);temporary=None
                    metadata.write_text(json.dumps(remote),encoding='utf-8')
                    self.remote_download_count+=1;self.remote_download_bytes+=total
                    self._validated.add(str(relative));return target
                except Exception as error:
                    last_error=error
                    if temporary is not None and temporary.is_file():temporary.unlink()
                    if attempt+1<self.remote_retries:time.sleep(min(2**attempt,4))
            raise RuntimeError('Cannot fetch remote feature '+str(relative)+'; check local service and SSH tunnel') from last_error

    def _load_tails(self,selected_paths):
        tails=[];exists=[];dtype=torch.float16
        for name in selected_paths:
            try:
                parts=PurePosixPath(str(name).replace('\\','/')).parts
                path=self._remote_file(name) if parts and parts[0] in self.remote_regions else self.raw_feat_data_dir/name
                data=_load_torch(path)
                if not isinstance(data,dict) or any(s not in data for s in SCALES):raise ValueError('Missing FPN scale')
                view={s:data[s] for s in SCALES};dtype=view['res2'].dtype
                tails.append(view);exists.append(True)
            except Exception as exc:
                if self.strict:raise RuntimeError('Failed to load validation tail feature: '+str(name)) from exc
                tails.append(self._zeros(dtype));exists.append(False)
        while len(tails)<self.tail_count:
            tails.append(self._zeros(dtype));exists.append(False)
        return tails,exists
