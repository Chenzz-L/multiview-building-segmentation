from typing import TypeVar, Optional, Iterator,Union,List
import math
import os

import numpy as np
from torch import Tensor
import torch
import torch.utils.data as tdata
import torch.distributed as dist


class SequentialDistributedSampler(tdata.distributed.DistributedSampler):
    """
    Distributed Sampler that subsamples indicies sequentially,
    making it easier to collate all results at the end.
    Even though we only use this sampler for eval and predict (no training),
    which means that the model params won't have to be synced (i.e. will not hang
    for synchronization even if varied number of forward passes), we still add extra
    samples to the sampler to make it evenly divisible (like in `DistributedSampler`)
    to make it easy to `gather` or `reduce` resulting tensors at the end of the loop.
    """

    def __init__(self, dataset: tdata.Dataset, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None,drop_last: bool = False) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.drop_last = drop_last
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (len(self.dataset) - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)  # type: ignore[arg-type]
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]

        # subsample
        indices = indices[self.rank * self.num_samples : (self.rank + 1) * self.num_samples]
        return iter(indices)

    def __len__(self):
        return self.num_samples

def padding_to_max(batch):
    out_data=[]
    shape_info = np.array([data[0].shape[2:] for data in batch])
    max_h=np.max(shape_info[:,0])
    max_w=np.max(shape_info[:,1])
    for i in range(len(batch[0])):
        out_data.append(torch.tensor(np.concatenate([np.pad(data[i],((0,0),(0,0),(0,max_h-data[i].shape[2]),(0,max_w-data[i].shape[3])),mode='constant',constant_values=0) for data in batch],axis=0).copy())) 
    return out_data

def dataset_collate_padding(batch):
    """
    对dataloader中输入的batch数据进行重组，将物方的batchsize维度上的数据拼接
    img,xyz_image_data, grid_data, label_data, xyz_label_data
    Args:
        batch:
    Returns:

    """
    out_data=[]
    batch_info = [len(b[0]) for b in batch]
    shape_info = np.array([b.shape[2:] for view_data in batch for b in view_data[0]])
    max_h=np.max(shape_info[:,0])
    max_w=np.max(shape_info[:,1])
    for i in range(len(batch[0])):
        if i==0 or i==3:
            out_data.append(torch.tensor(np.concatenate([np.pad(b,((0,0),(0,0),(0,max_h-b.shape[2]),(0,max_w-b.shape[3])),mode='constant',constant_values=0) for view_data in batch for b in view_data[0]],axis=0).copy())) 
        else:
            out_data.append(torch.tensor(np.concatenate([b[i] for b in batch],axis=0).copy()))
    out_data[2]=torch.tensor((out_data[2]+1)*(shape_info[:,None,None,::-1]/np.array([max_w-1,max_h-1]))-1,dtype=torch.float32).clone().detach()
    return out_data+[batch_info]

# def fusion_collate(batch):
#     """
#     对dataloader中输入的batch数据进行重组，将物方的batchsize维度上的数据拼接
#     Args:
#         batch:

#     Returns:

#     """
#     # x = np.concatenate([b[0] for b in batch],axis=0)
#     # grid = np.concatenate([b[1] for b in batch],axis=0)
#     # y1 = np.concatenate([b[2] for b in batch],axis=0)
#     # y2 = np.concatenate([b[3] for b in batch],axis=0)
#     # batch_info = [b[0].shape[0] for b in batch]
    
#     # x = torch.tensor(x.copy())
#     # grid = torch.tensor(grid.copy())
#     # y1 = torch.tensor(y1.copy())
#     # y2 = torch.tensor(y2.copy())
#     # batch_info = batch_info
#     # return x, grid, y1, y2, batch_info
#     out_data=[]
#     batch_info = [b[0].shape[0] for b in batch]
#     for i in range(len(batch[0])):
#         out_data.append(torch.tensor(np.concatenate([b[i] for b in batch],axis=0).copy()))
#     return out_data+[batch_info]

# def fusion_collate(batch):
#     """
#     对 dataloader 中输入的 batch 数据进行重组。
#     当前 Dataset 已返回 Tensor，因此直接在 Tensor 上拼接即可。
#     """
#     out_data = []
#     batch_info = [b[0].shape[0] for b in batch]  # 每个样本的子 patch 数

#     for i in range(len(batch[0])):
#         # 如果是 tensor，则用 torch.cat，否则退回 numpy 逻辑
#         if isinstance(batch[0][i], torch.Tensor):
#             out_data.append(torch.cat([b[i] for b in batch], dim=0))
#         else:
#             out_data.append(
#                 torch.tensor(np.concatenate([b[i] for b in batch], axis=0).copy())
#             )

#     return out_data + [batch_info]
# def fusion_collate(batch):
#     batch_info = []
#     for b in batch:
#         feat = b[0]
#         key = next(iter(feat))
#         batch_info.append(feat[key].shape[0])  # N_i

#     out_data = []
#     num_fields = len(batch[0])

#     for i in range(num_fields):
#         first_item = batch[0][i]

#         # ---------- 1. 像方特征 dict ----------
#         if i == 0 and isinstance(first_item, dict):
#             merged_dict = {}
#             for key in first_item.keys():
#                 merged_dict[key] = torch.cat(
#                     [b[0][key] for b in batch], dim=0
#                 )  # [sum N_i, C, H, W]
#             out_data.append(merged_dict)

#         # ---------- 2. image / dsm / label ----------
#         elif isinstance(first_item, torch.Tensor):
#             # 这里**一律 stack**
#             out_data.append(
#                 torch.stack([b[i] for b in batch], dim=0)
#             )  # [B, C, H, W]

#         # ---------- 3. 其它 ----------
#         else:
#             out_data.append([b[i] for b in batch])

#     return out_data + [batch_info]

def fusion_collate(batch):
    """
    batch: List[
        (
            feat_dict: Dict[str, Tensor[C,H,W]],
            image: Tensor[3,H,W],
            dsm:   Tensor[1,H,W],
            label: Tensor[H,W] or Tensor[1,H,W]
        )
    ]
    """

    # -------- feature --------
    feat = {}
    feat_keys = batch[0][0].keys()

    for k in feat_keys:
        feat[k] = torch.stack(
            [b[0][k] for b in batch], dim=0
        )  # [B, C, H, W]

    # -------- image / dsm / label --------
    image = torch.stack([b[1] for b in batch], dim=0)  # [B,3,H,W]
    dsm   = torch.stack([b[2] for b in batch], dim=0)  # [B,1,H,W]
    label = torch.stack([b[3] for b in batch], dim=0)

    # return feat, image, dsm, label,batch[0][4]
    return feat, image, dsm, label


def cam_fusion_collate(batch):
    """
    Collate function for Dataset_DSM_Fusion_RGB_DSM_CamFeature.

    batch: List[
        (
            cam_features: list[5 * dict{res2/3/4/5: Tensor[C,H,W]}],
            cam_exist:    list[5 * bool],
            image:        Tensor[3,H,W],
            dsm:          Tensor[1,H,W],
            label:        Tensor[H,W]
        )
    ]

    Returns:
        cam_features_batched: list[5], 每个元素 dict{res2/3/4/5: Tensor[B,C,H,W]}
        cam_exist_batched:    Tensor[B, 5] bool
        image:  Tensor[B,3,H,W]
        dsm:    Tensor[B,1,H,W]
        label:  Tensor[B,H,W]
    """
    num_cameras = len(batch[0][0])  # 5
    scale_keys = list(batch[0][0][0].keys())  # ["res2","res3","res4","res5"]

    # ---- per-camera features ----
    cam_features_batched = []
    for cam_idx in range(num_cameras):
        feat_dict = {}
        for k in scale_keys:
            feat_dict[k] = torch.stack(
                [b[0][cam_idx][k] for b in batch], dim=0
            )  # [B, C, H, W]
        cam_features_batched.append(feat_dict)

    # ---- cam exist mask ----
    cam_exist_batched = torch.tensor(
        [[b[1][cam_idx] for cam_idx in range(num_cameras)] for b in batch],
        dtype=torch.bool
    )  # [B, 5]

    # ---- image / dsm / label ----
    image = torch.stack([b[2] for b in batch], dim=0)  # [B,3,H,W]
    dsm   = torch.stack([b[3] for b in batch], dim=0)  # [B,1,H,W]
    label = torch.stack([b[4] for b in batch], dim=0)

    return cam_features_batched, cam_exist_batched, image, dsm, label


def sail_fusion_collate(batch):
    """Batch separate SAIL anchors and tails without mixing their roles.

    Dataset items are ``(anchors, anchor_exist, tails, tail_exist, image, dsm,
    label, name)``.  Both feature collections retain the existing list-of-dict
    format expected by ``SAILImageMemoryFusion``.
    """
    if not batch:
        raise ValueError("Cannot collate an empty SAIL batch")

    def stack_view_group(item_index: int):
        num_views = len(batch[0][item_index])
        if num_views == 0:
            raise ValueError("SAIL anchors/tails must be padded to a fixed positive length")
        scale_keys = list(batch[0][item_index][0].keys())
        output = []
        for view_index in range(num_views):
            output.append({
                scale: torch.stack(
                    [sample[item_index][view_index][scale] for sample in batch], dim=0
                )
                for scale in scale_keys
            })
        return output

    anchor_features = stack_view_group(0)
    anchor_exist = torch.stack([sample[1] for sample in batch], dim=0).bool()
    tail_features = stack_view_group(2)
    tail_exist = torch.stack([sample[3] for sample in batch], dim=0).bool()
    image = torch.stack([sample[4] for sample in batch], dim=0)
    dsm = torch.stack([sample[5] for sample in batch], dim=0)
    label = torch.stack([sample[6] for sample in batch], dim=0)
    names = [sample[7] for sample in batch]
    return (
        anchor_features, anchor_exist, tail_features, tail_exist,
        image, dsm, label, names,
    )


def sail_all_tail_collate(batch):
    """Collate anchors while keeping all tail paths on disk for streaming.

    All-tail SAIL prediction currently runs with ``batch_size=1``.  Retaining
    a list per sample also makes the contract explicit if a caller wants to
    add batching later.
    """
    if not batch:
        raise ValueError("Cannot collate an empty SAIL all-tail batch")

    num_anchors = len(batch[0][0])
    scale_keys = list(batch[0][0][0].keys())
    anchors = []
    for anchor_index in range(num_anchors):
        anchors.append({
            scale: torch.stack(
                [sample[0][anchor_index][scale] for sample in batch], dim=0
            )
            for scale in scale_keys
        })
    anchor_exist = torch.stack([sample[1] for sample in batch], dim=0).bool()
    all_tail_paths = [sample[2] for sample in batch]
    image = torch.stack([sample[3] for sample in batch], dim=0)
    dsm = torch.stack([sample[4] for sample in batch], dim=0)
    label = torch.stack([sample[5] for sample in batch], dim=0)
    names = [sample[6] for sample in batch]
    return anchors, anchor_exist, all_tail_paths, image, dsm, label, names


def cam_fusion_inference_collate(batch):
    """
    Collate function for Inference_Dataset_DSM_Fusion_RGB_DSM_CamFeature.
    返回 6 个项，最后一项是 xyz_name 列表。
    """
    num_cameras = len(batch[0][0])
    scale_keys = list(batch[0][0][0].keys())

    # ---- per-camera features ----
    cam_features_batched = []
    for cam_idx in range(num_cameras):
        feat_dict = {}
        for k in scale_keys:
            feat_dict[k] = torch.stack(
                [b[0][cam_idx][k] for b in batch], dim=0
            )
        cam_features_batched.append(feat_dict)

    # ---- cam exist mask ----
    cam_exist_batched = torch.tensor(
        [[b[1][cam_idx] for cam_idx in range(num_cameras)] for b in batch],
        dtype=torch.bool
    )

    # ---- image / dsm / label ----
    image = torch.stack([b[2] for b in batch], dim=0)
    dsm   = torch.stack([b[3] for b in batch], dim=0)
    label = torch.stack([b[4] for b in batch], dim=0)
    
    # ---- names ----
    names = [b[5] for b in batch]

    return cam_features_batched, cam_exist_batched, image, dsm, label, names




def viewfusion_collate(batch):
    """
    Collate function for dataset with fields:
    (img, label, view_index, mask)
    """
    imgs, labels, view_indices, masks = zip(*batch)  # 解包

    # img: numpy or tensor -> float32
    if isinstance(imgs[0], np.ndarray):
        imgs = torch.from_numpy(np.stack(imgs, axis=0)).float()
    else:
        imgs = torch.stack([x.float() for x in imgs], dim=0)

    if isinstance(labels[0], np.ndarray):
        labels = torch.from_numpy(np.stack(labels, axis=0)).float()
    else:
        labels = torch.tensor(labels, dtype=torch.float)

    # view_index: int -> long
    view_indices = torch.tensor(view_indices, dtype=torch.long)

    # mask: bool
    if isinstance(masks[0], np.ndarray):
        masks = torch.from_numpy(np.stack(masks, axis=0)).bool()
    else:
        masks = torch.tensor(masks, dtype=torch.bool)

    return imgs, labels, view_indices, masks

def fusion_inference_collate(batch):
    for key, value in batch[0].items():
        if key in ["image", "label", "points_pair", "batch_info", 'grid']:
            batch[0][key] = [torch.tensor(x) for x in value]
        elif key in ["xyz_label", "xyz_image"]:
            batch[0][key] = torch.tensor(value)
    return batch[0]

def normal_inference_collate(batch):
    for key, value in batch[0].items():
        if key in ["image", "label"]:
            batch[0][key] = torch.tensor(value)
    return batch[0]

def expand_datalist_2_BCHW(data_list:Union[List[np.ndarray],List[Tensor]]):
    for i,data in enumerate(data_list):
        data_list[i]=expand_data_2_BCHW(data)
    return data_list

def expand_data_2_BCHW(data:Union[np.ndarray,Tensor]):
    if data.ndim < 4:
        if isinstance(data,np.ndarray):
            data = np.expand_dims(data, axis=list(range(4 - data.ndim)))
        else:
            data.unsqueeze_(dim=0)
            if data.ndim < 4:
                data.unsqueeze_(dim=0)
    return data

def get_xyzname_from_pairfile(image_file,pair_data):
    image_name = os.path.splitext(os.path.basename(image_file))[0]
    xyz_num = image_name.split("_")[-3]
    num = image_name.split("_")[-2:]
    num = "_".join(num)
    xyz_suffix=pair_data[xyz_num] + "_" + num + ".tif"
    return xyz_suffix


def rgb_dsm_collate(batch):
    """
    专门用于 RGB + DSM + Mask2Former Target 的 collate 函数
    Batch 结构: [(img, dsm, target), (img, dsm, target), ...]
    """
    # 1. 解压 batch
    # imgs: tuple of (3, H, W) tensors
    # dsms: tuple of (1, H, W) tensors
    # targets: tuple of dicts
    imgs, dsms, targets = zip(*batch)
    
    # 2. 堆叠图像数据 (Stack) -> (B, C, H, W)
    # 前提是 transforms 里已经 Resize 到相同尺寸，或者使用了 Pad
    imgs = torch.stack(imgs, dim=0)
    dsms = torch.stack(dsms, dim=0)
    
    # 3. Targets 保持为列表 (List[Dict])
    # Mask2Former/DETR 需要 list 格式，不能 stack
    return imgs, dsms, list(targets)
