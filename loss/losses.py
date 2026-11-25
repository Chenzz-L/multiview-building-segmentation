import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional,List
from scipy.optimize import linear_sum_assignment
import torch.distributed as dist
from torch.amp import autocast
import torchvision
from torch import Tensor
from detectron2.utils.comm import get_world_size
from detectron2.projects.point_rend.point_features import (
    get_uncertain_point_coords_with_randomness,
    point_sample,
)

__all__ = [
    'FocalLoss',
    'LabelSmoothSoftmaxCE',
    'SigmoidDiceLoss',
    'SpatialDiceLoss',
    'LovaszSoftmaxLoss',
    'convert_to_one_hot',
    'BCEWithLogitsLoss',
]
eps = 1e-6


def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction == 'mean' else loss.sum() if reduction == 'sum' else loss


class FocalLoss(nn.Module):
    r""" Lin T Y, Goyal P, Girshick R, et al. <Focal loss for dense object detection>
    modified from https://github.com/CoinCheung/pytorch-loss
    Usage is same as nn.BCEWithLogits:
    """
    def __init__(self, alpha=0.25, gamma=2, reduction='mean', loss_weight=1.0):
        super(FocalLoss, self).__init__()
        assert 0 <= alpha <= 1
        assert reduction in (None, 'none', 'mean', 'sum')
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, inputs, targets):
        r""" Usage is same as nn.BCEWithLogits:
            # >>> criteria = FocalLossV1()
            # >>> inputs = torch.randn(8, 19, 384, 384)
            # >>> targets = torch.randint(0, 2, (8, 19, 384, 384)).float()
            # >>> loss = criteria(inputs, targets)
        """
        # inputs -> [batch_size, channel, height, width]
        # targets -> [batch_size, channel, height, width]
        if inputs.shape != targets.shape:  # adaptively convert targets to one_hot format
            if len(targets.shape) == len(inputs.shape) - 1:  # if targets -> [batch_size, height, width]
                targets = convert_to_one_hot(targets, min_length=inputs.shape[1])
            else:
                raise Exception('shape of inputs is not adaptive with shape of targets')
        probs = torch.sigmoid(inputs)
        coeff = torch.abs(targets - probs).pow(self.gamma).neg()
        log_probs = torch.where(inputs >= 0,
                                F.softplus(inputs, -1, 50),
                                inputs - F.softplus(inputs, 1, 50))
        log_1_probs = torch.where(inputs >= 0,
                                  -inputs + F.softplus(inputs, -1, 50),
                                  -F.softplus(inputs, 1, 50))
        loss = targets * self.alpha * log_probs + (1. - targets) * (1. - self.alpha) * log_1_probs
        loss = loss * coeff
        return (reduce_loss(loss, self.reduction)) * self.loss_weight


class LabelSmoothSoftmaxCE(nn.Module):
    r"""
    Same usage method as nn.CrossEntropyLoss
    modified from https://github.com/CoinCheung/pytorch-loss
    """

    def __init__(self, smoothing=0.1, reduction='mean', ignore_index=-100):
        super(LabelSmoothSoftmaxCE, self).__init__()
        assert 0 <= smoothing < 1
        assert reduction in (None, 'none', 'mean', 'sum')
        self.smoothing = smoothing
        self.reduction = reduction
        self.lb_ignore = ignore_index
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """ Same usage method as nn.CrossEntropyLoss:
            # >>> criteria = LabelSmoothSoftmaxCEV1()
            # >>> logits = torch.randn(8, 19, 384, 384) # nchw, float/half
            # >>> lbs = torch.randint(0, 19, (8, 384, 384)) # nhw, int64_t
            # >>> loss = criteria(logits, lbs)
        """
        # inputs -> [batch_size, channel, height, width]
        # targets -> [batch_size, height, width]
        inputs = inputs.float()  # use fp32 to avoid nan
        with torch.no_grad():  # overcome ignored label
            num_classes = inputs.size(1)
            targets = targets.clone().detach()
            ignore = targets.eq(self.lb_ignore)
            n_valid = ignore.eq(0).sum()
            targets[ignore] = 0
            lb_pos, lb_neg = 1. - self.smoothing, self.smoothing / num_classes
            lb_one_hot = torch.empty_like(inputs).fill_(
                lb_neg).scatter_(1, targets.unsqueeze(1), lb_pos).detach()

        logs = self.log_softmax(inputs)
        loss = -torch.sum(logs * lb_one_hot, dim=1)
        loss[ignore] = 0

        if self.reduction == 'mean':
            loss = loss.sum() / n_valid
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss


class SigmoidDiceLoss(nn.Module):
    r""" Milletari F, Navab N, Ahmadi S A.
    <V-net: Fully convolutional neural networks for volumetric medical image segmentation>
    modified from https://github.com/CoinCheung/pytorch-loss
    soft-dice loss, useful in binary segmentation
    Usage is same as nn.BCEWithLogits:
    """
    def __init__(self,  loss_weight=1.0, exponent=2, smooth=1, reduction='mean'):
        super(SigmoidDiceLoss, self).__init__()
        assert reduction in (None, 'none', 'mean', 'sum')
        self.reduction = reduction
        self.exponent = exponent
        self.smooth = smooth
        self.loss_weight = loss_weight

    def forward(self, inputs, targets):
        r""" Usage is same as nn.BCEWithLogits:
        Args:
            inputs: tensor [batch_size, channel, height, width]
            targets: tensor [batch_size, channel, height, width]
        :return
            loss: tensor of shape(1, )
        """
        assert inputs.shape == targets.shape
        probs = torch.sigmoid(inputs)
        num = torch.sum(torch.mul(probs, targets), dim=1) * 2 + self.smooth
        den = torch.sum(probs.pow(self.exponent) + targets.pow(self.exponent), dim=1) + self.smooth
        loss = 1. - num / den
        return reduce_loss(loss, self.reduction) * self.loss_weight


class SpatialDiceLoss(nn.Module):
    """Soft Dice loss computed over the spatial dimensions."""

    def __init__(self, loss_weight=1.0, exponent=2, smooth=1.0, reduction='mean'):
        super().__init__()
        assert reduction in (None, 'none', 'mean', 'sum')
        self.loss_weight = loss_weight
        self.exponent = exponent
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, inputs, targets):
        if inputs.shape != targets.shape:
            if targets.ndim == inputs.ndim - 1:
                targets = targets.unsqueeze(1)
            else:
                raise ValueError(
                    f"SpatialDiceLoss shape mismatch: {inputs.shape} vs {targets.shape}"
                )

        targets = targets.to(dtype=inputs.dtype)
        probs = torch.sigmoid(inputs.float())
        targets = targets.float()
        spatial_dims = tuple(range(2, inputs.ndim))

        intersection = (probs * targets).sum(dim=spatial_dims)
        denominator = (
            probs.pow(self.exponent).sum(dim=spatial_dims)
            + targets.pow(self.exponent).sum(dim=spatial_dims)
        )
        loss = 1.0 - (
            2.0 * intersection + self.smooth
        ) / (
            denominator + self.smooth
        )
        return reduce_loss(loss, self.reduction) * self.loss_weight


class LovaszSoftmaxLoss(nn.Module):
    """Berman M, Triki A R, Blaschko M B.
    <The lovász-softmax loss: A tractable surrogate for the optimization
    of the intersection-over-union measure in neural networks>
    modified from https://github.com/CoinCheung/pytorch-loss
    Same usage method as nn.CrossEntropyLoss
    """
    def __init__(self, reduction='mean', ignore_index=-100, get_errs=False):
        super(LovaszSoftmaxLoss, self).__init__()
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.get_errs = get_errs

    def forward(self, inputs, targets):
        """ Same usage method as nn.CrossEntropyLoss:
            # >>> criteria = LovaszSoftmaxLoss()
            # >>> inputs = torch.randn(8, 19, 384, 384) # nchw, float/half
            # >>> targets = torch.randint(0, 19, (8, 384, 384)) # nhw, int64_t
            # >>> loss = criteria(inputs, targets)
        """
        # inputs -> [batch_size, channel, height, width]
        # targets -> [batch_size, height, width]
        # overcome ignored label
        num_classes = inputs.size(1)
        inputs = inputs.transpose(0, 1).reshape(num_classes, -1).float()  # use fp32 to avoid nan
        targets = targets.view(-1)

        idx = targets.ne(self.ignore_index).nonzero(as_tuple=False).squeeze()
        probs = inputs.softmax(dim=0)[:, idx]

        targets = targets[idx]
        lb_one_hot = torch.zeros_like(probs).scatter_(
                0, targets.unsqueeze(0), 1).detach()

        errs = (lb_one_hot - probs).abs()
        errs_sort, errs_order = torch.sort(errs, dim=1, descending=True)
        n_samples = errs.size(1)

        # lovasz extension grad
        with torch.no_grad():
            lb_one_hot_sort = torch.cat([
                lb_one_hot[i, ord].unsqueeze(0)
                for i, ord in enumerate(errs_order)], dim=0)
            n_pos = lb_one_hot_sort.sum(dim=1, keepdim=True)
            inter = n_pos - lb_one_hot_sort.cumsum(dim=1)
            union = n_pos + (1. - lb_one_hot_sort).cumsum(dim=1)
            jacc = 1. - inter / union
            if n_samples > 1:
                jacc[:, 1:] = jacc[:, 1:] - jacc[:, :-1]

        losses = torch.einsum('ab,ab->a', errs_sort, jacc)
        if self.get_errs:
            return reduce_loss(losses, self.reduction), errs
        else:
            return reduce_loss(losses, self.reduction)


@torch.no_grad()
def convert_to_one_hot(x, min_length, ignore_idx=-1):
    """
    modified from https://github.com/CoinCheung/pytorch-loss
    encode input x into one hot
    inputs:
        x: tensor of shape (N, ...) with type long
        min_length: minimum length of one hot code, this should be larger than max value in x
        ignore_idx: the index in x that should be ignored, default is 255

    return:
        tensor of shape (N, min_length, ...) with type float
    """
    device = x.device
    # compute output shape
    size = list(x.size())
    size.insert(1, min_length)
    assert x[x != ignore_idx].max() < min_length, "min_length should larger than max value in x"

    if ignore_idx < 0:
        out = torch.zeros(size, device=device).scatter_(1, x.unsqueeze(1), 1)
    else:
        # overcome ignore index
        x = x.clone().detach()
        ignore = x == ignore_idx
        x[ignore] = 0
        out = torch.zeros(size, device=device).scatter_(1, x.unsqueeze(1), 1)
        ignore = ignore.nonzero(as_tuple=False)
        _, M = ignore.size()
        a, *b = ignore.chunk(M, dim=1)
        out[[a, torch.arange(min_length), *b]] = 0
    return out

class BCEWithLogitsLoss(nn.Module):
    def __init__(self, loss_weight=1.0, pos_weight=None, reduction='mean'):
        """
        Args:
            loss_weight (float): 该 Loss 的权重，用于多 Loss 组合 (如 BCE + Dice)。
            pos_weight (float or List[float]): 正样本权重。
            reduction (str): 'mean', 'sum' or 'none'.
        """
        super(BCEWithLogitsLoss, self).__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction
        
        # 处理正样本权重 (用于处理类别不平衡)
        if pos_weight is not None:
            if isinstance(pos_weight, (float, int)):
                pos_weight = torch.tensor([pos_weight])
            else:
                pos_weight = torch.tensor(pos_weight)
            # 注册为 buffer，这样它会自动随模型移动到 GPU 上
            self.register_buffer('pos_weight', pos_weight)

    def forward(self, pred, target):
        """
        Args:
            pred: 模型输出的 logits (未经过 Sigmoid)
            target: 真实标签
        """
        # 1. 维度对齐
        if pred.ndim != target.ndim:
            if target.ndim == 3: 
                target = target.unsqueeze(1) 

        # 2. 类型对齐
        if target.dtype != pred.dtype:
            target = target.type_as(pred)

        # 3. 计算 Loss (使用 functional API 确保 pos_weight 在正确的设备上)
        weight = getattr(self, 'pos_weight', None)
        if weight is not None:
            weight = weight.to(pred.device)
            
        loss = F.binary_cross_entropy_with_logits(
            pred, target, 
            pos_weight=weight, 
            reduction=self.reduction
        )
        
        return loss * self.loss_weight


# class Mask2FormerSetCriterion(nn.Module):
#     """
#     标准 Mask2Former / DETR 风格的集合损失
#     适合：实例数量较少的 UAV 建筑实例分割
#     """

#     def __init__(
#         self,
#         num_classes,
#         matcher,
#         weight_dict=None,
#         eos_coef=0.05,
#     ):
#         super().__init__()

#         self.num_classes = num_classes
#         self.matcher = matcher

#         self.weight_dict = weight_dict or {
#             "loss_ce": 1.0,
#             "loss_mask": 5.0,
#             "loss_dice": 5.0,
#         }

#         # no-object 权重
#         empty_weight = torch.ones(num_classes + 1)
#         empty_weight[-1] = eos_coef
#         self.register_buffer("empty_weight", empty_weight)

#     # ------------------------------------------------
#     # forward
#     # ------------------------------------------------
#     def forward(self, outputs, targets):
#         """
#         outputs:
#             pred_logits: [B, Q, C+1]
#             pred_masks:  [B, Q, H, W]

#         targets:
#             list[dict]:
#                 labels: [Ni]
#                 masks:  [Ni, H, W]
#         """
#         # 1. Hungarian matching
#         with torch.no_grad():
#             indices = self.matcher(outputs, targets)

#         # 2. 统计实例数（num_masks）
#         num_masks = sum(len(t["labels"]) for t in targets)
#         num_masks = torch.as_tensor(
#             [num_masks],
#             dtype=torch.float,
#             device=outputs["pred_logits"].device,
#         )

#         if dist.is_available() and dist.is_initialized():
#             dist.all_reduce(num_masks)
#             num_masks = num_masks / dist.get_world_size()

#         num_masks = torch.clamp(num_masks, min=1).item()

#         # 3. 计算各项 loss
#         losses = {}
#         losses.update(self.loss_labels(outputs, targets, indices, num_masks))
#         losses.update(self.loss_masks(outputs, targets, indices, num_masks))

#         return losses

#     # ------------------------------------------------
#     # 分类 loss（labels）
#     # ------------------------------------------------
#     def loss_labels(self, outputs, targets, indices, num_masks):
#         """
#         分类 loss：实例级，按 num_masks 归一化
#         """
#         src_logits = outputs["pred_logits"]  # [B, Q, C+1]

#         idx = self._get_src_permutation_idx(indices)

#         target_classes = torch.full(
#             src_logits.shape[:2],
#             self.num_classes,  # no-object
#             dtype=torch.int64,
#             device=src_logits.device,
#         )

#         target_classes[idx] = torch.cat(
#             [t["labels"][J] for t, (_, J) in zip(targets, indices)]
#         )

#         loss_ce = F.cross_entropy(
#             src_logits.transpose(1, 2),
#             target_classes,
#             weight=self.empty_weight,
#             reduction="sum",
#         ) / num_masks

#         return {"loss_ce": loss_ce}

#     # ------------------------------------------------
#     # mask loss（BCE + Dice）
#     # ------------------------------------------------
#     def loss_masks(self, outputs, targets, indices, num_masks):
#         src_idx = self._get_src_permutation_idx(indices)
#         src_masks = outputs["pred_masks"][src_idx]  # [N, H_pred, W_pred]

#         if src_masks.numel() == 0:
#             zero = outputs["pred_masks"].sum() * 0.0
#             return {
#                 "loss_mask": zero,
#                 "loss_dice": zero,
#             }

#         target_masks = torch.cat(
#             [t["masks"][J] for t, (_, J) in zip(targets, indices)],
#             dim=0,
#         ).to(src_masks.device)  # [N, H_gt, W_gt]

#         # ------------------------------------------------
#         # 【关键修复】预测 mask resize 到 GT 尺寸
#         # ------------------------------------------------
#         src_masks = F.interpolate(
#             src_masks[:, None],                 # [N, 1, H_pred, W_pred]
#             size=target_masks.shape[-2:],       # (H_gt, W_gt)
#             mode="bilinear",
#             align_corners=False,
#         ).squeeze(1)                            # [N, H_gt, W_gt]

#         # Dice（实例级）
#         loss_dice = self.dice_loss(src_masks, target_masks, num_masks)

#         # BCE（像素级）
#         loss_mask = F.binary_cross_entropy_with_logits(
#             src_masks,
#             target_masks.float(),
#             reduction="mean",
#         )

#         return {
#             "loss_mask": loss_mask,
#             "loss_dice": loss_dice,
#         }

#     # ------------------------------------------------
#     # Dice loss（实例级）
#     # ------------------------------------------------
#     def dice_loss(self, inputs, targets, num_masks):
#         inputs = inputs.sigmoid()
#         inputs = inputs.flatten(1)
#         targets = targets.flatten(1)

#         numerator = 2 * (inputs * targets).sum(1)
#         denominator = inputs.sum(1) + targets.sum(1)

#         loss = 1 - (numerator + 1) / (denominator + 1)
#         return loss.sum() / num_masks

#     # ------------------------------------------------
#     # index helper
#     # ------------------------------------------------
#     @staticmethod
#     def _get_src_permutation_idx(indices):
#         batch_idx = torch.cat(
#             [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
#         )
#         src_idx = torch.cat([src for (src, _) in indices])
#         return batch_idx, src_idx


# class HungarianMatcher(nn.Module):
#     """
#     匈牙利匹配器：计算 Cost Matrix 并寻找最优匹配
#     增加了 NaN/Inf 检查和 Float32 强制转换，防止训练崩溃。
#     """
#     def __init__(self, cost_class: float = 0.5, cost_mask: float = 1, cost_dice: float = 2):
#         super().__init__()
#         self.cost_class = cost_class
#         self.cost_mask = cost_mask
#         self.cost_dice = cost_dice

#     @torch.no_grad()
#     def forward(self, outputs, targets):
#         bs, num_queries = outputs["pred_logits"].shape[:2]

#         # 1. 扁平化 Predictions (强制转为 float32 防止 FP16 溢出)
#         # [BS * NQ, Num_Classes]
#         out_prob = outputs["pred_logits"].flatten(0, 1).float().softmax(-1)
#         # [BS * NQ, H, W]
#         out_mask = outputs["pred_masks"].flatten(0, 1).float()

#         # 2. 扁平化 Targets
#         tgt_ids = torch.cat([v["labels"] for v in targets])
#         tgt_masks = torch.cat([v["masks"] for v in targets])

#         # ==========================================================
#         # 3. 计算 Cost (全部在 Float32 下进行)
#         # ==========================================================
        
#         # A. 统一分辨率
#         # 将 Target 下采样到 Prediction 的大小
#         tgt_masks = F.interpolate(
#             tgt_masks[:, None].float(), 
#             size=out_mask.shape[-2:], 
#             mode="nearest"
#         ).squeeze(1)

#         # B. 维度准备
#         out_mask = out_mask.flatten(1)       # [N, HW]
#         tgt_masks = tgt_masks.flatten(1)     # [M, HW]
        
#         out_prob_mask = out_mask.sigmoid()   # 预测概率

#         # C. 维度扩展 -> [N, 1, HW] 和 [1, M, HW]
#         p = out_prob_mask.unsqueeze(1) 
#         t = tgt_masks.unsqueeze(0)

#         # --- Cost 1: Classification ---
#         # out_prob: [N, NumClasses] -> 取出对应 tgt_ids 列 -> [N, M]
#         cost_class = -out_prob[:, tgt_ids]

#         # --- Cost 2: Mask (L1 / BCE 近似) ---
#         # 简化版 Cost：概率差异
#         # [N, M]
#         cost_mask = -(p * t + (1 - p) * (1 - t)).mean(-1)

#         # --- Cost 3: Dice ---
#         # Dice = 2 * (P ∩ T) / (P + T)
#         # 加上 1.0 平滑防止除零
#         numerator = 2 * (p * t).sum(-1)
#         denominator = p.sum(-1) + t.sum(-1)
#         cost_dice = 1 - (numerator + 1) / (denominator + 1)

#         # 4. 总 Cost Matrix
#         # [BS * NQ, Total_Targets]
#         C = self.cost_mask * cost_mask + self.cost_class * cost_class + self.cost_dice * cost_dice
        
#         # 重塑: [BS, NQ, Total_Targets]
#         C = C.view(bs, num_queries, -1).cpu()

#         # ==========================================================
#         # 5. 安全性检查 (关键修复)
#         # ==========================================================
#         # 检查是否有 NaN 或 Inf
#         if torch.isnan(C).any() or torch.isinf(C).any():
#             # print("[Warning] Cost matrix contains NaN/Inf! Replacing with zeros for stability.")
#             # 将 NaN/Inf 替换为 0 或大数值，防止 scipy 报错导致训练中断
#             C = torch.nan_to_num(C, nan=0.0, posinf=100000.0, neginf=-100000.0)

#         # 6. 匈牙利匹配求解
#         sizes = [len(v["labels"]) for v in targets]
        
#         # split 将最后一维 (Total_Targets) 切分成每个 Batch 的 targets 数量
#         # C.split 返回 tuple, 元素 k 的 shape 为 [BS, NQ, sizes[k]]
#         indices = []
#         for i, c_split in enumerate(C.split(sizes, -1)):
#             # c_split 是第 i 张图的 targets 对应的 Cost 列
#             # 但它包含了所有 batch 的 queries (行)
#             # 我们只取第 i 个 batch 的 queries -> c_split[i]
#             # c_i shape: [NQ, sizes[i]]
#             c_i = c_split[i]
            
#             # 如果该图没有目标 (sizes[i] == 0)，直接返回空匹配
#             if c_i.shape[1] == 0:
#                 indices.append((torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.int64)))
#             else:
#                 # Scipy 运算
#                 ind = linear_sum_assignment(c_i.numpy())
#                 indices.append(ind)
        
#         return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


# def mask2former_dice_loss(inputs, targets, num_masks):
#     """
#     专为 Mask2Former 设计的 Dice Loss
#     Args:
#         inputs: [N, H, W] 预测的 logits (未经过 sigmoid)
#         targets: [N, H, W] 真实的 binary masks (0或1)
#         num_masks: 归一化分母 (总目标数)
#     """
#     # 1. Sigmoid 激活
#     inputs = inputs.sigmoid()
    
#     # 2. 展平 (N, HW)
#     inputs = inputs.flatten(1)
#     targets = targets.flatten(1)
    
#     # 3. 计算 Dice
#     # 平滑项 +1 防止除以 0
#     numerator = 2 * (inputs * targets).sum(1)
#     denominator = inputs.sum(-1) + targets.sum(-1)
#     loss = 1 - (numerator + 1) / (denominator + 1)
#     fg_weight = targets.sum(1).clamp(min=1.0)
#     loss = loss / fg_weight.sqrt()
#     # 4. 归一化 (除以总目标数)
#     return loss.sum() / num_masks





def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


dice_loss_jit = torch.jit.script(
    dice_loss
)  # type: torch.jit.ScriptModule


def sigmoid_ce_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    return loss.mean(1).sum() / num_masks


sigmoid_ce_loss_jit = torch.jit.script(
    sigmoid_ce_loss
)  # type: torch.jit.ScriptModule


def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))

def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    # TODO make this more general
    if tensor_list[0].ndim == 3:
        if torchvision._is_tracing():
            # nested_tensor_from_tensor_list() does not export well to ONNX
            # call _onnx_nested_tensor_from_tensor_list() instead
            return _onnx_nested_tensor_from_tensor_list(tensor_list)

        # TODO make it support different-sized images
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], : img.shape[2]] = False
    else:
        raise ValueError("not supported")
    return NestedTensor(tensor, mask)


# _onnx_nested_tensor_from_tensor_list() is an implementation of
# nested_tensor_from_tensor_list() that is supported by ONNX tracing.
@torch.jit.unused
def _onnx_nested_tensor_from_tensor_list(tensor_list: List[Tensor]) -> NestedTensor:
    max_size = []
    for i in range(tensor_list[0].dim()):
        max_size_i = torch.max(
            torch.stack([img.shape[i] for img in tensor_list]).to(torch.float32)
        ).to(torch.int64)
        max_size.append(max_size_i)
    max_size = tuple(max_size)

    # work around for
    # pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
    # m[: img.shape[1], :img.shape[2]] = False
    # which is not yet supported in onnx
    padded_imgs = []
    padded_masks = []
    for img in tensor_list:
        padding = [(s1 - s2) for s1, s2 in zip(max_size, tuple(img.shape))]
        padded_img = torch.nn.functional.pad(img, (0, padding[2], 0, padding[1], 0, padding[0]))
        padded_imgs.append(padded_img)

        m = torch.zeros_like(img[0], dtype=torch.int, device=img.device)
        padded_mask = torch.nn.functional.pad(m, (0, padding[2], 0, padding[1]), "constant", 1)
        padded_masks.append(padded_mask.to(torch.bool))

    tensor = torch.stack(padded_imgs)
    mask = torch.stack(padded_masks)

    return NestedTensor(tensor, mask=mask)


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

class SetCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses,
                 num_points, oversample_ratio, importance_sample_ratio):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

        # pointwise mask loss parameters
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio

    def loss_labels(self, outputs, targets, indices, num_masks):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"].float()

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {"loss_ce": loss_ce}
        return losses
    
    def loss_masks(self, outputs, targets, indices, num_masks):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # No need to upsample predictions as we are using normalized coordinates :)
        # N x 1 x H x W
        src_masks = src_masks[:, None]
        target_masks = target_masks[:, None]

        with torch.no_grad():
            # sample point_coords
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            # get gt labels
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)

        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)

        losses = {
            "loss_mask": sigmoid_ce_loss_jit(point_logits, point_labels, num_masks),
            "loss_dice": dice_loss_jit(point_logits, point_labels, num_masks),
        }

        del src_masks
        del target_masks
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_masks):
        loss_map = {
            'labels': self.loss_labels,
            'masks': self.loss_masks,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_masks)

    def forward(self, outputs, targets):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_masks))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_masks)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses



def batch_dice_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


batch_dice_loss_jit = torch.jit.script(
    batch_dice_loss
)  # type: torch.jit.ScriptModule


def batch_sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    hw = inputs.shape[1]

    pos = F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction="none"
    )
    neg = F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none"
    )

    loss = torch.einsum("nc,mc->nm", pos, targets) + torch.einsum(
        "nc,mc->nm", neg, (1 - targets)
    )

    return loss / hw


batch_sigmoid_ce_loss_jit = torch.jit.script(
    batch_sigmoid_ce_loss
)  # type: torch.jit.ScriptModule


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_mask: float = 1, cost_dice: float = 1, num_points: int = 0):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_mask: This is the relative weight of the focal loss of the binary mask in the matching cost
            cost_dice: This is the relative weight of the dice loss of the binary mask in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice

        assert cost_class != 0 or cost_mask != 0 or cost_dice != 0, "all costs cant be 0"

        self.num_points = num_points

    @torch.no_grad()
    def memory_efficient_forward(self, outputs, targets):
        """More memory-friendly matching"""
        bs, num_queries = outputs["pred_logits"].shape[:2]

        indices = []
        device = torch.device('cpu')

        # Iterate through batch size
        for b in range(bs):

            out_prob = outputs["pred_logits"][b].softmax(-1)  # [num_queries, num_classes]
            tgt_ids = targets[b]["labels"]
            if tgt_ids.numel() == 0:
                indices.append((torch.empty(0, dtype=torch.int64, device=device),
                                torch.empty(0, dtype=torch.int64, device=device)))
                continue
            # Compute the classification cost. Contrary to the loss, we don't use the NLL,
            # but approximate it in 1 - proba[target class].
            # The 1 is a constant that doesn't change the matching, it can be ommitted.
            cost_class = -out_prob[:, tgt_ids]

            out_mask = outputs["pred_masks"][b]  # [num_queries, H_pred, W_pred]
            # gt masks are already padded when preparing target
            tgt_mask = targets[b]["masks"].to(out_mask)

            out_mask = out_mask[:, None]
            tgt_mask = tgt_mask[:, None]
            # all masks share the same set of points for efficient matching!
            point_coords = torch.rand(1, self.num_points, 2, device=out_mask.device)
            # get gt labels
            tgt_mask = point_sample(
                tgt_mask,
                point_coords.repeat(tgt_mask.shape[0], 1, 1),
                align_corners=False,
            ).squeeze(1)

            out_mask = point_sample(
                out_mask,
                point_coords.repeat(out_mask.shape[0], 1, 1),
                align_corners=False,
            ).squeeze(1)

            with autocast('cuda', enabled=False):
                out_mask = out_mask.float()
                tgt_mask = tgt_mask.float()
                # Compute the focal loss between masks
                # cost_mask = batch_sigmoid_ce_loss_jit(out_mask, tgt_mask)

                # # Compute the dice loss betwen masks
                # cost_dice = batch_dice_loss_jit(out_mask, tgt_mask)
                cost_mask = batch_sigmoid_ce_loss(out_mask, tgt_mask)
                cost_dice = batch_dice_loss(out_mask, tgt_mask)
            # Final cost matrix
            C = (
                self.cost_mask * cost_mask
                + self.cost_class * cost_class
                + self.cost_dice * cost_dice
            )
            C = C.reshape(num_queries, -1).cpu()

            indices.append(linear_sum_assignment(C))

        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]

    @torch.no_grad()
    def forward(self, outputs, targets):
        """Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_masks": Tensor of dim [batch_size, num_queries, H_pred, W_pred] with the predicted masks

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "masks": Tensor of dim [num_target_boxes, H_gt, W_gt] containing the target masks

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        return self.memory_efficient_forward(outputs, targets)





# class SetCriterion(nn.Module):
#     def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses,
#                  num_points, oversample_ratio, importance_sample_ratio):
#         super().__init__()
#         self.num_classes = num_classes
#         self.matcher = matcher
#         self.weight_dict = weight_dict
#         self.eos_coef = eos_coef
#         self.losses = losses

#         empty_weight = torch.ones(self.num_classes + 1)
#         empty_weight[-1] = self.eos_coef
#         self.register_buffer("empty_weight", empty_weight)

#         self.num_points = num_points
#         self.oversample_ratio = oversample_ratio
#         self.importance_sample_ratio = importance_sample_ratio

#     def loss_labels(self, outputs, targets, indices, num_masks):
#         assert "pred_logits" in outputs
#         src_logits = outputs["pred_logits"].float()

#         idx = self._get_src_permutation_idx(indices)
#         target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
#         target_classes = torch.full(
#             src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
#         )
#         target_classes[idx] = target_classes_o

#         loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
#         return {"loss_ce": loss_ce}

#     def loss_masks(self, outputs, targets, indices, num_masks):
#         assert "pred_masks" in outputs

#         src_idx = self._get_src_permutation_idx(indices)
#         tgt_idx = self._get_tgt_permutation_idx(indices)
#         src_masks = outputs["pred_masks"]
#         src_masks = src_masks[src_idx]
#         masks = [t["masks"] for t in targets]

#         # 你这里的 nested_tensor_from_tensor_list 是你项目的函数
#         target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
#         target_masks = target_masks.to(src_masks.device)
#         target_masks = target_masks[tgt_idx]

#         src_masks = src_masks[:, None]
#         target_masks = target_masks[:, None]

#         with torch.no_grad():
#             point_coords = get_uncertain_point_coords_with_randomness(
#                 src_masks,
#                 lambda logits: calculate_uncertainty(logits),
#                 self.num_points,
#                 self.oversample_ratio,
#                 self.importance_sample_ratio,
#             )
#             point_labels = point_sample(target_masks, point_coords, align_corners=False).squeeze(1)

#         point_logits = point_sample(src_masks, point_coords, align_corners=False).squeeze(1)

#         losses = {
#             "loss_mask": sigmoid_ce_loss_jit(point_logits, point_labels, num_masks),
#             "loss_dice": dice_loss_jit(point_logits, point_labels, num_masks),
#         }
#         return losses

#     def loss_center_count(self, outputs, targets, indices, num_masks):
#         """
#         约束：一个预测mask覆盖多个实例中心时惩罚

#         需要targets中每个样本带 'centers' 字段，形状 [K, 2]，归一化坐标(0~1)

#         outputs["pred_masks"]: [B, Nq, H, W] (sigmoid前)
#         """
#         device = outputs["pred_masks"].device
#         pred_masks = outputs["pred_masks"]  # [B, Nq, H, W]
#         batch_size, num_queries, H, W = pred_masks.shape

#         batch_idx = []
#         query_idx = []
#         for b, (src, tgt) in enumerate(indices):
#             batch_idx.append(torch.full_like(src, b))
#             query_idx.append(src)
#         batch_idx = torch.cat(batch_idx)
#         query_idx = torch.cat(query_idx)

#         matched_masks = pred_masks[batch_idx, query_idx]  # [M, H, W]
#         matched_masks = matched_masks.sigmoid()

#         loss = 0.0
#         count = 0
#         thresh = 0.5

#         for i, (b, q) in enumerate(zip(batch_idx, query_idx)):
#             centers = targets[b]["centers"].to(device)  # [K, 2]
#             if centers.shape[0] == 0:
#                 continue

#             mask = matched_masks[i]

#             # centers坐标归一化映射到mask尺寸
#             centers_scaled = (centers * torch.tensor([W, H], device=device)).long()

#             cover_count = 0
#             for (x, y) in centers_scaled:
#                 x = x.clamp(0, W - 1)
#                 y = y.clamp(0, H - 1)
#                 if mask[y, x] > thresh:
#                     cover_count += 1

#             if cover_count > 1:
#                 loss += cover_count - 1
#             count += 1

#         if count == 0:
#             return {"loss_center_count": torch.tensor(0., device=device)}
#         else:
#             return {"loss_center_count": loss / count}

#     def _get_src_permutation_idx(self, indices):
#         batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
#         src_idx = torch.cat([src for (src, _) in indices])
#         return batch_idx, src_idx

#     def _get_tgt_permutation_idx(self, indices):
#         batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
#         tgt_idx = torch.cat([tgt for (_, tgt) in indices])
#         return batch_idx, tgt_idx

#     def get_loss(self, loss, outputs, targets, indices, num_masks):
#         loss_map = {
#             'labels': self.loss_labels,
#             'masks': self.loss_masks,
#             'center_count': self.loss_center_count,
#         }
#         assert loss in loss_map, f"Unknown loss: {loss}"
#         return loss_map[loss](outputs, targets, indices, num_masks)

#     def forward(self, outputs, targets):
#         outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}
#         indices = self.matcher(outputs_without_aux, targets)

#         num_masks = sum(len(t["labels"]) for t in targets)
#         num_masks = torch.as_tensor([num_masks], dtype=torch.float, device=next(iter(outputs.values())).device)
#         if torch.distributed.is_initialized():
#             torch.distributed.all_reduce(num_masks)
#         num_masks = torch.clamp(num_masks / torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1, min=1).item()

#         losses = {}
#         for loss in self.losses:
#             losses.update(self.get_loss(loss, outputs, targets, indices, num_masks))

#         if "aux_outputs" in outputs:
#             for i, aux_outputs in enumerate(outputs["aux_outputs"]):
#                 indices = self.matcher(aux_outputs, targets)
#                 for loss in self.losses:
#                     l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_masks)
#                     l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
#                     losses.update(l_dict)

#         return losses




if __name__ == "__main__":
    # multi class loss ################################################################
    y_true = torch.tensor([[[0, 0, 1, 0], [1, 0, 0, 1], [1, 1, 1, 1], [1, 0, 0, 1]]], requires_grad=False)
    y_pred = torch.tensor([[[[0.65, 0.04, 0.20, 0.70], [0.30, 0.55, 0.20, 0.10],
                             [0.55, 0.30, 0.60, 0.05], [0.03, 0.05, 0.98, 0.95]],
                            [[0.35, 0.96, 0.80, 0.30], [0.70, 0.45, 0.80, 0.90],
                             [0.45, 0.70, 0.40, 0.95], [0.97, 0.95, 0.02, 0.05]]]], requires_grad=True)
    # print(y_true.shape)  # [1, 4, 4]
    # print(y_pred.shape)  # [1, 2, 4, 4]
    y_pred_, y_true_ = y_pred.clone().requires_grad_(), y_true.clone()
    #
    # criterion = nn.CrossEntropyLoss()
    # loss = criterion(y_pred_, y_true_.long())
    # print('nn.CrossEntropyLoss', loss.item())
    #
    # criterion = LabelSmoothSoftmaxCE(smoothing=0.1)
    # loss = criterion(y_pred_, y_true_.long())
    # print('LabelSmoothingCrossEntropy', loss.item())

    # criteria = FocalLoss(alpha=0.25, gamma=2)
    # loss = criteria(logits, lbs)
    # print('FocalLoss', loss.item())
    #
    # criteria = FocalLoss(alpha=0.25, gamma=2)
    # loss = criteria(logits, lbs2)
    # print('FocalLoss', loss.item())

    ################################################################################
    # y_true = torch.randint(4, size=(1, 4, 4), requires_grad=False)
    # y_pred = torch.rand(size=(1, 4, 4, 4), requires_grad=True)
    y_true = torch.tensor([[[[0, 0, 1, 0], [1, 0, 0, 1], [1, 1, 1, 1], [1, 0, 0, 1]]]], requires_grad=False)
    # y_true = torch.tensor([[[0, 0, 1, 0], [1, 0, 0, 1], [1, 1, 1, 1], [1, 0, 0, 1]]], requires_grad=False)
    y_pred = torch.tensor([[[[0.65, 0.04, 0.20, 0.70], [0.30, 0.55, 0.20, 0.10],
                             [0.55, 0.30, 0.60, 0.05], [0.03, 0.05, 0.98, 0.95]]]], requires_grad=True)
    # print(y_true.shape)  # [1, 1, 4, 4]
    # print(y_pred.shape)  # [1, 1, 4, 4]

    # criterion = SigmoidDiceLoss(exponent=2, smooth=1)
    # loss = criterion(y_pred, y_true)
    # print('SigmoidDiceLoss', loss.item())
    #
    # criteria = FocalLoss(alpha=0.25, gamma=2)
    # loss = criteria(y_pred, y_true)
    # print('FocalLoss', loss.item())

    y_pred = torch.randn(8, 19, 384, 384)  # nchw, float/half
    y_true1 = torch.randint(0, 19, (8, 384, 384))  # nchw, int64_t
    y_true2 = torch.randint(0, 1, (8, 19, 384, 384))  # nchw, int64_t

    # print(y_pred.shape)  # [1, 1, 4, 4]
    # print(y_true1.shape)  # [1, 1, 4, 4]
    criteria = LovaszSoftmaxLoss()
    loss = criteria(y_pred, y_true1)
    print('LovaszSoftmaxLoss', loss)

    # y_true1 = convert_to_one_hot(y_true1, 19)
    # print(y_pred.shape)  # [1, 1, 4, 4]
    # print(y_true1.shape)  # [1, 1, 4, 4]
    # criteria = LovaszSoftmax()
    # loss = criteria(y_pred, y_true1)
    # print('LovaszSoftmax', loss.item())

    # criterion = nn.BCEWithLogitsLoss()
    # loss = criterion(y_pred, y_true.long())
    # print('Focal_Loss', loss.item())
    # print(y_true.shape)  # [1, 4, 4]
    # print(y_pred.shape)  # [1, 1, 4, 4]
    # y_pred_, y_true_ = y_pred.clone().requires_grad_(), y_true.clone()
    #
    # criterion = Focal_Loss(alpha=0.5, gamma=2)
    # loss = criterion(y_pred_, y_true_)
    # print('Focal_Loss', loss.item())
    #
    # criterion = FocalLoss(alpha=0.5, gamma=2, balance_index=0, smooth=1e-5, size_average=True)
    # loss = criterion(y_pred_, y_true_)
    # print('FocalLoss', loss.item())
    # criterion = BinaryCrossEntropy()
    # loss = criterion(y_pred_, y_true_.float())
    # print('BinaryCrossEntropy', loss.item())

    # criterion = nn.BCEWithLogitsLoss()
    # loss = criterion(y_pred_, y_true_.float())
    # print('nn.BCEWithLogitsLoss', loss.item())
