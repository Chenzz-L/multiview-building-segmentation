import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# 如果你安装了 detectron2，保留这些引用
from detectron2.layers import ShapeSpec, cat
from detectron2.structures import Instances, Boxes
from fvcore.nn import sigmoid_focal_loss_jit

# -------------------------------------------------------------------------
#  1. 工具函数 (Loss, NMS) - 内置防止 import 错误
# -------------------------------------------------------------------------

def dice_loss(input, target):
    input = input.contiguous().view(input.size()[0], -1)
    target = target.contiguous().view(target.size()[0], -1).float()
    a = torch.sum(input * target, 1)
    b = torch.sum(input * input, 1) + 0.001
    c = torch.sum(target * target, 1) + 0.001
    d = (2 * a) / (b + c)
    return 1 - d

def point_nms(heat, kernel=2):
    """
    修复版 Point NMS
    使用 kernel=3, padding=1 保证尺寸不变
    """
    # 强制使用 3x3 window 寻找极大值，这是最稳定的做法
    # 也可以手动计算 padding = (kernel - 1) // 2
    # 但如果 kernel=2, padding=0 会导致尺寸减小，所以建议至少用 kernel=3
    hmax = F.max_pool2d(heat, (3, 3), stride=1, padding=1)
    keep = (hmax == heat).float()
    return heat * keep

def matrix_nms(seg_masks, cate_labels, cate_scores, sigma=2.0, kernel='gaussian', sum_masks=None):
    """Matrix NMS 核心实现"""
    n_samples = len(cate_labels)
    if n_samples == 0: return []
    
    # 铺平 Mask: [N, H, W] -> [N, H*W]
    seg_masks = seg_masks.reshape(n_samples, -1).float()
    
    # 计算 IoU 矩阵
    inter_matrix = torch.mm(seg_masks, seg_masks.transpose(1, 0))
    if sum_masks is None:
        sum_masks = seg_masks.sum(1)
    sum_masks_x = sum_masks.expand(n_samples, n_samples)
    # IoU = inter / (area1 + area2 - inter)
    iou_matrix = (inter_matrix / (sum_masks_x + sum_masks_x.transpose(1, 0) - inter_matrix)).triu(diagonal=1)
    
    # Label 矩阵 (同类才抑制)
    cate_labels_x = cate_labels.expand(n_samples, n_samples)
    label_matrix = (cate_labels_x == cate_labels_x.transpose(1, 0)).float().triu(diagonal=1)
    
    # IoU 衰减
    decay_iou = iou_matrix * label_matrix
    compensate_iou, _ = decay_iou.max(0)
    compensate_iou = compensate_iou.expand(n_samples, n_samples).transpose(1, 0)
    
    if kernel == 'gaussian':
        decay_matrix = torch.exp(-1 * sigma * (decay_iou ** 2))
        compensate_matrix = torch.exp(-1 * sigma * (compensate_iou ** 2))
        decay_coefficient, _ = (decay_matrix / compensate_matrix).min(0)
    else:
        decay_matrix = (1 - decay_iou) / (1 - compensate_iou)
        decay_coefficient, _ = decay_matrix.min(0)
        
    return cate_scores * decay_coefficient


# -------------------------------------------------------------------------
#  2. 主解码器类 SOLOv2Decoder
# -------------------------------------------------------------------------

class SOLOv2Decoder(nn.Module):
    def __init__(
        self,
        in_channels=256,
        num_classes=1,
        num_kernels=256,
        num_grids=[40, 36, 24, 16, 12],
        strides=[8, 8, 16, 32, 32],
        scale_ranges=((1, 96), (48, 192), (96, 384), (192, 768), (384, 2048)),
        sigma=0.2,
        # Loss 权重
        ins_loss_weight=5.0,
        focal_loss_gamma=2.0,
        focal_loss_alpha=0.25,
        focal_loss_weight=1.0,
        # 推理参数
        score_thr=0.1,
        mask_thr=0.5,
        update_thr=0.05,
        nms_pre=500,
        max_per_img=100,
        **kwargs
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_kernels = num_kernels
        self.num_grids = num_grids
        self.strides = strides
        self.scale_ranges = scale_ranges
        self.sigma = sigma
        
        # Loss params
        self.ins_loss_weight = ins_loss_weight
        self.focal_loss_gamma = focal_loss_gamma
        self.focal_loss_alpha = focal_loss_alpha
        self.focal_loss_weight = focal_loss_weight

        # Inference params
        self.score_thr = score_thr
        self.mask_thr = mask_thr
        self.update_thr = update_thr
        self.nms_pre = nms_pre
        self.max_per_img = max_per_img

        # [修复] 这里的初始化参数必须和下面的 SOLOv2InsHead 定义匹配
        self.ins_head = SOLOv2InsHead(in_channels, num_classes, num_kernels, num_grids)
        self.mask_head = SOLOv2MaskHead(in_channels, num_kernels, num_levels=len(num_grids)-1)

    def forward(self, features, gt_instances=None, image_sizes=None):
        """
        Args:
            features (dict): {'p2': ..., 'p3': ...}
            gt_instances (list[Instances]): 训练必须
            image_sizes (list[tuple]): 推理必须
        """
        # 1. 整理特征
        if isinstance(features, dict):
            keys = sorted(features.keys()) # p2, p3, p4, p5
            features = [features[k] for k in keys]

        # 构造输入 [P2, P3, P4, P5, P6]
        # 假设输入特征只有 4 层 [P2, P3, P4, P5]
        ins_feats = [
            F.interpolate(features[0], scale_factor=0.5, mode='bilinear'), # P2 -> P3 size (SOLO特有)
            features[1],
            features[2],
            features[3],
            F.interpolate(features[3], scale_factor=0.5, mode='bilinear')  # P5 -> P6 size
        ]
        mask_feats = features[:4] # P2-P5

        # 2. 运行 Heads
        cate_preds, kernel_preds = self.ins_head(ins_feats)
        mask_pred = self.mask_head(mask_feats)

        # 3. 分支：训练 / 推理
        if self.training:
            assert gt_instances is not None, "Training requires gt_instances"
            mask_feat_size = mask_pred.size()[-2:]
            
            targets = self.get_ground_truth(gt_instances, mask_feat_size)
            
            return self.loss(cate_preds, kernel_preds, mask_pred, targets)
        else:
            return self.inference(cate_preds, kernel_preds, mask_pred, image_sizes)

    # =======================================================
    #  Label Assignment (生成 GT)
    # =======================================================
    @torch.no_grad()
    def get_ground_truth(self, gt_instances, mask_feat_size=None):
        ins_label_list, cate_label_list, ins_ind_label_list, grid_order_list = [], [], [], []
        
        for img_idx in range(len(gt_instances)):
            cur_ins, cur_cate, cur_ind, cur_grid = self.get_ground_truth_single(
                img_idx, gt_instances, mask_feat_size
            )
            ins_label_list.append(cur_ins)
            cate_label_list.append(cur_cate)
            ins_ind_label_list.append(cur_ind)
            grid_order_list.append(cur_grid)
            
        return ins_label_list, cate_label_list, ins_ind_label_list, grid_order_list

    def get_ground_truth_single(self, img_idx, gt_instances, mask_feat_size):
        gt_bboxes_raw = gt_instances[img_idx].gt_boxes.tensor
        gt_labels_raw = gt_instances[img_idx].gt_classes
        if hasattr(gt_instances[img_idx].gt_masks, 'tensor'):
             gt_masks_raw = gt_instances[img_idx].gt_masks.tensor
        else:
             gt_masks_raw = gt_instances[img_idx].gt_masks
             
        device = gt_labels_raw.device
        
        # 计算面积
        gt_areas = torch.sqrt((gt_bboxes_raw[:, 2] - gt_bboxes_raw[:, 0]) * (
                gt_bboxes_raw[:, 3] - gt_bboxes_raw[:, 1]))

        ins_label_list = []
        cate_label_list = []
        ins_ind_label_list = []
        grid_order_list = []

        # 遍历 FPN 层
        for (lower_bound, upper_bound), stride, num_grid in zip(self.scale_ranges, self.strides, self.num_grids):
            
            hit_indices = ((gt_areas >= lower_bound) & (gt_areas <= upper_bound)).nonzero().flatten()
            num_ins = len(hit_indices)

            ins_label = []
            grid_order = []
            cate_label = torch.zeros([num_grid, num_grid], dtype=torch.int64, device=device)
            cate_label = torch.fill_(cate_label, self.num_classes)
            ins_ind_label = torch.zeros([num_grid ** 2], dtype=torch.bool, device=device)

            if num_ins == 0:
                ins_label = torch.zeros([0, mask_feat_size[0], mask_feat_size[1]], dtype=torch.uint8, device=device)
                ins_label_list.append(ins_label)
                cate_label_list.append(cate_label)
                ins_ind_label_list.append(ins_ind_label)
                grid_order_list.append([])
                continue

            gt_bboxes = gt_bboxes_raw[hit_indices]
            gt_labels = gt_labels_raw[hit_indices]
            gt_masks = gt_masks_raw[hit_indices, ...]

            half_ws = 0.5 * (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * self.sigma
            half_hs = 0.5 * (gt_bboxes[:, 3] - gt_bboxes[:, 1]) * self.sigma

            center_ws, center_hs = self.center_of_mass(gt_masks)
            valid_mask_flags = gt_masks.sum(dim=-1).sum(dim=-1) > 0

            # Resize GT Mask
            gt_masks = F.interpolate(gt_masks.unsqueeze(1).float(), size=mask_feat_size, mode='nearest').squeeze(1).byte()
            
            for seg_mask, gt_label, half_h, half_w, center_h, center_w, valid_flag in zip(gt_masks, gt_labels, half_hs, half_ws, center_hs, center_ws, valid_mask_flags):
                if not valid_flag: continue
                
                # 4 是 mask feature stride
                coord_w = int((center_w / (mask_feat_size[1] * 4)) // (1. / num_grid))
                coord_h = int((center_h / (mask_feat_size[0] * 4)) // (1. / num_grid))

                top_box = max(0, int(((center_h - half_h) / (mask_feat_size[0] * 4)) // (1. / num_grid)))
                down_box = min(num_grid - 1, int(((center_h + half_h) / (mask_feat_size[0] * 4)) // (1. / num_grid)))
                left_box = max(0, int(((center_w - half_w) / (mask_feat_size[1] * 4)) // (1. / num_grid)))
                right_box = min(num_grid - 1, int(((center_w + half_w) / (mask_feat_size[1] * 4)) // (1. / num_grid)))

                top = max(top_box, coord_h-1)
                down = min(down_box, coord_h+1)
                left = max(coord_w-1, left_box)
                right = min(right_box, coord_w+1)

                cate_label[top:(down+1), left:(right+1)] = gt_label
                for i in range(top, down+1):
                    for j in range(left, right+1):
                        label = int(i * num_grid + j)
                        ins_label.append(seg_mask)
                        ins_ind_label[label] = True
                        grid_order.append(label)

            if len(ins_label) == 0:
                ins_label = torch.zeros([0, mask_feat_size[0], mask_feat_size[1]], dtype=torch.uint8, device=device)
            else:
                ins_label = torch.stack(ins_label, 0)
                
            ins_label_list.append(ins_label)
            cate_label_list.append(cate_label)
            ins_ind_label_list.append(ins_ind_label)
            grid_order_list.append(grid_order)
            
        return ins_label_list, cate_label_list, ins_ind_label_list, grid_order_list

    # =======================================================
    #  Loss 计算
    # =======================================================
    def loss(self, cate_preds, kernel_preds, ins_pred, targets):
        ins_label_list, cate_label_list, ins_ind_label_list, grid_order_list = targets
        
        ins_labels = [torch.cat([ins_labels_level_img for ins_labels_level_img in ins_labels_level], 0)
                      for ins_labels_level in zip(*ins_label_list)]

        kernel_preds = [[kernel_preds_level_img.view(kernel_preds_level_img.shape[0], -1)[:, grid_orders_level_img]
                         for kernel_preds_level_img, grid_orders_level_img in zip(kernel_preds_level, grid_orders_level)]
                        for kernel_preds_level, grid_orders_level in zip(kernel_preds, zip(*grid_order_list))]

        # 生成预测 Mask
        ins_pred_list = []
        for b_kernel_pred in kernel_preds: 
            b_mask_pred = []
            for idx, kernel_pred in enumerate(b_kernel_pred): 
                if kernel_pred.size()[-1] == 0: continue
                
                cur_ins_pred = ins_pred[idx, ...] 
                H, W = cur_ins_pred.shape[-2:]
                N, I = kernel_pred.shape
                
                cur_ins_pred = cur_ins_pred.unsqueeze(0)
                kernel_pred = kernel_pred.permute(1, 0).view(I, -1, 1, 1)
                cur_ins_pred = F.conv2d(cur_ins_pred, kernel_pred, stride=1).view(-1, H, W)
                b_mask_pred.append(cur_ins_pred)
                
            if len(b_mask_pred) == 0:
                b_mask_pred = None
            else:
                b_mask_pred = torch.cat(b_mask_pred, 0)
            ins_pred_list.append(b_mask_pred)

        # Dice Loss
        loss_ins = []
        for input, target in zip(ins_pred_list, ins_labels):
            if input is None: continue
            input = torch.sigmoid(input)
            loss_ins.append(dice_loss(input, target))
            
        if len(loss_ins) == 0:
            loss_ins = torch.tensor(0.0, device=cate_preds[0].device)
        else:
            loss_ins = torch.cat(loss_ins).mean() * self.ins_loss_weight

        # Focal Loss
        cate_labels = [torch.cat([cate_labels_level_img.flatten() for cate_labels_level_img in cate_labels_level])
                       for cate_labels_level in zip(*cate_label_list)]
        flatten_cate_labels = torch.cat(cate_labels)
        
        cate_preds = [cate_pred.permute(0, 2, 3, 1).reshape(-1, self.num_classes) for cate_pred in cate_preds]
        flatten_cate_preds = torch.cat(cate_preds)

        pos_inds = torch.nonzero(flatten_cate_labels != self.num_classes).squeeze(1)
        flatten_cate_labels_oh = torch.zeros_like(flatten_cate_preds)
        flatten_cate_labels_oh[pos_inds, flatten_cate_labels[pos_inds]] = 1

        num_pos = max(1.0, pos_inds.numel())
        loss_cate = self.focal_loss_weight * sigmoid_focal_loss_jit(
            flatten_cate_preds, flatten_cate_labels_oh,
            gamma=self.focal_loss_gamma,
            alpha=self.focal_loss_alpha,
            reduction="sum"
        ) / num_pos

        return {'loss_ins': loss_ins, 'loss_cate': loss_cate}

    # =======================================================
    #  推理逻辑 (Inference)
    # =======================================================
    def inference(self, cate_preds, kernel_preds, mask_pred, image_sizes):
        results = []
        
        # 3. Point NMS (Fix: 这里调用修改后的 point_nms，kernel=3)
        cate_preds = [point_nms(cate_p.sigmoid(), kernel=3).permute(0, 2, 3, 1)
                     for cate_p in cate_preds]

        num_ins_levels = len(cate_preds)
        
        for img_idx in range(len(image_sizes)):
            ori_size = image_sizes[img_idx]

            # 收集
            pred_cate = [cate_preds[i][img_idx].view(-1, self.num_classes).detach()
                          for i in range(num_ins_levels)]
            pred_kernel = [kernel_preds[i][img_idx].permute(1, 2, 0).view(-1, self.num_kernels).detach()
                            for i in range(num_ins_levels)]
            pred_mask = mask_pred[img_idx, ...].unsqueeze(0)

            pred_cate = torch.cat(pred_cate, dim=0)
            pred_kernel = torch.cat(pred_kernel, dim=0)

            # 单张推理
            result = self.inference_single_image(pred_cate, pred_kernel, pred_mask, ori_size)
            results.append(result)
        return results

    def inference_single_image(self, cate_preds, kernel_preds, seg_preds, ori_size):
        # 1. Score Threshold
        inds = (cate_preds > self.score_thr)
        cate_scores = cate_preds[inds]
        if len(cate_scores) == 0:
            return self._empty_result(ori_size, cate_preds.device)

        inds = inds.nonzero()
        cate_labels = inds[:, 1]
        kernel_preds = kernel_preds[inds[:, 0]]

        # 2. Dynamic Conv
        N, I = kernel_preds.shape
        kernel_preds = kernel_preds.view(N, I, 1, 1)
        seg_preds = F.conv2d(seg_preds, kernel_preds, stride=1).squeeze(0).sigmoid()

        # 3. Mask Threshold
        seg_masks = seg_preds > self.mask_thr
        sum_masks = seg_masks.sum((1, 2)).float()

        # 4. Filter
        keep = sum_masks > 0
        if keep.sum() == 0:
             return self._empty_result(ori_size, cate_preds.device)

        seg_masks = seg_masks[keep]
        seg_preds = seg_preds[keep]
        sum_masks = sum_masks[keep]
        cate_scores = cate_scores[keep]
        cate_labels = cate_labels[keep]

        # 5. Maskness & Sort
        seg_scores = (seg_preds * seg_masks.float()).sum((1, 2)) / sum_masks
        cate_scores *= seg_scores

        sort_inds = torch.argsort(cate_scores, descending=True)
        if len(sort_inds) > self.nms_pre:
            sort_inds = sort_inds[:self.nms_pre]
        
        seg_masks = seg_masks[sort_inds, :, :]
        seg_preds = seg_preds[sort_inds, :, :]
        sum_masks = sum_masks[sort_inds]
        cate_scores = cate_scores[sort_inds]
        cate_labels = cate_labels[sort_inds]

        # 6. Matrix NMS
        cate_scores = matrix_nms(seg_masks, cate_labels, cate_scores, sum_masks=sum_masks)
        keep = cate_scores >= self.update_thr
        
        if keep.sum() == 0:
            return self._empty_result(ori_size, cate_preds.device)

        seg_preds = seg_preds[keep]
        cate_scores = cate_scores[keep]
        cate_labels = cate_labels[keep]

        # 7. Final TopK & Resize
        sort_inds = torch.argsort(cate_scores, descending=True)
        if len(sort_inds) > self.max_per_img:
            sort_inds = sort_inds[:self.max_per_img]
        seg_preds = seg_preds[sort_inds]
        cate_scores = cate_scores[sort_inds]
        cate_labels = cate_labels[sort_inds]

        # 还原到原图大小
        seg_preds = F.interpolate(seg_preds.unsqueeze(0), size=ori_size, mode='bilinear', align_corners=False).squeeze(0)
        seg_masks = seg_preds > self.mask_thr

        # Output
        results = Instances(ori_size)
        results.scores = cate_scores
        results.pred_classes = cate_labels
        results.pred_masks = seg_masks
        results.pred_boxes = Boxes(torch.zeros(len(cate_scores), 4, device=cate_scores.device))
        return results

    def _empty_result(self, size, device):
        r = Instances(size)
        r.scores = torch.tensor([], device=device)
        r.pred_classes = torch.tensor([], device=device)
        r.pred_masks = torch.tensor([], device=device)
        r.pred_boxes = Boxes(torch.tensor([], device=device))
        return r
    
    @staticmethod
    def center_of_mass(bitmasks):
        _, h, w = bitmasks.size()
        ys = torch.arange(0, h, dtype=torch.float32, device=bitmasks.device)
        xs = torch.arange(0, w, dtype=torch.float32, device=bitmasks.device)
        m00 = bitmasks.sum(dim=-1).sum(dim=-1).clamp(min=1e-6)
        m10 = (bitmasks * xs).sum(dim=-1).sum(dim=-1)
        m01 = (bitmasks * ys.view(-1, 1)).sum(dim=-1).sum(dim=-1)
        return m10 / m00, m01 / m00


# -------------------------------------------------------------------------
#  3. Heads (纯 PyTorch 实现，移除了 Detectron2 Cfg 依赖)
# -------------------------------------------------------------------------

class SOLOv2InsHead(nn.Module):
    def __init__(self, in_channels, num_classes, num_kernels, num_grids):
        super().__init__()
        self.num_grids = num_grids
        
        # 定义 Tower
        self.cate_tower = self._make_tower(in_channels, in_channels)
        self.kernel_tower = self._make_tower(in_channels + 2, in_channels) # +2 for Coord

        self.cate_pred = nn.Conv2d(in_channels, num_classes, 3, padding=1)
        self.kernel_pred = nn.Conv2d(in_channels, num_kernels, 3, padding=1)
        
        # Init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None: nn.init.constant_(m.bias, 0)
        
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.cate_pred.bias, bias_value)

    def _make_tower(self, in_ch, out_ch, num_convs=4):
        layers = []
        for i in range(num_convs):
            layers.append(nn.Conv2d(in_ch if i==0 else out_ch, out_ch, 3, padding=1, bias=False))
            layers.append(nn.GroupNorm(32, out_ch))
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, features):
        cate_preds, kernel_preds = [], []
        # features: [P2, P3, P4, P5, P6]
        for idx, feat in enumerate(features):
            if idx >= len(self.num_grids): break

            # CoordConv
            N, _, H, W = feat.shape
            x = torch.linspace(-1, 1, W, device=feat.device).view(1, 1, 1, W).expand(N, 1, H, W)
            y = torch.linspace(-1, 1, H, device=feat.device).view(1, 1, H, 1).expand(N, 1, H, W)
            coord = torch.cat([x, y], 1)
            feat_with_coord = torch.cat([feat, coord], 1)
            
            # Resize
            kernel_feat = F.interpolate(feat_with_coord, size=self.num_grids[idx], mode='bilinear')
            cate_feat = kernel_feat[:, :-2, :, :]

            kernel_preds.append(self.kernel_pred(self.kernel_tower(kernel_feat)))
            cate_preds.append(self.cate_pred(self.cate_tower(cate_feat)))
            
        return cate_preds, kernel_preds

class SOLOv2MaskHead(nn.Module):
    def __init__(self, in_channels, out_channels, num_levels=4):
        super().__init__()
        self.convs = nn.ModuleList()
        # 简单融合策略
        for i in range(num_levels):
            ch = in_channels + 2 if i == (num_levels - 1) else in_channels
            self.convs.append(nn.Sequential(
                nn.Conv2d(ch, 128, 3, padding=1, bias=False),
                nn.GroupNorm(32, 128),
                nn.ReLU(inplace=True)
            ))
        
        self.conv_out = nn.Sequential(
            nn.Conv2d(128, out_channels, 1, padding=0, bias=False),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True)
        )
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None: nn.init.constant_(m.bias, 0)

    def forward(self, features):
        # P2 Base
        feat_sum = self.convs[0](features[0])
        
        for i in range(1, len(self.convs)):
            feat = features[i]
            if i == len(self.convs) - 1: # Last level add coord
                 N, _, H, W = feat.shape
                 x = torch.linspace(-1, 1, W, device=feat.device).view(1, 1, 1, W).expand(N, 1, H, W)
                 y = torch.linspace(-1, 1, H, device=feat.device).view(1, 1, H, 1).expand(N, 1, H, W)
                 feat = torch.cat([feat, x, y], 1)
            
            feat = self.convs[i](feat)
            if feat.shape[-2:] != feat_sum.shape[-2:]:
                feat = F.interpolate(feat, size=feat_sum.shape[-2:], mode='bilinear')
            
            feat_sum = feat_sum + feat
            
        return self.conv_out(feat_sum)
