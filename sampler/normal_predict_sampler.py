import numpy as np
import torch

def normal_sampler(batch, crop_width, crop_height, crop_stride):
    if isinstance(batch, dict):
        image_data=batch['image']
    else:
        image_data=batch
    image_shape=image_data.shape #1,C,H,W
    xoffs = list(range(0, image_shape[-1] - crop_width, crop_stride))
    xoffs.append(image_shape[-1] - crop_width)
    xoffs=sorted(list(set(xoffs)))
    yoffs = list(range(0, image_shape[-2] - crop_height, crop_stride))
    yoffs.append(image_shape[-2] - crop_height)
    yoffs=sorted(list(set(yoffs)))

    for xoff in xoffs:
        for yoff in yoffs:
            crop_image_data = image_data[..., yoff:yoff + crop_height, xoff:xoff + crop_width]

            if torch.all(crop_image_data == 0):
                continue
            yield crop_image_data,xoff,yoff
    
def normal_sampler_with_dsm(batch, crop_width, crop_height, crop_stride):
    """
    支持 batch 中同时包含 image + dsm 的 sampler
    返回拼接后的多通道 crop
    """

    # -------- 1. 解析 batch --------
    if isinstance(batch, dict):
        image_data = batch["image"]      # [1, C_img, H, W]
        dsm_data   = batch["dsm"]        # [1, 1, H, W]
    else:
        # 兼容 tuple / list
        # 约定: batch = (image, dsm) 或 [image, dsm]
        image_data, dsm_data = batch[0], batch[1]

    if len(image_data) == 4:
        _,_, H, W = image_data.shape
    else:
        _, H, W = image_data.shape

    # -------- 2. 计算滑窗偏移 --------
    xoffs = list(range(0, W - crop_width, crop_stride))
    xoffs.append(W - crop_width)
    xoffs = sorted(set(xoffs))

    yoffs = list(range(0, H - crop_height, crop_stride))
    yoffs.append(H - crop_height)
    yoffs = sorted(set(yoffs))

    # -------- 3. 逐块裁剪 --------
    for xoff in xoffs:
        for yoff in yoffs:
            crop_img = image_data[..., yoff:yoff + crop_height,
                                   xoff:xoff + crop_width]
            crop_dsm = dsm_data[..., yoff:yoff + crop_height,
                                 xoff:xoff + crop_width]

            if not isinstance(crop_dsm, torch.Tensor):
                crop_dsm = torch.from_numpy(crop_dsm)
            crop_dsm = crop_dsm.to(crop_img.device)
            yield crop_img, crop_dsm, xoff, yoff


def normal_stitch(pred_output, offsets, out_w, out_h, class_num):
    device = pred_output[0].device
    pred_sum = torch.zeros([class_num, out_h, out_w], dtype = torch.float32, device = device)
    count = torch.zeros([out_h, out_w], dtype = torch.uint8, device = device)
    crop_height = pred_output[0].shape[-2]
    crop_width = pred_output[0].shape[-1]
    for out_put, offset in zip(pred_output, offsets):
        pred_sum[:, offset[1]:offset[1] + crop_height, offset[0]:offset[0] + crop_width]+= out_put.squeeze_(dim=0)
        count[offset[1]:offset[1] + crop_height, offset[0]:offset[0] + crop_width]+= 1
    count[count==0]=1
    pred_sum /= count
    # return pred_sum.numpy()
    return pred_sum

@torch.no_grad()
def streaming_stitch_step(model, image_data, sampler_info, threshold, device):
    """
    超大影像滑动窗口流式推理（不累积 prob_output）
    - 模型 prob 始终保持在 GPU
    - stitch 过程在 CPU 累积到 prob_map
    """

    # 输入影像尺寸
    _, _, H, W = image_data.shape  
    
    prob_map = None
    count_map = torch.zeros((H, W), dtype=torch.float32, device='cpu')

    # 遍历滑窗
    for input_patch, xoff, yoff in normal_sampler(
        image_data, sampler_info[0], sampler_info[1], sampler_info[2]
    ):
        # 模型推理（GPU）
        prob = model(input_patch.to(device))

        out_class = prob.shape[1]
        if out_class > 1:
            prob = torch.softmax(prob, dim=1)
        else:
            prob = torch.sigmoid(prob)

        # 将 prob 转到 CPU（马上释放 GPU）
        prob = prob.squeeze(0).cpu()     # shape: (C, ph, pw)

        ph, pw = prob.shape[-2], prob.shape[-1]

        # 初始化 prob_map 结构
        if prob_map is None:
            prob_map = torch.zeros((out_class, H, W), dtype=torch.float32, device='cpu')

        # 写入 prob_map（累加）
        prob_map[:, yoff:yoff+ph, xoff:xoff+pw] += prob
        count_map[yoff:yoff+ph, xoff:xoff+pw] += 1.0

        del prob
        torch.cuda.empty_cache()

    # 平均化：除 count_map（避免空值）
    count_map[count_map == 0] = 1.0
    prob_map = prob_map / count_map.unsqueeze(0)

    # 得到最终分类图
    if out_class > 1:
        out_class_map = torch.argmax(prob_map, dim=0)
    else:
        out_class_map = (prob_map[0] > threshold).to(torch.int64)

    return prob_map, out_class_map


def stitch_neck_features_dict(
    all_neck_feats,
    image_shape,
    strides_dict,
):
    """
    Args:
        all_neck_feats: List[{
            "offset": (x0, y0),
            "feat": Dict[str, Tensor[C, h, w]]
        }]
        image_shape: (H, W) of original image
        strides_dict: Dict[str, int], e.g. {"res2":4, ...}

    Returns:
        full_neck_feats: Dict[str, Tensor[C, H_s, W_s]]
    """

    H, W = image_shape

    # ===============================
    # 1. 初始化 full feature + count
    # ===============================
    full_feats = {}
    count_maps = {}

    for level, stride in strides_dict.items():
        # 找一个样本确定 C
        sample_feat = all_neck_feats[0]["feat"][level]
        C = sample_feat.shape[0]

        Hs = (H + stride - 1) // stride
        Ws = (W + stride - 1) // stride

        full_feats[level] = torch.zeros((C, Hs, Ws), dtype=sample_feat.dtype)
        count_maps[level] = torch.zeros(
            (1, Hs, Ws),
            dtype=sample_feat.dtype,   # ⭐ 跟 feature 一致（fp16）
        )

    # ===============================
    # 2. 累加所有 patch 特征
    # ===============================
    for item in all_neck_feats:
        x0, y0 = item["offset"]
        feats = item["feat"]

        for level, feat in feats.items():
            stride = strides_dict[level]

            _, h, w = feat.shape

            # 原图像素 → 特征图坐标
            fx0 = x0 // stride
            fy0 = y0 // stride
            fx1 = fx0 + w
            fy1 = fy0 + h

            # 安全裁剪（防止越界）
            fx1 = min(fx1, full_feats[level].shape[2])
            fy1 = min(fy1, full_feats[level].shape[1])

            valid_w = fx1 - fx0
            valid_h = fy1 - fy0

            if valid_w <= 0 or valid_h <= 0:
                continue

            full_feats[level][:, fy0:fy1, fx0:fx1] += feat[:, :valid_h, :valid_w]
            count_maps[level][:, fy0:fy1, fx0:fx1] += 1.0

    # ===============================
    # 3. 重叠区域取平均
    # ===============================
    for level in full_feats:
        count = count_maps[level]
        count[count == 0] = 1.0  # 防止除零
        full_feats[level] = full_feats[level] / count

    return full_feats
