# -*- coding: utf-8 -*-
import os
import cv2
import glob
import numpy as np
from utils.raster_io import *
from utils.basic_func import *
from scipy.ndimage.morphology import distance_transform_edt
# get_boundary_data 没有取影像尽头的边缘 ; binary_mask_to_edges_2d 取了影像尽头的边缘 ;


def get_boundary_data(label_data: np.ndarray, contour_group=None, buff_size=None) -> np.ndarray:
    r"""
    保存标签数据的边缘图像，
    :param label_data: 原图，必须为2维的np数组
    :param contour_group: 如果给出了轮廓，就不重新计算原图的边缘
    :param buff_size: 对边缘添加一定的缓冲区，如果为None，就不添加
    :return: 边缘图像
    """
    if label_data.ndim != 2:
        raise Exception('label_data.ndim != 2')
    if contour_group is None:  # if not given, calculate contour
        contour_group = get_cnt_group(label_data)
    else:
        if contour_group.ndim != 2:
            raise Exception('contour_group.ndim != 2')

    boundary_data = np.zeros_like(label_data, dtype=np.uint8)
    if contour_group is not None:  # if calculated contour
        boundary_data[contour_group[:, 1], contour_group[:, 0]] = 1

    if buff_size is not None and buff_size > 1:  # pad boundary with buffer
        boundary_data = cv2.dilate(boundary_data, kernel=np.ones((buff_size, buff_size), np.uint8))

    return boundary_data


def get_cnt_group(label_data: np.ndarray) -> np.ndarray or None:
    if label_data.ndim != 2:
        raise Exception('label_data.ndims != 2')
    #opencv4.0以下：
    # _, binary_contours, _ = cv2.findContours(label_data, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
   #opencv4.0以上 :
    binary_contours, _ = cv2.findContours(label_data, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if binary_contours is None:
        return binary_contours
    binary_contour_group = None
    for i, cnt in enumerate(binary_contours):  # [cnt,]
        if i == 0:
            binary_contour_group = cnt  # [pid, 1 ,2]
        else:
            binary_contour_group = np.concatenate((binary_contour_group, cnt), axis=0)
    return np.squeeze(binary_contour_group, axis=1)  # [pid, 2]


def save_label_boundary_with_buff(label_file, out_file, buff_size=None):
    label_data = RasterIO(label_file).read_as_array().astype('float32')
    if label_data.ndim != 2:
        raise Exception('label_data.ndim != 2')
    create_root_dir(out_file)
    boundary_data = get_boundary_data(label_data, contour_group=None, buff_size=buff_size)
    # save result
    fastly_save_image(image_data=boundary_data, out_file=out_file, ref_file=label_file, geo_trans=None, options=OPTIONS)


def save_label_boundary_dir(label_dir, out_dir, buff_size=None, img_type='tif'):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for label_file in glob.glob(label_dir + '/*.' + img_type):
        label_name = os.path.basename(label_file)
        out_file = os.path.join(out_dir, os.path.splitext(label_name)[0] + '.' + img_type)
        save_label_boundary_with_buff(label_file, out_file, buff_size=buff_size)


def binary_mask_to_edges_2d(mask: np.ndarray, radius: float = 1.5, buff_size: int = None) -> np.ndarray:
    r"""
    Converts a segmentation label mask (H,W) to a binary edge map (H,W)
    """
    assert mask.ndim == 2
    assert radius >= 0
    if buff_size is not None:
        assert buff_size > 1
    # We need to pad the borders for boundary conditions
    mask_pad = np.pad(mask, ((1, 1), (1, 1)), mode='constant', constant_values=0)
    dist_map = distance_transform_edt(mask_pad) + distance_transform_edt(1.0 - mask_pad)
    dist_map = dist_map[1:-1, 1:-1]
    dist_map[dist_map > radius] = 0
    edge_map = dist_map > 0
    if buff_size:  # pad boundary with buffer
        edge_map = cv2.dilate(edge_map, kernel=np.ones((buff_size, buff_size), np.uint8))
    return edge_map


def save_label_edge_2d(label_file, out_file, radius=1.5, buff_size=None):
    label_data = RasterIO(label_file).read_as_array().astype('float32')
    boundary_data = binary_mask_to_edges_2d(label_data, radius=radius, buff_size=buff_size)
    create_root_dir(out_file)
    fastly_save_image(image_data=boundary_data, out_file=out_file, ref_file=label_file, geo_trans=None, options=OPTIONS)


def save_label_edge_2d_dir(label_dir, out_dir, radius=1.5, buff_size=None, img_type='tif'):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for label_file in glob.glob(label_dir + '/*.' + img_type):
        label_name = os.path.basename(label_file)
        out_file = os.path.join(out_dir, os.path.splitext(label_name)[0] + '.' + img_type)
        save_label_edge_2d(label_file, out_file, radius=radius, buff_size=buff_size)


def get_edge_data(mask: np.ndarray, radius: float = 1.5):
    """
    得到边界的原图（不做缓冲的）
    :param mask:
    :param radius:
    :return:
    """
    assert mask.ndim == 2
    assert radius >= 0
    # We need to pad the borders for boundary conditions
    mask_pad = np.pad(mask, ((1, 1), (1, 1)), mode='constant', constant_values=0)  # 上下左右各填1行0
    dist_map = distance_transform_edt(mask_pad) + distance_transform_edt(1.0 - mask_pad)
    dist_map = dist_map[1:-1, 1:-1]
    dist_map[dist_map > radius] = 0
    edge_map = dist_map > 0
    edge_map = edge_map.astype("uint8")  # 膨胀必须是uint8
    return edge_map
