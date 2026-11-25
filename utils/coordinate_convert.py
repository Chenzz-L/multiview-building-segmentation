#!/conda/env python
# -*- coding:utf-8 -*-
import numpy as np

def geo_coordinate_to_img_coordinate(trans,geo_x,geo_y):
    left_top_x=trans[0]
    left_top_y=trans[3]
    x_resolution=trans[1]
    y_resolution=trans[5]
    if geo_x < left_top_x:
        return (-2,-2)
    else:
        geo_x = geo_x - left_top_x
    if geo_y > left_top_y:
        return (-2,-2)
    else:
        geo_y = left_top_y - geo_y
    if x_resolution == 0 or y_resolution == 0:
        return (-2,-2)
    if np.isnan(geo_x) or np.isnan(geo_y):
        return (-2,-2)
    else:
        img_x = int(geo_x / abs(x_resolution))
        img_y = int(geo_y / abs(y_resolution))

    return (img_x,img_y)

def geo_coordinate_to_img_coordinate2(trans,geo_x,geo_y):
    img_x = int((trans[5] * (geo_x - trans[0]) - trans[2] * (geo_y - trans[3])) / (
            trans[5] * trans[1]) + 0.5)
    img_y = int((trans[1] * (geo_y - trans[3]) - trans[4] * (geo_x - trans[0])) / (
            trans[5] * trans[1]) + 0.5)
    return (img_x, img_y)

def geo_coordinate_to_img_coordinate_list(trans,x_y_list):
    out_points=[]
    for x_y in x_y_list:
        out_points.append(geo_coordinate_to_img_coordinate2(trans,x_y[0],x_y[1]))
    return np.array(out_points)

def img_coordinate_to_geo_coordinate(trans,img_x,img_y):
    geo_x = trans[0] + img_x * trans[1] + 0.5
    geo_y = trans[3] + img_y * trans[5] - 0.5
    return (geo_x,geo_y)

def img_coordinate_to_geo_coordinate_list(trans,img_x_y_list):
    out_points = []
    for x_y in img_x_y_list:
        out_points.append(img_coordinate_to_geo_coordinate(trans, x_y[0], x_y[1]))
    return np.array(out_points)

def img_coordinate_to_geo_coordinate_dot_matri_version(trans,img_x_y):
    trans_mat=np.array([[trans[1],0,trans[0]+0.5],
                       [0,trans[5],trans[3]-0.5],
                       [0,0,1]])
    row_ones=np.ones(shape=(img_x_y.shape[0],1))
    img_x_y=np.column_stack((img_x_y, row_ones)).T

    geo_x_y=np.dot(trans_mat,img_x_y)[0:2,:].T
    return geo_x_y

def geo_coordinate_to_img_coordinate_dot_matri_version(trans,geo_x_y,return_mat=False):
    """
    使用矩阵进行坐标转换的计算
    Args:
        trans:仿射参数-----2*3
        geo_x_y:投影坐标-----2*n或3*n
        return_mat:是否返回dot的结果，否--返回与geo_x_y格式的T

    Returns:

    """
    trans_mat = np.array([[1/trans[1], 0, -trans[0]/trans[1]],
                          [0, 1/trans[5] , -trans[3]/trans[5]],
                          [0, 0, 1]])
    if trans_mat.shape[1]!=geo_x_y.shape[0]:
        row_ones = np.ones(shape=(geo_x_y.shape[0],1))
        geo_x_y = np.column_stack((geo_x_y, row_ones)).T
    if return_mat:
        img_x_y = np.round(np.dot(trans_mat, geo_x_y))
    else:
        img_x_y = np.round(np.dot(trans_mat, geo_x_y)[0:2, :].T)
    
    img_x_y=img_x_y.astype(np.int32)
    # img_x_y_int=np.zeros_like(img_x_y,dtype=np.int32)
    # img_x_y_int[:,:]=img_x_y
    return img_x_y
