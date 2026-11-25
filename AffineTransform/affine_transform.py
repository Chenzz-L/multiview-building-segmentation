import math
import sys
import random
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
import cv2
import numpy as np
from osgeo import gdal,ogr
import torch
import torch.nn.functional as F

from utils.crop_images import *
from utils.image_process import *
from utils.make_samples import *
from utils.raster_io import *
from utils.raster_process import *
from utils.shp_io import *


def cal_transformer_mat(from_points, to_points):
    """
    :param FromPoints:[[x,y],[x1,y1].....]
    :param ToPoints:[[x,y],[x1,y1].....]
    :return: 返回仿射变换矩阵
    """
    from_points = np.array(from_points)
    to_points = np.array(to_points)
    M, inliers = cv2.estimateAffine2D(from_points, to_points)
    return M

def cal_transformer_RANSAC(from_points, to_points,inner_threshold,valid_rate,confidence):
    point_len=len(from_points)
    valid_num=int(np.floor(point_len*valid_rate)) if int(np.floor(point_len*valid_rate))>=3 else 3
    once_rate=math.factorial(valid_num)/math.factorial(valid_num-3)
    once_rate/=math.factorial(point_len)/math.factorial(point_len-3)
    once_rate=1-once_rate
    iteration=int(np.ceil(math.log(1.0-confidence)/math.log(once_rate))) if once_rate!=0 else 1

    from_points = np.array(from_points)
    to_points = np.array(to_points)

    inner_num=-1
    residual_history=float('inf')
    affine_para=None

    for i in range(iteration):
        choice_index=random.sample(list(range(point_len)),valid_num)
        valid_from_points=from_points[choice_index,:]
        valid_to_points=to_points[choice_index,:]
        M=cal_transformer_mat(valid_from_points,valid_to_points)
        if M is None:
            continue
        residual=transformer_wrap(from_points,M).astype(np.float32)
        residual=np.sqrt(np.sum((residual-to_points)**2,axis=1))
        inner_index=np.argwhere(residual<inner_threshold)
        if len(inner_index)==len(to_points):
            return M
        residual=np.sum(residual)
        if residual<residual_history:
            affine_para=M.copy()
            inner_num=max(len(inner_index),inner_num)
            residual_history=residual
    return affine_para

def transformer_wrap(Points, TransformerMat):
    column = np.ones((Points.shape[0], 1))
    Points = np.column_stack((Points, column)).T
    mat=TransformerMat.copy()
    if mat.shape!=(3,3):
        mat = np.row_stack((mat, [0, 0, 1]))
    out = np.dot(mat, Points)
    out = out[0:2, :].T
    return out


def affine_with_dsm_points(lpstr_2d_dsm, lpstr_XY, lpstr_2dshp, lpstr_outshp):
    raster_io_2d_dsm = RasterIO(lpstr_2d_dsm)
    raster_io_XY = RasterIO(lpstr_XY)
    shp_io_2d = ShpIO(lpstr_2dshp)
    shp_io_out = ShpIO.create(lpstr_outshp)

    data_2d_dsm = raster_io_2d_dsm.read_as_array()
    data_XY = raster_io_XY.read_as_array()

    data_shp = raster_io_2d_dsm.shape()

    # 在内存中创建一个建筑物的dataset
    buidling_ds = RasterIO().create("", data_shp[1], data_shp[2], 1,
                                    options=["TILED=YES", "COMPRESS=DEFLATE"], gdal_type=gdal.GDT_Int32)
    buidling_ds.set_geo_transform(RasterIO.get_geo_transform(raster_io_2d_dsm))

    # 直接从像方矢量中栅格化，获取建筑物位置
    gdal.RasterizeLayer(buidling_ds, [1], shp_io_2d.get_layer(), options=["ATTRIBUTE=ID"])

    data_building = buidling_ds.read_as_array()

    # 为输出创建字段
    shp_io_out.create_fields({"a": ogr.OFTReal, "b": ogr.OFTReal, "c": ogr.OFTReal,
                              "d": ogr.OFTReal, "e": ogr.OFTReal, "f": ogr.OFTReal})

    print("Getting BuildingMap From Array....")
    building_map = get_building_map(data_building)
    print("Finishing Getting BuildingMap From Array....")

    for key, index_list in building_map:
        print(f"start building:{key}...")
        from_points = []
        to_points = []
        x_list = np.array(index_list) % data_shp[2]
        y_list = np.array(index_list) / data_shp[2]
        for x, y in zip(x_list, y_list):
            if data_2d_dsm[int(y), int(x)] < 0:
                continue
            from_points.append([x, y])
            to_points.append([data_XY[0, int(y), int(x)], data_XY[1, int(y), int(x)]])

        if from_points.count() < 20:
            continue

        transfromer_mat = cal_transformer_mat(from_points, to_points)
        if transfromer_mat is None:
            continue

        feature = shp_io_2d.get_feature(key - 1)


def generate_grid_point_each_building(points_pair, model_predict, geo_trans, predict_trans):
    """
         根据像点对生成grid，利用没一个建筑物内的所有有效的dsm点进行仿射变换参数估计，每一个房子的仿射变换参数不一样
         Returns:
    """

    affine_grid = np.zeros(shape=(model_predict.shape[0], model_predict.shape[1], 2), dtype=np.float32)
    affine_grid.fill(-2)

    # 获取每一个建筑物边界、等级结构、内部点
    contours_list, hierarchy_list, contours_in_list = get_countours(model_predict)

    # 定义opencv的坐标系(x,y)到[-1,-1]的转换矩阵
    coordiante_trans_mat = np.array([[2 / (model_predict.shape[1] - 1), 0, -1],
                                     [0, 2 / (model_predict.shape[0] - 1), -1],
                                     [0, 0, 1]
                                     ])

    p = np.zeros(shape=model_predict.shape, dtype=np.uint8)

    for i, zip_result in enumerate(zip(contours_list, hierarchy_list, contours_in_list)):
        contours, hierarchy, contours_inside = zip_result

        # 只针对外环进行处理
        if hierarchy[3] == -1:
            to_points = np.array(points_pair[:, contours_inside[:, 1], contours_inside[:, 0]])  # 3*n

            # 去除DSM无效的点
            index = to_points[0] - 0 > 1e-3
            to_points = to_points[1:, index].T  # m*2
            from_points = contours_inside[index, :]  # m*2
            from_points[:, 0] = from_points[:, 0] + predict_trans[0] * abs(predict_trans[1])
            from_points[:, 1] = from_points[:, 1] + predict_trans[3] * abs(predict_trans[5])
            if from_points.size == 0:
                continue
            affine_mat = cal_transformer_mat(from_points, to_points)  # 2*3

            if affine_mat is None or (True in np.isnan(affine_mat)):
                continue

            contours[:, :, 0] = contours[:, :, 0] + predict_trans[0] * abs(predict_trans[1])
            contours[:, :, 1] = contours[:, :, 1] + predict_trans[3] * abs(predict_trans[5])
            transform_rings = [contours]

            # 有内环的情况
            if hierarchy[2] != -1:
                j = i
                break_out = False
                while not break_out:
                    j += 1
                    if hierarchy_list[j][3] != i:
                        break_out = True
                    else:
                        contours_list[j][:, :, 0] = contours_list[j][:, :, 0] + predict_trans[0] * abs(predict_trans[1])
                        contours_list[j][:, :, 1] = contours_list[j][:, :, 1] + predict_trans[3] * abs(predict_trans[5])
                        transform_rings.append(contours_list[j])

            transform_out_rings = []
            # 将外环和内环的点投影至物方
            for ring in transform_rings:
                # 将像方建筑物环投影至物方---投影坐标
                out_points = transformer_wrap(ring.squeeze(axis=1), affine_mat)  # m*2
                # out_points[~np.isnan(out_points)]
                # 计算外环在物方图像上的像素坐标
                out_points = geo_coordinate_to_img_coordinate_list(geo_trans, out_points)
                out_points = np.expand_dims(out_points, axis=1)
                transform_out_rings.append(out_points)

            # 根据转换后的环计算在物方建筑物上的位置
            transform_out_contours_inside_YX = get_contours_in(transform_out_rings, model_predict.shape)  # n*2
            transform_out_contours_inside = transform_out_contours_inside_YX.copy()

            cv2.drawContours(p, transform_out_rings, -1, 255, 1)
            # p[transform_out_contours_inside[:,1],transform_out_contours_inside[:,0]]=255

            # 将物方像素坐标转到投影坐标上
            transform_out_contours_inside = img_coordinate_to_geo_coordinate_list(geo_trans,
                                                                                  transform_out_contours_inside)

            # 将投影坐标从X,Y----->Y,X
            # transform_out_contours_inside=transform_out_contours_inside[:,[1,0]]

            # 仿射参数是2*3的矩阵，需转为3*3的矩阵求解逆矩阵
            column = np.ones((transform_out_contours_inside.shape[0], 1))
            transform_out_contours_inside = np.column_stack((transform_out_contours_inside, column)).T
            # if i==61:
            #     print("ok")
            affine_mat = np.row_stack((affine_mat, [0, 0, 1]))
            affine_mat_inv = np.linalg.inv(affine_mat)

            # 物方的建筑物内部投影到像方的位置
            transform_out_2d = np.dot(affine_mat_inv, transform_out_contours_inside)  # 3*n

            transform_out_2d[0, :] = transform_out_2d[0, :] - predict_trans[0] * abs(predict_trans[1])
            transform_out_2d[1, :] = transform_out_2d[1, :] - predict_trans[3] * abs(predict_trans[5])

            # 归一化到[-1,1]的范围内
            transform_out_2d = np.dot(coordiante_trans_mat, transform_out_2d)
            transform_out_2d = transform_out_2d[0:2, :].T

            affine_grid[transform_out_contours_inside_YX[:, 1], transform_out_contours_inside_YX[:, 0],
            :] = transform_out_2d
    # fastly_save_image(p,r"/path/to/project/Users\35416\Desktop\test\22222.tif",ref_file=r"/path/to/project/working_man\ToGWX\data\Affine\Raster\CNN_MV_test\3dshp\3\binary\test_xyz_1.tif")
    return affine_grid


def generate_grid_point_each_building_V2(points_pair, model_predict, ref_shape, geo_trans, predict_trans):
    """
         根据像点对生成grid，利用没一个建筑物内的所有有效的dsm点进行仿射变换参数估计，每一个房子的仿射变换参数不一样,
         没有房子的地方使用最近房子的仿射变换参数将其投影至物方
         Returns:
    """
    if len(ref_shape) > 2:
        ref_shape = (ref_shape[1], ref_shape[2])
    affine_grid = np.zeros(shape=(ref_shape[0], ref_shape[1], 2), dtype=np.float32)
    affine_grid.fill(-2)
    model_predict = cv2.dilate(model_predict, kernel=np.ones((5, 5), np.uint8))
    # fastly_save_image(model_predict,r"/path/to/project/Users\35416\Desktop\test\x029517_test_20_morph.tif",
    #                   ref_file=r"/path/to/project/Xuzhou-Aerial\0_Xuzhou_DeepL\Xuzhou-IMG_MV\test\label\x029517_test_20.tif")
    # 获取每一个建筑物边界、等级结构、内部点
    contours_list, hierarchy_list, contours_in_list = get_countours(model_predict)

    # 定义opencv的坐标系(x,y)到[-1,-1]的转换矩阵
    coordiante_trans_mat = np.array([[2 / (model_predict.shape[1] - 1), 0, -1],
                                     [0, 2 / (model_predict.shape[0] - 1), -1],
                                     [0, 0, 1]
                                     ])

    p = np.zeros(shape=ref_shape, dtype=np.uint8)

    building_inv_affine = []
    building_centers = []  # [x,y]

    # cv.drawContours(p,contours_list,-1,255,1)
    # if os.path.exists(r"/path/to/project/Users\35416\Desktop\test\1234134.tif"):
    #     os.remove(r"/path/to/project/Users\35416\Desktop\test\1234134.tif")
    # fastly_save_image(p, r"/path/to/project/Users\35416\Desktop\test\1234134.tif",
    #                   ref_file=r"/path/to/project/Xuzhou-Aerial\0_Xuzhou_DeepL\Xuzhou-XYZ\test\label\test_xyz_4.tif")

    for i, zip_result in enumerate(zip(contours_list, hierarchy_list, contours_in_list)):
        contours, hierarchy, contours_inside = zip_result

        # 只针对外环进行处理
        if hierarchy[3] == -1:

            # 1.使用像方建筑物内所有点进行参数拟合
            to_points = np.array(points_pair[:, contours_inside[:, 1], contours_inside[:, 0]])  # 3*n

            # 2.去除DSM无效的点
            index = to_points[0] - 0 > 1e-3
            to_points = to_points[1:, index].T  # m*2
            from_points = contours_inside[index, :]  # m*2
            from_points[:, 0] = from_points[:, 0] + predict_trans[0] * abs(predict_trans[1])
            from_points[:, 1] = from_points[:, 1] + predict_trans[3] * abs(predict_trans[5])

            if from_points.size == 0:
                continue
            affine_mat = cal_transformer_mat(from_points, to_points)  # 2*3

            if affine_mat is None or (True in np.isnan(affine_mat)):
                continue

            # 3.将像方建筑物的环投到物方
            contours_ring = contours.copy()
            contours_ring[:, :, 0] = contours_ring[:, :, 0] + predict_trans[0] * abs(predict_trans[1])
            contours_ring[:, :, 1] = contours_ring[:, :, 1] + predict_trans[3] * abs(predict_trans[5])
            transform_rings = [contours_ring]

            # 有内环的情况
            if hierarchy[2] != -1:
                j = i
                break_out = False
                while not break_out:
                    j += 1
                    if j > len(hierarchy_list) - 1:
                        break
                    if hierarchy_list[j][3] != i:
                        break_out = True
                    else:
                        contours_list[j][:, :, 0] = contours_list[j][:, :, 0] + predict_trans[0] * abs(predict_trans[1])
                        contours_list[j][:, :, 1] = contours_list[j][:, :, 1] + predict_trans[3] * abs(predict_trans[5])
                        transform_rings.append(contours_list[j])

            transform_out_rings = []
            # 将外环和内环的点投影至物方
            for ring in transform_rings:
                # 将像方建筑物环投影至物方---投影坐标
                out_points = transformer_wrap(ring.squeeze(axis=1), affine_mat)  # m*2
                # out_points[~np.isnan(out_points)]
                # 计算外环在物方图像上的像素坐标
                out_points = geo_coordinate_to_img_coordinate_list(geo_trans, out_points)
                index = np.logical_and(out_points[:, 0] > 0, out_points[:, 0] < ref_shape[1])
                index2 = np.logical_and(out_points[:, 1] > 0, out_points[:, 1] < ref_shape[0])
                index = np.logical_and(index, index2)
                out_points = out_points[index, :]

                p[out_points[:, 1], out_points[:, 0]] = 255
                out_points = np.expand_dims(out_points, axis=1)
                transform_out_rings.append(out_points)

            # 4.获取转换后在物方上的建筑物内部点
            transform_out_contours_inside_XY = get_contours_in(transform_out_rings, ref_shape)  # n*2

            if transform_out_contours_inside_XY.size == 0:
                continue

            transform_out_contours_inside = transform_out_contours_inside_XY.copy()

            # 计算每一个建筑物的中心点
            building_centers.append(
                [np.mean(transform_out_contours_inside[:, 0]), np.mean(transform_out_contours_inside[:, 1])])

            # 将物方像素坐标转到投影坐标上
            transform_out_contours_inside = img_coordinate_to_geo_coordinate_list(geo_trans,
                                                                                  transform_out_contours_inside)
            # 将投影坐标从X,Y----->Y,X
            # transform_out_contours_inside=transform_out_contours_inside[:,[1,0]]

            # 仿射参数是2*3的矩阵，需转为3*3的矩阵求解逆矩阵
            column = np.ones((transform_out_contours_inside.shape[0], 1))
            transform_out_contours_inside = np.column_stack((transform_out_contours_inside, column)).T
            affine_mat = np.row_stack((affine_mat, [0, 0, 1]))
            affine_mat_inv = np.linalg.inv(affine_mat)

            building_inv_affine.append(affine_mat_inv)

            transform_out_2d = np.dot(affine_mat_inv, transform_out_contours_inside)  # 3*n

            transform_out_2d[0, :] = transform_out_2d[0, :] - predict_trans[0] * abs(predict_trans[1])
            transform_out_2d[1, :] = transform_out_2d[1, :] - predict_trans[3] * abs(predict_trans[5])

            # points=np.zeros_like(transform_out_2d,dtype=np.int64)
            # points[0,:]=np.rint(transform_out_2d[0,:])
            # points[1,:]=np.rint(transform_out_2d[1,:])
            # p[transform_out_contours_inside_XY[:,1],transform_out_contours_inside_XY[:,0]]=model_predict[points[1,:],points[0,:]]
            #
            # if os.path.exists(r"/path/to/project/Users\35416\Desktop\test\1234134.tif"):
            #     os.remove(r"/path/to/project/Users\35416\Desktop\test\1234134.tif")
            # fastly_save_image(p, r"/path/to/project/Users\35416\Desktop\test\1234134.tif",
            #                   ref_file=r"/path/to/project/Xuzhou-Aerial\0_Xuzhou_DeepL\Xuzhou-XYZ\test\label\test_xyz_4.tif")
            # 归一化到[-1,1]的范围内
            transform_out_2d = np.dot(coordiante_trans_mat, transform_out_2d)
            transform_out_2d = transform_out_2d[0:2, :].T

            affine_grid[transform_out_contours_inside_XY[:, 1], transform_out_contours_inside_XY[:, 0],
            :] = transform_out_2d
    # fastly_save_image(p,r"/path/to/project/Users\35416\Desktop\test\1234134.tif",
    #                   ref_file=r"/path/to/project/Xuzhou-Aerial\0_Xuzhou_DeepL\Xuzhou-XYZ\test\label\test_xyz_1.tif")

    # not_match_indexs=np.where(affine_grid[:,:,0]<-1)
    # not_match_indexs=np.array(list(zip(not_match_indexs[1],not_match_indexs[0])))

    for y in range(ref_shape[0]):
        for x in range(ref_shape[1]):
            if -1 > affine_grid[y, x, 0] or affine_grid[y, x, 0] > 1:
                distance = float("inf")
                match_index = -1
                for i, building_center in enumerate(building_centers):
                    distance_current = math.pow(x - building_center[0], 2) + math.pow(y - building_center[1], 2)
                    if distance_current < distance:
                        distance = distance_current
                        match_index = i

                geo_x_y = img_coordinate_to_geo_coordinate(geo_trans, x, y)
                column = np.array([[geo_x_y[0]], [geo_x_y[1]], [1]])
                affine_out = np.dot(building_inv_affine[match_index], column)
                affine_out[0, :] = affine_out[0, :] - predict_trans[0] * abs(predict_trans[1])
                affine_out[1, :] = affine_out[1, :] - predict_trans[3] * abs(predict_trans[5])
                affine_out = np.dot(coordiante_trans_mat, affine_out)
                affine_out = affine_out[0:2, :].T
                affine_grid[y, x, :] = affine_out

    return affine_grid


def generate_grid_point_all_image(points_pair, model_predict, ref_shape, geo_trans, predict_trans,none_data=0):
    """
         根据像点对生成grid，使用整张图内有效的dsm进行仿射变换参数的拟合，并应用于整张图的每一个像点
         Returns:
    """
    if len(ref_shape) > 2:
        ref_shape = (ref_shape[1], ref_shape[2])

    affine_grid = np.zeros(shape=(ref_shape[0], ref_shape[1], 2), dtype=np.float32)
    affine_grid.fill(-2)

    # 定义opencv的坐标系(x,y)到[-1,-1]的转换矩阵
    coordiante_trans_mat = np.array([[2 / (model_predict.shape[1] - 1), 0, -1],
                                     [0, 2 / (model_predict.shape[0] - 1), -1],
                                     [0, 0, 1]
                                     ])

    to_points = np.array(points_pair)  # 3*n
    # 去除DSM无效的点
    index = to_points[0] - none_data > 1e-3
    to_points = to_points[1:, index]  # m*2

    index2 = to_points[0] - none_data > 1e-3
    to_points = to_points[:, index2].T  # m*2

    x_y = generate_image_index(model_predict.shape)

    from_points = x_y[:, index]
    from_points = from_points[:, index2].T
    from_points[:, 0] = from_points[:, 0] + predict_trans[0] * abs(predict_trans[1])
    from_points[:, 1] = from_points[:, 1] + predict_trans[3] * abs(predict_trans[5])

    if from_points.shape[0]<2:
        return None

    affine_mat = cal_transformer_mat(from_points, to_points)  # 2*3

    x_y = generate_image_index(ref_shape)
    x_y = np.reshape(x_y, newshape=(2, -1)).T

    geo_coordintaes = img_coordinate_to_geo_coordinate_dot_matri_version(geo_trans, x_y)
    column = np.ones((geo_coordintaes.shape[0], 1))
    geo_coordintaes = np.column_stack((geo_coordintaes, column)).T

    affine_mat = np.row_stack((affine_mat, [0, 0, 1]))
    affine_mat_inv = np.linalg.inv(affine_mat)
    transform_out_2d = np.dot(affine_mat_inv, geo_coordintaes)  # 3*n
    transform_out_2d[0, :] = transform_out_2d[0, :] - predict_trans[0] * abs(predict_trans[1])
    transform_out_2d[1, :] = transform_out_2d[1, :] - predict_trans[3] * abs(predict_trans[5])
    # 归一化到[-1,1]的范围内
    transform_out_2d = np.dot(coordiante_trans_mat, transform_out_2d)
    transform_out_2d = transform_out_2d[0:2, :].T
    transform_out_2d = np.reshape(transform_out_2d, newshape=(ref_shape[0], ref_shape[1], 2))
    affine_grid[:, :, :] = transform_out_2d
    return affine_grid

def generate_grid_point_all_image_dir(points_pair_dir, model_predict_dir, dsm_2d_dir, pair_file, ref_dir,out_dir,none_data=0,symbol=None):
    with open(pair_file, "r") as f:
        scans = f.readlines()
        scans = [line.replace("\t", " ").strip().split(" ") for line in scans]
        scans = np.array(scans)
        scans_dict = dict(zip(scans[:, 1], scans[:, 0]))

    model_predict_files = glob.glob(os.path.join(model_predict_dir, "*.tif"))
    for model_predict_file in model_predict_files:
        name = os.path.basename(model_predict_file)
        if symbol!=None and not symbol in name :
            continue
        out_file = os.path.join(out_dir, name)
        if os.path.exists(out_file):
            print(f"file: {out_file} is exsited")
            continue

        # name="_".join(name.split(".")[0].split("_")[0:-1])+".tif"
        num = None
        i=-1
        while i:
            tnum=name.split(".")[0].split("_")[i]
            if tnum.isnumeric():
                num=tnum
                break
            i-=1

        predict_raster_io = RasterIO(model_predict_file)
        predict_trans = predict_raster_io.get_geo_trans()
        model_predict = predict_raster_io.read_as_array()

        dsm_2d_file = os.path.join(dsm_2d_dir, name)
        dsm_2d_raster_io = RasterIO(dsm_2d_file)
        dsm_data = dsm_2d_raster_io.read_as_array()
        dsm_data = np.expand_dims(dsm_data, axis=0)

        XY_raster_file = os.path.join(points_pair_dir, name)
        XY_raster_io = RasterIO(XY_raster_file)
        XY_data = XY_raster_io.read_as_array()

        point_pair = np.concatenate([dsm_data, XY_data], axis=0)

        ref_file = os.path.join(ref_dir, scans_dict[num] + ".tif")
        ref_raster_io = RasterIO(ref_file)
        ref_trans = ref_raster_io.get_geo_trans()
        ref_shape = ref_raster_io.shape()
        grid = generate_grid_point_all_image(point_pair, model_predict, ref_shape,ref_trans, predict_trans,none_data)
        if grid is None:
            continue

        grid = grid.transpose([2, 0, 1])

        fastly_save_image(grid, out_file=out_file,
                          gdal_type=gdal.GDT_Float32,
                          ref_file=ref_file)
        print(f'finish file: {name}')

def generate_grid_point_each_building_V3(points_pair, model_predict, ref_shape, geo_trans, predict_trans):
    """
         根据像点对生成grid，利用没一个建筑物内的所有有效的dsm点进行仿射变换参数估计，每一个房子的仿射变换参数不一样,
         没有房子的地方使用全图的仿射变换参数
         Returns:
    """
    if len(ref_shape) > 2:
        ref_shape = (ref_shape[1], ref_shape[2])
    affine_grid = np.zeros(shape=(ref_shape[0], ref_shape[1], 2), dtype=np.float32)
    affine_grid.fill(-2)
    model_predict = cv2.dilate(model_predict, kernel=np.ones((5, 5), np.uint8))
    # fastly_save_image(model_predict,r"/path/to/project/Users\35416\Desktop\test\x029517_test_20_morph.tif",
    #                   ref_file=r"/path/to/project/Xuzhou-Aerial\0_Xuzhou_DeepL\Xuzhou-IMG_MV\test\label\x029517_test_20.tif")
    # 获取每一个建筑物边界、等级结构、内部点
    contours_list, hierarchy_list, contours_in_list = get_countours(model_predict)

    # 定义opencv的坐标系(x,y)到[-1,-1]的转换矩阵
    coordiante_trans_mat = np.array([[2 / (model_predict.shape[1] - 1), 0, -1],
                                     [0, 2 / (model_predict.shape[0] - 1), -1],
                                     [0, 0, 1]
                                     ])

    p = np.zeros(shape=ref_shape, dtype=np.uint8)

    building_inv_affine = []
    building_centers = []  # [x,y]

    # cv.drawContours(p,contours_list,-1,255,1)
    # if os.path.exists(r"/path/to/project/Users\35416\Desktop\test\1234134.tif"):
    #     os.remove(r"/path/to/project/Users\35416\Desktop\test\1234134.tif")
    # fastly_save_image(p, r"/path/to/project/Users\35416\Desktop\test\1234134.tif",
    #                   ref_file=r"/path/to/project/Xuzhou-Aerial\0_Xuzhou_DeepL\Xuzhou-XYZ\test\label\test_xyz_4.tif")

    for i, zip_result in enumerate(zip(contours_list, hierarchy_list, contours_in_list)):
        contours, hierarchy, contours_inside = zip_result
        # 只针对外环进行处理
        if hierarchy[3] == -1:

            # 1.使用像方建筑物内所有点进行参数拟合
            to_points = np.array(points_pair[:, contours_inside[:, 1], contours_inside[:, 0]])  # 3*n

            # 2.去除DSM无效的点
            index = to_points[0] - 0 > 1e-3
            to_points = to_points[1:, index].T  # m*2
            from_points = contours_inside[index, :]  # m*2
            from_points[:, 0] = from_points[:, 0] + predict_trans[0] * abs(predict_trans[1])
            from_points[:, 1] = from_points[:, 1] + predict_trans[3] * abs(predict_trans[5])

            if from_points.size == 0:
                continue
            affine_mat = cal_transformer_mat(from_points, to_points)  # 2*3

            if affine_mat is None or (True in np.isnan(affine_mat)):
                continue

            # 3.将像方建筑物的环投到物方
            contours_ring = contours.copy()
            contours_ring[:, :, 0] = contours_ring[:, :, 0] + predict_trans[0] * abs(predict_trans[1])
            contours_ring[:, :, 1] = contours_ring[:, :, 1] + predict_trans[3] * abs(predict_trans[5])
            transform_rings = [contours_ring]

            # 有内环的情况
            if hierarchy[2] != -1:
                j = i
                break_out = False
                while not break_out:
                    j += 1
                    if j > len(hierarchy_list) - 1:
                        break
                    if hierarchy_list[j][3] != i:
                        break_out = True
                    else:
                        contours_list[j][:, :, 0] = contours_list[j][:, :, 0] + predict_trans[0] * abs(predict_trans[1])
                        contours_list[j][:, :, 1] = contours_list[j][:, :, 1] + predict_trans[3] * abs(predict_trans[5])
                        transform_rings.append(contours_list[j])

            transform_out_rings = []
            # 将外环和内环的点投影至物方
            for ring in transform_rings:
                # 将像方建筑物环投影至物方---投影坐标
                out_points = transformer_wrap(ring.squeeze(axis=1), affine_mat)  # m*2
                # out_points[~np.isnan(out_points)]
                # 计算外环在物方图像上的像素坐标
                out_points = geo_coordinate_to_img_coordinate_list(geo_trans, out_points)
                index = np.logical_and(out_points[:, 0] > 0, out_points[:, 0] < ref_shape[1])
                index2 = np.logical_and(out_points[:, 1] > 0, out_points[:, 1] < ref_shape[0])
                index = np.logical_and(index, index2)
                out_points = out_points[index, :]

                p[out_points[:, 1], out_points[:, 0]] = 255
                out_points = np.expand_dims(out_points, axis=1)
                transform_out_rings.append(out_points)

            # 4.获取转换后在物方上的建筑物内部点
            transform_out_contours_inside_XY = get_contours_in(transform_out_rings, ref_shape)  # n*2

            if transform_out_contours_inside_XY.size == 0:
                continue

            transform_out_contours_inside = transform_out_contours_inside_XY.copy()

            # 计算每一个建筑物的中心点
            building_centers.append(
                [np.mean(transform_out_contours_inside[:, 0]), np.mean(transform_out_contours_inside[:, 1])])

            # 将物方像素坐标转到投影坐标上
            transform_out_contours_inside = img_coordinate_to_geo_coordinate_list(geo_trans,
                                                                                  transform_out_contours_inside)
            # 将投影坐标从X,Y----->Y,X
            # transform_out_contours_inside=transform_out_contours_inside[:,[1,0]]

            # 仿射参数是2*3的矩阵，需转为3*3的矩阵求解逆矩阵
            column = np.ones((transform_out_contours_inside.shape[0], 1))
            transform_out_contours_inside = np.column_stack((transform_out_contours_inside, column)).T
            affine_mat = np.row_stack((affine_mat, [0, 0, 1]))
            affine_mat_inv = np.linalg.inv(affine_mat)

            building_inv_affine.append(affine_mat_inv)

            transform_out_2d = np.dot(affine_mat_inv, transform_out_contours_inside)  # 3*n

            transform_out_2d[0, :] = transform_out_2d[0, :] - predict_trans[0] * abs(predict_trans[1])
            transform_out_2d[1, :] = transform_out_2d[1, :] - predict_trans[3] * abs(predict_trans[5])

            # points=np.zeros_like(transform_out_2d,dtype=np.int64)
            # points[0,:]=np.rint(transform_out_2d[0,:])
            # points[1,:]=np.rint(transform_out_2d[1,:])
            # p[transform_out_contours_inside_XY[:,1],transform_out_contours_inside_XY[:,0]]=model_predict[points[1,:],points[0,:]]
            #
            # if os.path.exists(r"/path/to/project/Users\35416\Desktop\test\1234134.tif"):
            #     os.remove(r"/path/to/project/Users\35416\Desktop\test\1234134.tif")
            # fastly_save_image(p, r"/path/to/project/Users\35416\Desktop\test\1234134.tif",
            #                   ref_file=r"/path/to/project/Xuzhou-Aerial\0_Xuzhou_DeepL\Xuzhou-XYZ\test\label\test_xyz_4.tif")
            # 归一化到[-1,1]的范围内
            transform_out_2d = np.dot(coordiante_trans_mat, transform_out_2d)
            transform_out_2d = transform_out_2d[0:2, :].T

            affine_grid[transform_out_contours_inside_XY[:, 1], transform_out_contours_inside_XY[:, 0],
            :] = transform_out_2d

    # 计算全图的仿射变换参数
    to_points = np.array(points_pair)  # 3*n
    # 去除DSM无效的点
    index = to_points[0] - 0 > 1e-3
    to_points = to_points[1:, index]  # m*2

    index2 = to_points[0] - 0 > 1e-3
    to_points = to_points[:, index2].T  # m*2

    x_y = generate_image_index(model_predict.shape)

    from_points = x_y[:, index]
    from_points = from_points[:, index2].T
    from_points[:, 0] = from_points[:, 0] + predict_trans[0] * abs(predict_trans[1])
    from_points[:, 1] = from_points[:, 1] + predict_trans[3] * abs(predict_trans[5])

    affine_mat = cal_transformer_mat(from_points, to_points)  # 2*3

    x_y = generate_image_index(ref_shape)
    x_y = np.reshape(x_y, newshape=(2, -1)).T

    geo_coordintaes = img_coordinate_to_geo_coordinate_list(geo_trans, x_y)
    column = np.ones((geo_coordintaes.shape[0], 1))
    geo_coordintaes = np.column_stack((geo_coordintaes, column)).T

    affine_mat = np.row_stack((affine_mat, [0, 0, 1]))
    affine_mat_inv = np.linalg.inv(affine_mat)
    transform_out_2d = np.dot(affine_mat_inv, geo_coordintaes)  # 3*n
    transform_out_2d[0, :] = transform_out_2d[0, :] - predict_trans[0] * abs(predict_trans[1])
    transform_out_2d[1, :] = transform_out_2d[1, :] - predict_trans[3] * abs(predict_trans[5])
    # 归一化到[-1,1]的范围内
    transform_out_2d = np.dot(coordiante_trans_mat, transform_out_2d)
    transform_out_2d = transform_out_2d[0:2, :].T
    transform_out_2d = np.reshape(transform_out_2d, newshape=(ref_shape[0], ref_shape[1], 2))

    index = affine_grid[:, :, 0] < -1
    affine_grid[index, :] = transform_out_2d[index, :]

    return affine_grid

# 以下是新版本，对上述的函数进行了解耦
def generate_grid(points_pair, model_predict, ref_shape, geo_trans, predict_trans, expand_bounary = False, RANSAC = False, none_data = 0):
    """

    Args:
        points_pair:像点对数据：numpy.narray
        model_predict: 模型预测结果：numpy.narray
        ref_shape:物方图像大小：[int,int]
        geo_trans:物方仿射变换参数
        predict_trans:像方反射变换参数
        expand_bounary:是否扩大边界

    Returns:
        affine_grid：grid的结果
        building_inv_affine；每个房子拟合的仿射变换参数的结果[narray]
        building_centers:每个房子的中心坐标
    """
    if len(ref_shape) > 2:
        ref_shape = (ref_shape[-2], ref_shape[-1])

    affine_grid = np.zeros(shape=(ref_shape[0], ref_shape[1], 2), dtype=np.float32)
    affine_grid.fill(-2)
    # model_predict = cv2.dilate(model_predict, kernel=np.ones((5, 5), np.uint8))

    # 获取每一个建筑物边界、等级结构、内部点
    contours_list, hierarchy_list, contours_in_list = get_countours(model_predict)

    if contours_list == None:
        return None, None, None

    # 定义opencv的坐标系(x,y)到[-1,-1]的转换矩阵
    coordiante_trans_mat = np.array([[2 / (model_predict.shape[1] - 1), 0, -1],
                                     [0, 2 / (model_predict.shape[0] - 1), -1],
                                     [0, 0, 1]
                                     ])

    building_inv_affine = []
    building_centers = []  # [x,y]

    #################################### test
    # bounary_before=np.zeros(shape=(model_predict.shape[-2], model_predict.shape[-1]), dtype=np.uint8)
    # bounary_after=np.zeros(shape=(ref_shape[0], ref_shape[1]), dtype=np.uint8)
    # bounary_after_buffer=np.zeros(shape=(ref_shape[0], ref_shape[1]), dtype=np.uint8)

    for i, zip_result in enumerate(zip(contours_list, hierarchy_list, contours_in_list)):
        contours, hierarchy, contours_inside = zip_result

        # 只针对外环进行处理
        if hierarchy[3] == -1:

            # 1.使用像方建筑物内所有点进行参数拟合
            to_points = np.array(points_pair[:, contours_inside[:, 1], contours_inside[:, 0]])  # 3*n

            # 2.去除DSM无效的点
            index = to_points[0] - none_data > 1e-3
            to_points = to_points[1:, index].T  # m*2
            from_points = contours_inside[index, :]  # m*2
            # from_points[:, 0] = from_points[:, 0] + predict_trans[0] * abs(predict_trans[1])
            # from_points[:, 1] = from_points[:, 1] + predict_trans[3] * abs(predict_trans[5])

            if len(from_points)<3:
                continue
            affine_mat = cal_transformer_RANSAC(from_points,to_points,0.3,0.8,0.99) if RANSAC else cal_transformer_mat(from_points, to_points)  # 2*3

            if affine_mat is None or (True in np.isnan(affine_mat)):
                continue

            # 绘制像方边缘！！！！！！！！！！！！
            # cv2.drawContours(bounary_before, contours, -1, 255, 4)

            # 3.将像方建筑物的环投到物方
            # 3.1、(i,j)----->(geo_x,geo_y)
            contours_ring = contours.copy()
            # contours_ring[:, :, 0] = contours_ring[:, :, 0] + predict_trans[0] * abs(predict_trans[1])
            # contours_ring[:, :, 1] = contours_ring[:, :, 1] + predict_trans[3] * abs(predict_trans[5])
            transform_rings = [contours_ring]

            # 有内环的情况
            if hierarchy[2] != -1:
                j = i
                break_out = False
                while not break_out:
                    j += 1
                    if j > len(hierarchy_list) - 1:
                        break
                    if hierarchy_list[j][3] != i:
                        break_out = True
                    else:
                        # contours_list[j][:, :, 0] = contours_list[j][:, :, 0] + predict_trans[0] * abs(predict_trans[1])
                        # contours_list[j][:, :, 1] = contours_list[j][:, :, 1] + predict_trans[3] * abs(predict_trans[5])
                        transform_rings.append(contours_list[j])

            transform_out_rings = []
            # 将外环和内环的点投影至物方
            for ring in transform_rings:
                # 将像方建筑物环投影至物方---投影坐标
                out_points = transformer_wrap(ring.squeeze(axis=1), affine_mat)  # m*2
                # out_points[~np.isnan(out_points)]
                # 计算外环在物方图像上的像素坐标
                out_points = geo_coordinate_to_img_coordinate_dot_matri_version(geo_trans, out_points)

                index = np.logical_and(out_points[:, 0] > 0, out_points[:, 0] < ref_shape[1])
                index2 = np.logical_and(out_points[:, 1] > 0, out_points[:, 1] < ref_shape[0])
                index = np.logical_and(index, index2)
                out_points = out_points[index, :]

                if out_points.size == 0:
                    continue

                out_points = np.expand_dims(out_points, axis=1)
                transform_out_rings.append(out_points)


            # 绘制物方边缘！！！！！！！！！！！！
            # cv2.drawContours(bounary_after, transform_out_rings, -1, 255, 4)
            # cv2.drawContours(bounary_after_buffer, transform_out_rings, -1, 255, 20)
            # cv2.drawContours(bounary_after_buffer, transform_out_rings, -1, 255, -1)

            # 4.获取转换后在物方上的建筑物内部点
            transform_out_contours_inside_XY = get_contours_in(transform_out_rings, ref_shape, expand_bounary)  # n*2

            if transform_out_contours_inside_XY.size == 0:
                continue

            transform_out_contours_inside = transform_out_contours_inside_XY.copy()

            # 计算每一个建筑物的中心点
            building_centers.append(
                [np.mean(transform_out_contours_inside[:, 0]), np.mean(transform_out_contours_inside[:, 1])])

            # 将物方像素坐标转到投影坐标上
            transform_out_contours_inside = img_coordinate_to_geo_coordinate_dot_matri_version(geo_trans,
                                                                                               transform_out_contours_inside)

            # 仿射参数是2*3的矩阵，需转为3*3的矩阵求解逆矩阵
            column = np.ones((transform_out_contours_inside.shape[0], 1))
            transform_out_contours_inside = np.column_stack((transform_out_contours_inside, column)).T
            affine_mat = np.row_stack((affine_mat, [0, 0, 1]))
            try:
                affine_mat_inv = np.linalg.inv(affine_mat)
            except:
                print("矩阵不可逆")
            else:
                building_inv_affine.append(affine_mat_inv)

                transform_out_2d = np.dot(affine_mat_inv, transform_out_contours_inside)  # 3*n

                # transform_out_2d[0, :] = transform_out_2d[0, :] - predict_trans[0] * abs(predict_trans[1])
                # transform_out_2d[1, :] = transform_out_2d[1, :] - predict_trans[3] * abs(predict_trans[5])

                # 归一化到[-1,1]的范围内
                transform_out_2d = np.dot(coordiante_trans_mat, transform_out_2d)
                transform_out_2d = transform_out_2d[0:2, :].T

                affine_grid[transform_out_contours_inside_XY[:, 1], transform_out_contours_inside_XY[:, 0],
                :] = transform_out_2d

    ##### test
    # out_file=r'/path/to/user/NAS/gwx/experiment_result/Hrnet32_all_fusion/Hrnet32_Fusion_DSM_stage4_NONE/DSM_Hrnet32_All_Fusion_Dataset_DSM_Fusion_image_xyz_lr_5.00e-04/weights/draw_paper_pic2/xyz/boundary_draw'
    # if not os.path.exists(out_file):
    #     os.mkdir(out_file)
    # num=len(glob.glob(out_file+'/*.tif'))
    # # fastly_save_image(bounary_before,out_file=os.path.join(out_file,f'{num}_boundary.tif'),gdal_type=gdal.GDT_UInt16)
    # fastly_save_image(bounary_after,out_file=os.path.join(out_file,f'boundary.tif'),gdal_type=gdal.GDT_UInt16)

    ##### test
    return affine_grid, building_inv_affine, building_centers

def generate_grid_instance(
    points_pair,
    model_predict,
    ref_shape,
    geo_trans,
    predict_trans,
    expand_bounary=False,
    RANSAC=False,
    none_data=0
):
    """
    使用【实例分割结果】生成 grid
    每个实例单独拟合一套仿射参数

    Args:
        points_pair: (3, H, W) [DSM, X_img, Y_img]
        model_predict: (H, W), 0=背景, 1..N=实例ID
        ref_shape: (H_ref, W_ref) or (C,H,W)
        geo_trans: 物方仿射参数
        predict_trans: 像方仿射参数
        expand_bounary: 是否膨胀实例区域
        RANSAC: 是否使用 RANSAC
        none_data: DSM 无效值

    Returns:
        affine_grid: (H_ref, W_ref, 2)
        building_inv_affine: list of (3,3)
        building_centers: list of [x,y]
    """

    # -----------------------------
    # 0. 形状处理
    # -----------------------------
    if len(ref_shape) > 2:
        ref_shape = (ref_shape[-2], ref_shape[-1])

    H_ref, W_ref = ref_shape

    affine_grid = np.full((H_ref, W_ref, 2), -2, dtype=np.float32)

    # 像方 → [-1,1] 归一化矩阵
    coordiante_trans_mat = np.array([
        [2 / (model_predict.shape[1] - 1), 0, -1],
        [0, 2 / (model_predict.shape[0] - 1), -1],
        [0, 0, 1]
    ])

    building_inv_affine = []
    building_centers = []

    # -----------------------------
    # 1. 遍历实例 ID
    # -----------------------------
    instance_ids = np.unique(model_predict)
    instance_ids = instance_ids[instance_ids > 0]  # 去背景

    if len(instance_ids) == 0:
        return None, None, None

    for inst_id in instance_ids:

        instance_mask = (model_predict == inst_id)
        ys, xs = np.where(instance_mask)
        if xs.size < 3:
            continue

        contours_inside = np.column_stack([xs, ys])  # n*2

        # -----------------------------
        # 2. 提取实例内部点对
        # -----------------------------
        to_points = np.array(
            points_pair[:, contours_inside[:, 1], contours_inside[:, 0]]
        )  # 3*n

        # DSM 有效点
        valid = (to_points[0] - none_data) > 1e-3
        if np.sum(valid) < 3:
            continue

        from_points = contours_inside[valid]         # m*2
        to_points = to_points[1:, valid].T            # m*2

        # -----------------------------
        # 3. 拟合仿射（实例级）
        # -----------------------------
        if RANSAC:
            affine_mat = cal_transformer_RANSAC(
                from_points, to_points, 0.3, 0.8, 0.99
            )
        else:
            affine_mat = cal_transformer_mat(from_points, to_points)

        if affine_mat is None or np.any(np.isnan(affine_mat)):
            continue

        # -----------------------------
        # 4. 实例区域投影到物方
        # -----------------------------
        out_points = transformer_wrap(contours_inside, affine_mat)
        out_points = geo_coordinate_to_img_coordinate_dot_matri_version(
            geo_trans, out_points
        )

        out_points = np.round(out_points).astype(np.int32)

        valid = (
            (out_points[:, 0] >= 0) & (out_points[:, 0] < W_ref) &
            (out_points[:, 1] >= 0) & (out_points[:, 1] < H_ref)
        )
        out_points = out_points[valid]

        if out_points.size == 0:
            continue

        # -----------------------------
        # 5. 膨胀实例区域（可选）
        # -----------------------------
        if expand_bounary:
            mask = np.zeros((H_ref, W_ref), np.uint8)
            mask[out_points[:, 1], out_points[:, 0]] = 1
            mask = cv2.dilate(mask, np.ones((10, 10), np.uint8))
            ys2, xs2 = np.where(mask > 0)
            out_points = np.column_stack([xs2, ys2])

        # -----------------------------
        # 6. 实例中心
        # -----------------------------
        building_centers.append([
            float(np.mean(out_points[:, 0])),
            float(np.mean(out_points[:, 1]))
        ])

        # -----------------------------
        # 7. grid 反投影
        # -----------------------------
        geo_xy = img_coordinate_to_geo_coordinate_dot_matri_version(
            geo_trans, out_points
        )
        geo_xy = np.column_stack([geo_xy, np.ones(len(geo_xy))]).T

        affine_mat_h = np.row_stack([affine_mat, [0, 0, 1]])

        try:
            affine_mat_inv = np.linalg.inv(affine_mat_h)
        except np.linalg.LinAlgError:
            continue

        building_inv_affine.append(affine_mat_inv)

        transform_out = affine_mat_inv @ geo_xy
        transform_out = coordiante_trans_mat @ transform_out
        transform_out = transform_out[:2].T

        affine_grid[out_points[:, 1], out_points[:, 0], :] = transform_out

    return affine_grid, building_inv_affine, building_centers

def generate_grid_instance_v2(
    points_pair,
    model_predict,
    ref_shape,
    geo_trans,
    predict_trans,
    expand_bounary=False,
    RANSAC=False,
    none_data=0
):
    """
    行为 100% 对齐第一版 generate_grid
    仅区别在于：输入是实例 mask（ID），而不是语义 mask
    """

    if len(ref_shape) > 2:
        ref_shape = (ref_shape[-2], ref_shape[-1])

    H_ref, W_ref = ref_shape

    affine_grid = np.zeros((H_ref, W_ref, 2), dtype=np.float32)
    affine_grid.fill(-2)

    # 像方 → [-1,1]
    coordiante_trans_mat = np.array([
        [2 / (model_predict.shape[1] - 1), 0, -1],
        [0, 2 / (model_predict.shape[0] - 1), -1],
        [0, 0, 1]
    ])

    building_inv_affine = []
    building_centers = []

    instance_ids = np.unique(model_predict)
    instance_ids = instance_ids[instance_ids > 0]

    if len(instance_ids) == 0:
        return None, None, None

    # ==========================================================
    # 对每一个实例，完全走一遍「第一版 generate_grid」的流程
    # ==========================================================
    for inst_id in instance_ids:

        # ---------- 1. 构造“单实例语义 mask” ----------
        single_mask = np.zeros_like(model_predict, dtype=np.uint8)
        single_mask[model_predict == inst_id] = 1

        # ---------- 2. 使用第一版的轮廓提取 ----------
        contours_list, hierarchy_list, contours_in_list = get_countours(single_mask)

        if contours_list is None:
            continue

        # ---------- 3. 完全复用第一版的 for 循环 ----------
        for i, zip_result in enumerate(
            zip(contours_list, hierarchy_list, contours_in_list)
        ):
            contours, hierarchy, contours_inside = zip_result

            # 只处理外环（第一版原逻辑）
            if hierarchy[3] != -1:
                continue

            # ---------- 4. 像方内部点 → 物方点 ----------
            to_points = np.array(
                points_pair[:, contours_inside[:, 1], contours_inside[:, 0]]
            )

            # DSM 有效点
            index = to_points[0] - none_data > 1e-3
            to_points = to_points[1:, index].T
            from_points = contours_inside[index, :]

            if len(from_points) < 3:
                continue

            # ---------- 5. 仿射拟合 ----------
            affine_mat = (
                cal_transformer_RANSAC(from_points, to_points, 0.3, 0.8, 0.99)
                if RANSAC else
                cal_transformer_mat(from_points, to_points)
            )

            if affine_mat is None or np.any(np.isnan(affine_mat)):
                continue

            # ---------- 6. 处理外环 + 内环（完全一致） ----------
            contours_ring = contours.copy()
            transform_rings = [contours_ring]

            if hierarchy[2] != -1:
                j = i
                break_out = False
                while not break_out:
                    j += 1
                    if j > len(hierarchy_list) - 1:
                        break
                    if hierarchy_list[j][3] != i:
                        break_out = True
                    else:
                        transform_rings.append(contours_list[j])

            transform_out_rings = []

            for ring in transform_rings:
                out_points = transformer_wrap(ring.squeeze(axis=1), affine_mat)
                out_points = geo_coordinate_to_img_coordinate_dot_matri_version(
                    geo_trans, out_points
                )

                index_x = (out_points[:, 0] > 0) & (out_points[:, 0] < W_ref)
                index_y = (out_points[:, 1] > 0) & (out_points[:, 1] < H_ref)
                index = index_x & index_y
                out_points = out_points[index]

                if out_points.size == 0:
                    continue

                transform_out_rings.append(
                    np.expand_dims(out_points, axis=1)
                )

            if len(transform_out_rings) == 0:
                continue

            # ---------- 7. 物方内部点（第一版核心差异点） ----------
            transform_out_contours_inside_XY = get_contours_in(
                transform_out_rings,
                ref_shape,
                expand_bounary
            )

            if transform_out_contours_inside_XY.size == 0:
                continue

            # ---------- 8. 建筑中心 ----------
            building_centers.append([
                float(np.mean(transform_out_contours_inside_XY[:, 0])),
                float(np.mean(transform_out_contours_inside_XY[:, 1]))
            ])

            # ---------- 9. 反投影写 grid ----------
            geo_xy = img_coordinate_to_geo_coordinate_dot_matri_version(
                geo_trans,
                transform_out_contours_inside_XY
            )

            geo_xy = np.column_stack([geo_xy, np.ones(len(geo_xy))]).T

            affine_mat_h = np.row_stack([affine_mat, [0, 0, 1]])
            try:
                affine_mat_inv = np.linalg.inv(affine_mat_h)
            except np.linalg.LinAlgError:
                continue

            building_inv_affine.append(affine_mat_inv)

            transform_out = affine_mat_inv @ geo_xy
            transform_out = coordiante_trans_mat @ transform_out
            transform_out = transform_out[:2].T

            affine_grid[
                transform_out_contours_inside_XY[:, 1],
                transform_out_contours_inside_XY[:, 0],
                :
            ] = transform_out

    return affine_grid, building_inv_affine, building_centers




def generate_grid_with_near_building_affine(points_pair, model_predict, ref_shape, geo_trans, predict_trans):
    """
         根据像点对生成grid，利用没一个建筑物内的所有有效的dsm点进行仿射变换参数估计，每一个房子的仿射变换参数不一样,
         没有房子的地方使用最近房子的仿射变换参数将其投影至物方
         Returns:
    """
    if len(ref_shape) > 2:
        ref_shape = (ref_shape[1], ref_shape[2])

        # 定义opencv2的坐标系(x,y)到[-1,-1]的转换矩阵
    coordiante_trans_mat = np.array([[2 / (model_predict.shape[1] - 1), 0, -1],
                                     [0, 2 / (model_predict.shape[0] - 1), -1],
                                     [0, 0, 1]
                                     ])

    affine_grid, building_inv_affine, building_centers = generate_grid(points_pair, model_predict, ref_shape, geo_trans,
                                                                       predict_trans)
    for y in range(ref_shape[0]):
        for x in range(ref_shape[1]):
            if -1 > affine_grid[y, x, 0] or affine_grid[y, x, 0] > 1:
                distance = float("inf")
                match_index = -1
                for i, building_center in enumerate(building_centers):
                    distance_current = math.pow(x - building_center[0], 2) + math.pow(y - building_center[1], 2)
                    if distance_current < distance:
                        distance = distance_current
                        match_index = i

                geo_x_y = img_coordinate_to_geo_coordinate(geo_trans, x, y)
                column = np.array([[geo_x_y[0]], [geo_x_y[1]], [1]])
                affine_out = np.dot(building_inv_affine[match_index], column)
                affine_out[0, :] = affine_out[0, :] - predict_trans[0] * abs(predict_trans[1])
                affine_out[1, :] = affine_out[1, :] - predict_trans[3] * abs(predict_trans[5])
                affine_out = np.dot(coordiante_trans_mat, affine_out)
                affine_out = affine_out[0:2, :].T
                affine_grid[y, x, :] = affine_out

    return affine_grid

def generate_grid_only_building_otherwithmode(points_pair, model_predict, ref_shape, geo_trans,
                                           predict_trans, mode="GROUND", expand_bounary=False,RANSAC=False,none_data=0):
    """
    根据像点对生成grid，利用没一个建筑物内的所有有效的dsm点进行仿射变换参数估计，每一个房子的仿射变换参数不一样,
    没有房子的地方使用全图的仿射变换参数或者不处理
    Args:
        points_pair:物方像方像点对
        model_predict:模型预测结果
        ref_shape:物方参考图像的大小
        geo_trans:参考图像的仿射6参数
        predict_trans:模型预测结果的仿射6参数
        mode:建筑物以外的地方使用什么方式生成仿射变换参数----['NONE',"BUILDING", "GROUND", "ALL"]
        expand_bounary:是否在轮廓转换到物方之后进行边缘膨胀，膨胀10个像素

    Returns:

    """
    if len(ref_shape) > 2:
        ref_shape = (ref_shape[1], ref_shape[2])

        # 定义opencv的坐标系(x,y)到[-1,-1]的转换矩阵

    coordiante_trans_mat = np.array([[2 / (model_predict.shape[1] - 1), 0, -1],
                                     [0, 2 / (model_predict.shape[0] - 1), -1],
                                     [0, 0, 1]
                                     ])

    affine_grid, _, _ = generate_grid_instance_v2(points_pair, model_predict, ref_shape, geo_trans, predict_trans, expand_bounary,RANSAC,none_data)

    if affine_grid is None:
        return None

    if mode.upper()=="NONE":
        return affine_grid
    # 计算全图的仿射变换参数
    affine_mat = cal_affine_mat_with_ground_or_building(points_pair, model_predict, predict_trans, mode)
    x_y = generate_image_index(ref_shape)
    x_y = np.reshape(x_y, newshape=(2, -1)).T

    # 对非建筑物点使用全图的仿射变换参数
    geo_coordintaes = img_coordinate_to_geo_coordinate_dot_matri_version(geo_trans, x_y)
    column = np.ones((geo_coordintaes.shape[0], 1))
    geo_coordintaes = np.column_stack((geo_coordintaes, column)).T

    affine_mat = np.row_stack((affine_mat, [0, 0, 1]))
    affine_mat_inv = np.linalg.inv(affine_mat)
    transform_out_2d = np.dot(affine_mat_inv, geo_coordintaes)  # 3*n
    transform_out_2d[0, :] = transform_out_2d[0, :] - predict_trans[0] * abs(predict_trans[1])
    transform_out_2d[1, :] = transform_out_2d[1, :] - predict_trans[3] * abs(predict_trans[5])
    # 归一化到[-1,1]的范围内
    transform_out_2d = np.dot(coordiante_trans_mat, transform_out_2d)
    transform_out_2d = transform_out_2d[0:2, :].T
    transform_out_2d = np.reshape(transform_out_2d, newshape=(ref_shape[0], ref_shape[1], 2))

    index = affine_grid[:, :, 0] < -1
    affine_grid[index, :] = transform_out_2d[index, :]

    # tmax=np.max(affine_grid)
    # tmin=np.min(affine_grid)
    # print(tmax,tmin)
    return affine_grid

def generate_grid_with_all_building_affine_V2(points_pair, model_predict, ref_shape, geo_trans, predict_trans):
    """
         根据像点对生成grid，利用没一个建筑物内的所有有效的dsm点进行仿射变换参数估计，每一个房子的仿射变换参数不一样,
         房子的外接矩形内的点使用房子的仿射参数
         没有房子的地方使用全图的仿射变换参数
         Returns:
    """
    if len(ref_shape) > 2:
        ref_shape = (ref_shape[1], ref_shape[2])

        # 定义opencv的坐标系(x,y)到[-1,-1]的转换矩阵
    coordiante_trans_mat = np.array([[2 / (model_predict.shape[1] - 1), 0, -1],
                                     [0, 2 / (model_predict.shape[0] - 1), -1],
                                     [0, 0, 1]
                                     ])

    affine_grid, _, _ = generate_grid(points_pair, model_predict, ref_shape, geo_trans, predict_trans)

    model_predict = cv2.dilate(model_predict, kernel=np.ones((5, 5), np.uint8))
    p = np.zeros(shape=model_predict.shape, dtype=np.uint8)
    # 获取每一个建筑物边界、等级结构、内部点
    contours_list, hierarchy_list, contours_in_list = get_countours(model_predict, mode=cv2.RETR_EXTERNAL,
                                                                    method=cv2.CHAIN_APPROX_SIMPLE)

    for c in range(len(contours_list)):
        rect = cv2.minAreaRect(contours_list[c])
        cx, cy = rect[0]
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(p, [box], 0, 255, 2)

    fastly_save_image(p, r"/path/to/project/Users\35416\Desktop\test\near_building_affine2222.tif")
    cv2.imshow("img", p)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 计算全图的仿射变换参数
    to_points = np.array(points_pair)  # 3*n
    # 去除DSM无效的点
    index = to_points[0] - 0 > 1e-3
    to_points = to_points[1:, index]  # m*2

    index2 = to_points[0] - 0 > 1e-3
    to_points = to_points[:, index2].T  # m*2

    x_y = generate_image_index(model_predict.shape)

    from_points = x_y[:, index]
    from_points = from_points[:, index2].T
    from_points[:, 0] = from_points[:, 0] + predict_trans[0] * abs(predict_trans[1])
    from_points[:, 1] = from_points[:, 1] + predict_trans[3] * abs(predict_trans[5])

    affine_mat = cal_transformer_mat(from_points, to_points)  # 2*3

    x_y = generate_image_index(ref_shape)
    x_y = np.reshape(x_y, newshape=(2, -1)).T

    geo_coordintaes = img_coordinate_to_geo_coordinate_list(geo_trans, x_y)
    column = np.ones((geo_coordintaes.shape[0], 1))
    geo_coordintaes = np.column_stack((geo_coordintaes, column)).T

    affine_mat = np.row_stack((affine_mat, [0, 0, 1]))
    affine_mat_inv = np.linalg.inv(affine_mat)
    transform_out_2d = np.dot(affine_mat_inv, geo_coordintaes)  # 3*n
    transform_out_2d[0, :] = transform_out_2d[0, :] - predict_trans[0] * abs(predict_trans[1])
    transform_out_2d[1, :] = transform_out_2d[1, :] - predict_trans[3] * abs(predict_trans[5])
    # 归一化到[-1,1]的范围内
    transform_out_2d = np.dot(coordiante_trans_mat, transform_out_2d)
    transform_out_2d = transform_out_2d[0:2, :].T
    transform_out_2d = np.reshape(transform_out_2d, newshape=(ref_shape[0], ref_shape[1], 2))

    index = affine_grid[:, :, 0] < -1
    affine_grid[index, :] = transform_out_2d[index, :]

    return affine_grid


def cal_affine_mat_with_ground_or_building(points_pair, model_predict, predict_trans, mode="GROUND"):
    mode_list = ["BUILDING", "GROUND", "ALL"]
    if mode not in mode_list:
        raise ValueError("mode must in ['BUILDING','GROUND','ALL']")
    mode_index = None
    if mode == "BUILDING":
        mode_index = model_predict - 0.0 > 1e-3
    elif mode == "GROUND":
        mode_index = model_predict - 0.0 < 1e-3
    elif mode == "ALL":
        p = np.ones_like(model_predict)
        mode_index = p > 0

    # 计算全图的仿射变换参数
    to_points = np.array(points_pair)[:, mode_index]  # 3*n
    # 去除DSM无效的点
    index = to_points[0] - 0 > 1e-3
    to_points = to_points[1:, index]  # m*2

    index2 = to_points[0] - 0 > 1e-3
    to_points = to_points[:, index2].T  # m*2

    x_y = generate_image_index(model_predict.shape)[:, mode_index]

    from_points = x_y[:, index]
    from_points = from_points[:, index2].T
    from_points[:, 0] = from_points[:, 0] + predict_trans[0] * abs(predict_trans[1])
    from_points[:, 1] = from_points[:, 1] + predict_trans[3] * abs(predict_trans[5])

    affine_mat = cal_transformer_mat(from_points, to_points)  # 2*3

    return affine_mat


def devide_grid(shape, kernel_size=32):
    """
    为图像生成格网
    Args:
        shape: 图像尺寸
        kernel_size: 格网大小

    Returns:

    """
    xoffs = list(range(0, shape[1] - kernel_size, kernel_size))
    xoffs.append(shape[1] - kernel_size - kernel_size)

    yoffs = list(range(0, shape[0] - kernel_size, kernel_size))
    yoffs.append(shape[0] - kernel_size - kernel_size)

    return xoffs, yoffs


# def generate_grid_only_building_otherwithmode_dir(points_pair_dir, model_predict_dir, dsm_2d_dir, pair_file, ref_dir,
#                                                out_dir,mode="GROUND", expand_bounary=False,RANSAC=False,none_data=0,symbol=None):
#     """
#     Args:
#         points_pair_dir:像点对的路径
#         model_predict_dir:模型预测结果路径/label路径
#         dsm_2d_dir:像方DSM的路径
#         pair_file:像方和物方对应图像的文件
#         ref_dir:物方图像的路径
#         out_dir:输出路径
#         mode:使用全部地面点或建筑物点对没有房子的地方进行参数拟合
#         expand_bounary:是否扩大边界，在建筑物周围10像素内，使用建筑物的拟合参数进行转换
#         RANSAC=False : 是否使用ransac
#     Returns:

#     """

#     with open(pair_file, "r") as f:
#         scans = f.readlines()
#         scans = [line.replace("\t", " ").strip().split(" ") for line in scans]
#         scans = np.array(scans)
#         scans_dict = dict(zip(scans[:, 1], scans[:, 0]))

#     model_predict_files = glob.glob(os.path.join(model_predict_dir, "*.tif"))
#     for model_predict_file in model_predict_files:
#         name = os.path.basename(model_predict_file)
#         if symbol!=None and not symbol in name :
#             continue
#         # name="_".join(name.split(".")[0].split("_")[0:-1])+".tif"
#         num = None
#         i=-1
#         while i:
#             tnum=name.split(".")[0].split("_")[i]
#             if tnum.isnumeric():
#                 num=tnum
#                 break
#             i-=1

#         out_file = os.path.join(out_dir, name)
#         if os.path.exists(out_file):
#             print(f"file: {out_file} is exsited")
#             continue

#         predict_raster_io = RasterIO(model_predict_file)
#         predict_trans = predict_raster_io.get_geo_trans()
#         model_predict = predict_raster_io.read_as_array()
#         # if np.sum(model_predict)/(model_predict.shape[0]*model_predict.shape[1])<0.2:
#         #     continue
#         if np.sum(model_predict)==0:
#             continue
#         model_predict=np.array(model_predict*255,dtype=np.uint8)

#         dsm_2d_file = os.path.join(dsm_2d_dir, name)
#         dsm_2d_raster_io = RasterIO(dsm_2d_file)
#         dsm_data = dsm_2d_raster_io.read_as_array()
#         dsm_data = np.expand_dims(dsm_data, axis=0)

#         XY_raster_file = os.path.join(points_pair_dir, name)
#         XY_raster_io = RasterIO(XY_raster_file)
#         XY_data = XY_raster_io.read_as_array()

#         point_pair = np.concatenate([dsm_data, XY_data], axis=0)

#         ref_file = os.path.join(ref_dir, scans_dict[num] + ".tif")
#         ref_raster_io = RasterIO(ref_file)
#         ref_trans = ref_raster_io.get_geo_trans()
#         ref_shape = ref_raster_io.shape()
#         grid = generate_grid_only_building_otherwithmode(point_pair, model_predict, ref_shape,
#                                                       ref_trans, predict_trans, mode, expand_bounary,RANSAC,none_data)

#         grid = grid.transpose([2, 0, 1])
#         fastly_save_image(grid, out_file=out_file,
#                           gdal_type=gdal.GDT_Float32,
#                           ref_file=ref_file)
#         print(f'finish file: {name}')

def generate_grid_only_building_otherwithmode_dir(points_pair_dir, model_predict_dir, dsm_2d_dir, pair_file, ref_dir,
                                               out_dir,mode="GROUND", expand_bounary=False,RANSAC=False,none_data=0,symbol=None):
    """
    Args:
        points_pair_dir:像点对的路径
        model_predict_dir:模型预测结果路径/label路径
        dsm_2d_dir:像方DSM的路径
        pair_file:像方和物方对应图像的文件
        ref_dir:物方图像的路径
        out_dir:输出路径
        mode:使用全部地面点或建筑物点对没有房子的地方进行参数拟合
        expand_bounary:是否扩大边界，在建筑物周围10像素内，使用建筑物的拟合参数进行转换
        RANSAC=False : 是否使用ransac
    Returns:

    """

    with open(pair_file, "r") as f:
        scans = f.readlines()
        scans = [line.replace("\t", " ").strip().split(" ") for line in scans]
        scans = np.array(scans)
        scans_dict = dict(zip(scans[:, 1], scans[:, 0]))

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    model_predict_files = glob.glob(os.path.join(model_predict_dir, "*.tif"))+glob.glob(os.path.join(model_predict_dir, "*.tiff"))
    for model_predict_file in model_predict_files:
        name = os.path.basename(model_predict_file)
        out_file = os.path.join(out_dir, name)
        if os.path.exists(out_file):
            print(f"file: {out_file} is exsited")
            continue
        
        num=name.split("-")[-1].split(".")[0]
        ref_file = os.path.join(ref_dir, scans_dict[num]+".tif")
        ref_raster_io = RasterIO(ref_file)
        ref_trans = ref_raster_io.get_geo_trans()
        ref_shape = ref_raster_io.shape()
        
        predict_raster_io = RasterIO(model_predict_file)
        predict_trans = predict_raster_io.get_geo_trans()
        model_predict = predict_raster_io.read_as_array()
        # if np.sum(model_predict)/(model_predict.shape[0]*model_predict.shape[1])<0.2:
        #     continue
        if np.sum(model_predict)==0:
            continue
        model_predict=np.array(model_predict*255,dtype=np.uint8)

        dsm_2d_file = os.path.join(dsm_2d_dir, name)
        if not os.path.exists(dsm_2d_file):
            continue
        dsm_2d_raster_io = RasterIO(dsm_2d_file)
        dsm_data = dsm_2d_raster_io.read_as_array()
        dsm_data = np.expand_dims(dsm_data, axis=0)

        XY_raster_file = os.path.join(points_pair_dir, name)
        if not os.path.exists(XY_raster_file):
            continue
        XY_raster_io = RasterIO(XY_raster_file)
        XY_data = XY_raster_io.read_as_array()

        point_pair = np.concatenate([dsm_data, XY_data], axis=0)

        grid = generate_grid_only_building_otherwithmode(point_pair, model_predict, ref_shape,
                                                      ref_trans, predict_trans, mode, expand_bounary,RANSAC,none_data)

        grid = grid.transpose([2, 0, 1])
        fastly_save_image(grid, out_file=out_file,
                          gdal_type=gdal.GDT_Float32,
                          ref_file=ref_file)
        print(f'finish file: {name}')



###########################################
def transform_img_to_xyz(grid_file:str,img_file:str,xyz_ref:str,out_file:str,intput_label=False):
    img_raster_io = RasterIO(img_file)
    img_data = img_raster_io.read_as_array()
    prob = np.zeros_like(img_data, dtype=np.float32)
    prob[:, :] = img_data[:, :]
    img_data = prob
    if img_data.ndim<4:
        img_data = np.expand_dims(img_data, axis=list(range(4-img_data.ndim)))

    grid_raster_io = RasterIO(grid_file)
    grid = grid_raster_io.read_as_array().astype("float32")
    grid = np.expand_dims(grid, axis=0)
    grid.astype(np.float32)
    grid = np.transpose(grid, [0, 2, 3, 1])
    out = F.grid_sample(torch.tensor(img_data), torch.tensor(grid), align_corners=True)
    out = out.numpy().squeeze(axis=0)

    #grid_sample进行了重采样，会导致255的值小于255，需要做以下处理
    if intput_label:
        out[out>=125]=255
        out[out<125]=0

    fastly_save_image(out, out_file,ref_file=xyz_ref, gdal_type=gdal.GDT_Float32)
def transform_img_to_xyz_dir(grid_file_dir:str,img_file_dir:str,xyz_ref_dir:str,out_file_dir:str,xyz_img_map_file:str,intput_label=False):
    if not os.path.exists(out_file_dir):
        os.makedirs(out_file_dir)
    xyz_img_dict=read_xyz_img_pair_file(xyz_img_map_file)
    img_files=glob.glob(os.path.join(img_file_dir,"*.tif"))
    for img_file in img_files:
        name=os.path.basename(img_file)
        grid_file=os.path.join(grid_file_dir,name)
        if not os.path.exists(grid_file):
            continue
        xyz_num = os.path.splitext(name)[0].split('_')[-1]
        xyz_ref_file = os.path.join(xyz_ref_dir, xyz_img_dict[xyz_num]  + ".tif")

        out_file=os.path.join(out_file_dir,name)
        transform_img_to_xyz(grid_file,img_file,xyz_ref_file,out_file,intput_label)


#######################################
"""
    1、使用groundtruth的shp文件的角点拟合仿射变换参数
    2、将像方的groundtruth进行栅格化带有ID号，对相应的单个建筑物进行buffer
    3、单个建筑物的buffer有重叠的部分计算到最近的房子进行赋值
    4、将带有编号的建筑物的外轮廓投影使用对应的仿射参数投影到物方
    5、填充物方外轮廓并求逆，对建筑物内部进行插值
"""
def generate_grid_with_groundtruth(label_file2d,shpfile_3d):
    raster_io=RasterIO()
    raster_io.open(label_file2d)

    shp_io=ShpIO(shpfile_3d)
    id_affineparam_dict=shp_io.get_id_affineparam_dict("connectID")
    label_2d_data=raster_io.read_as_array().astype('int32')
    min_building_id=np.min(label_2d_data)
    max_building_id=np.max(label_2d_data)

    dilate_label_2d_binary_data=np.zeros_like(label_2d_data)
    dilate_label_2d_id_data=np.zeros_like(label_2d_data)

    center_x_y_dict={}
    #1、计算膨胀前的中心坐标以及对图像进行膨胀
    for i in range(min_building_id+1,max_building_id+1):
        once_building_data=np.zeros_like(label_2d_data,dtype=np.uint8)
        index=label_2d_data==i
        coordinate_index=np.where(label_2d_data==i)
        if np.sum(index)==0:
            print(f"building {i} is not exist skip it")
            continue
        once_building_data[coordinate_index]=1
        center_y=np.mean(coordinate_index[0])
        center_x=np.mean(coordinate_index[1])
        center_x_y_dict[str(i)]=[center_x,center_y]

        dilate_once_building_data=cv2.dilate(once_building_data,kernel=np.ones(shape=[3,3],dtype=np.uint8))
        dilate_label_2d_id_data[dilate_once_building_data>0]=i

        #只保留扩大的外围
        dilate_once_building_data-=once_building_data
        dilate_label_2d_binary_data+=dilate_once_building_data
        print(f"finish buidling {i}")

    #2、对重叠部分进行最近建筑物的赋值
    overlap_region=np.where(dilate_label_2d_binary_data>1)
    for x,y in zip(overlap_region[0],overlap_region[1]):
        distance=float('inf')
        matched_id=-1

        for key,val in center_x_y_dict.items():
            center_x,center_y=val
            t_dis=math.pow(x-center_x,2)+math.pow(y-center_y,2)
            if t_dis<distance:
                distance=t_dis
                matched_id=key

        dilate_label_2d_id_data[y,x]=matched_id


    #3、将每一个建筑物投影到物方
    #2、计算buffer重叠部分所属的建筑物
        # contours_list, hierarchy_list, contours_in_list = get_countours(once_building_data,expand_bounary=True)
        #
        #
        # once_building_data[index]=1
        # affine_param=id_affineparam_dict[str(i)]
        # affine_mat=np.array([affine_param[0:3],affine_param[3:6],[0,0,1]])
        #
        # inv_affine_mat=np.linalg.inv(affine_mat)

def generate_grid_with_groundtruth_v2(label_file2d,shpfile_3d,ref_file_3d):
    """
      不同于上一个版本，对像方进行buffer，buffer区域内，进行先来后到的计算
    Args:
        label_file2d:
        shpfile_3d:

    Returns:

    """
    raster_io=RasterIO()
    raster_io.open(label_file2d)
    shp_io=ShpIO(shpfile_3d)
    id_affineparam_dict=shp_io.get_id_affineparam_dict("connectID")

    label_2d_data=raster_io.read_as_array()
    label_2d_data[label_2d_data<0]=0
    label_2d_data.astype('int32')
    image_shape=raster_io.shape()
    trans_2d=raster_io.get_geo_trans()
    raster_io.open(ref_file_3d)
    geo_trans_3d=raster_io.get_geo_trans()
    ref_shape=raster_io.shape()[1:]

    t_label_2d_data=label_2d_data.copy()
    t_label_2d_data[t_label_2d_data==0]=99999
    min_building_id=np.min(t_label_2d_data)
    t_label_2d_data[t_label_2d_data == 99999] = 0
    max_building_id=np.max(t_label_2d_data)

    min_building_id=min_building_id+1 if min_building_id==0 else min_building_id
    del t_label_2d_data
    t_label_2d_data=None


    label_2d_binary_data=np.zeros_like(label_2d_data)
    label_2d_binary_data[label_2d_data>0]=1

    affine_grid = np.zeros(shape=(ref_shape[-2], ref_shape[-1], 2), dtype=np.float32)
    affine_grid.fill(-2)
    # 定义opencv的坐标系(x,y)到[-1,-1]的转换矩阵
    coordiante_trans_mat = np.array([[2 / (image_shape[-1] - 1), 0, -1],
                                     [0, 2 / (image_shape[-2] - 1), -1],
                                     [0, 0, 1]
                                     ])

    ttt = np.zeros(shape=(ref_shape[-2], ref_shape[-1]))
    # ttt = np.zeros_like(label_2d_data)

    #1、对单个建筑物进行膨胀，先来后到的方式
    for k in range(int(min_building_id),int(max_building_id)+1):
        once_building_data=np.zeros_like(label_2d_data,dtype=np.uint8)
        index=label_2d_data==k
        coordinate_index=np.where(label_2d_data==k)
        if np.sum(index)==0:
            print(f"building {k} is not exist skip it")
            continue
        once_building_data[coordinate_index]=1
        #膨胀
        dilate_once_building_data=cv2.dilate(once_building_data,kernel=np.ones(shape=[10,10],dtype=np.uint8))
        #扩张的区域
        dilate_once_building_data-=once_building_data

        #获取膨胀后和其他建筑物没有交集的区域
        #扩张区域的下标
        dilate_building_bounary_index=np.where(dilate_once_building_data>0)

        #对扩张区域+1
        dilate_building_bounary=label_2d_binary_data.copy()
        dilate_building_bounary[dilate_building_bounary_index]+=1

        #数值>1的区域就是与其他建筑物重叠的区域
        other_building_index=np.where(dilate_building_bounary>1)

        dilate_index_one_dim=dilate_building_bounary_index[0]*image_shape[-2]+dilate_building_bounary_index[1]
        other_building_index_one_dim=other_building_index[0]*image_shape[-2]+other_building_index[1]

        expand_index=list(set(dilate_index_one_dim)^set(other_building_index_one_dim))
        y=[y//image_shape[-2] for y in expand_index]
        x=[x%image_shape[-2] for x in expand_index]


        # #对非重叠区域进行赋值
        # dilate_label_2d_id_data[y,x]=i

        #获取扩张后建筑物的边界
        # once_building_data[y,x]=k
        label_2d_data[y,x]=k

        # contours_list, hierarchy_list, contours_in_list = get_countours(once_building_data)
        #
        #
        # affine_mat=np.array(id_affineparam_dict[str(k)]).reshape(2,3)
        #
        # for i, zip_result in enumerate(zip(contours_list, hierarchy_list, contours_in_list)):
        #     contours, hierarchy, contours_inside = zip_result
        #     if hierarchy[3] == -1:
        #         #将像方建筑物的环投到物方
        #         contours_ring = contours.copy().squeeze(1)
        #
        #         contours_ring=np.expand_dims(img_coordinate_to_geo_coordinate_dot_matri_version(trans_2d,contours_ring),axis=1)
        #         transform_rings = [contours_ring]
        #
        #         # 有内环的情况
        #         if hierarchy[2] != -1:
        #             j = i
        #             break_out = False
        #             while not break_out:
        #                 j += 1
        #                 if j > len(hierarchy_list) - 1:
        #                     break
        #                 if hierarchy_list[j][3] != i:
        #                     break_out = True
        #                 else:
        #                     t_contours=contours_list[j].copy().squeeze(1)
        #                     t_contours = np.expand_dims(
        #                         img_coordinate_to_geo_coordinate_dot_matri_version(trans_2d, t_contours), axis=1)
        #                     transform_rings.append(t_contours)
        #
        #         transform_out_rings = []
        #         # 将外环和内环的点投影至物方
        #         for ring in transform_rings:
        #             # 将像方建筑物环投影至物方---投影坐标
        #             out_points = transformer_wrap(ring.squeeze(axis=1), affine_mat)  # m*2
        #             out_points = geo_coordinate_to_img_coordinate_dot_matri_version(geo_trans_3d, out_points)
        #
        #             index = np.logical_and(out_points[:, 0] > 0, out_points[:, 0] < ref_shape[-1])
        #             index2 = np.logical_and(out_points[:, 1] > 0, out_points[:, 1] < ref_shape[-2])
        #             index = np.logical_and(index, index2)
        #             out_points = out_points[index, :]
        #
        #             if out_points.size == 0:
        #                 continue
        #
        #             out_points = np.expand_dims(out_points, axis=1)
        #             transform_out_rings.append(out_points)
        #
        #         # 获取转换后在物方上的建筑物内部点
        #         transform_out_contours_inside_XY = get_contours_in(transform_out_rings, ref_shape)  # n*2
        #
        #         if transform_out_contours_inside_XY.size == 0:
        #             continue
        #
        #         transform_out_contours_inside = transform_out_contours_inside_XY.copy()
        #
        #
        #         ttt[transform_out_contours_inside_XY[:, 1], transform_out_contours_inside_XY[:, 0]]=k
        #         fastly_save_image(ttt, r"/path/to/project/working_man\ToGWX\data\Affine\shp2\grid\x029518_train_21.tif",gdal_type=gdal.GDT_Float32)
        #
        #         # 将物方像素坐标转到投影坐标上
        #         transform_out_contours_inside = img_coordinate_to_geo_coordinate_dot_matri_version(geo_trans_3d,
        #                                                                                            transform_out_contours_inside)
        #
        #         # 仿射参数是2*3的矩阵，需转为3*3的矩阵求解逆矩阵
        #         column = np.ones((transform_out_contours_inside.shape[0], 1))
        #         transform_out_contours_inside = np.column_stack((transform_out_contours_inside, column)).T
        #         t_affine_mat = np.row_stack((affine_mat, [0, 0, 1]))
        #         try:
        #             affine_mat_inv = np.linalg.inv(t_affine_mat)
        #         except:
        #             print("矩阵不可逆")
        #         else:
        #
        #             transform_out_2d = np.dot(affine_mat_inv, transform_out_contours_inside)  # 3*n
        #
        #             transform_out_2d=geo_coordinate_to_img_coordinate_dot_matri_version(trans_2d,transform_out_2d,True)
        #
        #
        #             # 归一化到[-1,1]的范围内
        #             transform_out_2d = np.dot(coordiante_trans_mat, transform_out_2d)
        #             transform_out_2d = transform_out_2d[0:2, :].T
        #
        #             affine_grid[transform_out_contours_inside_XY[:, 1], transform_out_contours_inside_XY[:, 0],
        #             :] = transform_out_2d
        # print(f"finish buidling {k}")

    for k in range(int(min_building_id),int(max_building_id)+1):
        if k==3537:
            print("ok")
        once_building_data=np.zeros_like(label_2d_data,dtype=np.uint8)
        index=label_2d_data==k
        coordinate_index=np.where(label_2d_data==k)
        if np.sum(index)==0:
            print(f"building {k} is not exist skip it")
            continue
        once_building_data[coordinate_index]=1

        contours_list, hierarchy_list, contours_in_list = get_countours(once_building_data)


        affine_mat=np.array(id_affineparam_dict[str(k)]).reshape(2,3)

        for i, zip_result in enumerate(zip(contours_list, hierarchy_list, contours_in_list)):
            contours, hierarchy, contours_inside = zip_result
            if hierarchy[3] == -1:
                #将像方建筑物的环投到物方
                contours_ring = contours.copy().squeeze(1)

                contours_ring=np.expand_dims(img_coordinate_to_geo_coordinate_dot_matri_version(trans_2d,contours_ring),axis=1)
                transform_rings = [contours_ring]

                # 有内环的情况
                if hierarchy[2] != -1:
                    j = i
                    break_out = False
                    while not break_out:
                        j += 1
                        if j > len(hierarchy_list) - 1:
                            break
                        if hierarchy_list[j][3] != i:
                            break_out = True
                        else:
                            t_contours=contours_list[j].copy().squeeze(1)
                            t_contours = np.expand_dims(
                                img_coordinate_to_geo_coordinate_dot_matri_version(trans_2d, t_contours), axis=1)
                            transform_rings.append(t_contours)

                transform_out_rings = []
                # 将外环和内环的点投影至物方
                for ring in transform_rings:
                    # 将像方建筑物环投影至物方---投影坐标
                    out_points = transformer_wrap(ring.squeeze(axis=1), affine_mat)  # m*2
                    out_points = geo_coordinate_to_img_coordinate_dot_matri_version(geo_trans_3d, out_points)

                    index = np.logical_and(out_points[:, 0] > 0, out_points[:, 0] < ref_shape[-1])
                    index2 = np.logical_and(out_points[:, 1] > 0, out_points[:, 1] < ref_shape[-2])
                    index = np.logical_and(index, index2)
                    out_points = out_points[index, :]

                    if out_points.size == 0:
                        continue

                    out_points = np.expand_dims(out_points, axis=1)
                    transform_out_rings.append(out_points)

                # 获取转换后在物方上的建筑物内部点
                transform_out_contours_inside_XY = get_contours_in(transform_out_rings, ref_shape)  # n*2

                if transform_out_contours_inside_XY.size == 0:
                    continue

                transform_out_contours_inside = transform_out_contours_inside_XY.copy()


                ttt[transform_out_contours_inside_XY[:, 1], transform_out_contours_inside_XY[:, 0]]=1


                # 将物方像素坐标转到投影坐标上
                transform_out_contours_inside = img_coordinate_to_geo_coordinate_dot_matri_version(geo_trans_3d,
                                                                                                   transform_out_contours_inside)

                # 仿射参数是2*3的矩阵，需转为3*3的矩阵求解逆矩阵
                column = np.ones((transform_out_contours_inside.shape[0], 1))
                transform_out_contours_inside = np.column_stack((transform_out_contours_inside, column)).T
                t_affine_mat = np.row_stack((affine_mat, [0, 0, 1]))
                try:
                    affine_mat_inv = np.linalg.inv(t_affine_mat)
                except:
                    print("矩阵不可逆")
                else:

                    transform_out_2d = np.dot(affine_mat_inv, transform_out_contours_inside)  # 3*n

                    transform_out_2d=geo_coordinate_to_img_coordinate_dot_matri_version(trans_2d,transform_out_2d,True)


                    # 归一化到[-1,1]的范围内
                    transform_out_2d = np.dot(coordiante_trans_mat, transform_out_2d)
                    transform_out_2d = transform_out_2d[0:2, :].T

                    affine_grid[transform_out_contours_inside_XY[:, 1], transform_out_contours_inside_XY[:, 0],
                    :] = transform_out_2d

    fastly_save_image(ttt, r"/path/to/project/working_man\ToGWX\data\Affine\shp2\grid\x.tif",
                      ref_file=r"/path/to/project/Xuzhou-Aerial\0_Xuzhou_DeepL\Xuzhou-XYZ\train\label\train_xyz_15.tif",
                      gdal_type=gdal.GDT_Float32)
    return affine_grid

def generate_grid_with_groundtruth_v3(raster_building_2d_id, shpfile_affinebuilding3d, raster_ref_3d,expand_bounary=True):
    """
        对物方进行buffer，buffer区域内，进行先来后到的计算
    Args:
        raster_building_2d_id:
        shpfile_affinebuilding3d:

    Returns:

    """
    raster_io=RasterIO()
    raster_io.open(raster_building_2d_id)
    shp_io=ShpIO(shpfile_affinebuilding3d)
    id_affineparam_dict=shp_io.get_id_affineparam_dict("connectID")

    name=os.path.basename(raster_building_2d_id)

    label_2d_data=raster_io.read_as_array()
    label_2d_data[label_2d_data<0]=0
    label_2d_data.astype('int32')
    image_shape=raster_io.shape()
    trans_2d=raster_io.get_geo_trans()
    raster_io.open(raster_ref_3d)
    geo_trans_3d=raster_io.get_geo_trans()
    ref_shape=raster_io.shape()[1:]

    t_label_2d_data=label_2d_data.copy()
    t_label_2d_data[t_label_2d_data==0]=99999
    min_building_id=np.min(t_label_2d_data)
    t_label_2d_data[t_label_2d_data == 99999] = 0
    max_building_id=np.max(t_label_2d_data)

    min_building_id=min_building_id+1 if min_building_id==0 else min_building_id
    del t_label_2d_data
    t_label_2d_data=None


    label_2d_binary_data=np.zeros_like(label_2d_data)
    label_2d_binary_data[label_2d_data>0]=1

    affine_grid = np.zeros(shape=(ref_shape[-2], ref_shape[-1], 2), dtype=np.float32)
    affine_grid.fill(-2)
    # 定义opencv的坐标系(x,y)到[-1,-1]的转换矩阵
    coordiante_trans_mat = np.array([[2 / (image_shape[-1] - 1), 0, -1],
                                     [0, 2 / (image_shape[-2] - 1), -1],
                                     [0, 0, 1]
                                     ])

    building_bounary_after=np.zeros(shape=(ref_shape[-2], ref_shape[-1]),dtype=np.uint8)
    building_bounary_before=np.zeros_like(label_2d_binary_data,dtype=np.uint8)

    for k in range(int(min_building_id),int(max_building_id)+1):
        if k==10802:
            print("ok")
        once_building_data=np.zeros_like(label_2d_data,dtype=np.uint8)
        index=label_2d_data==k
        coordinate_index=np.where(label_2d_data==k)
        if np.sum(index)==0:
            print(f"building {k} is not exist skip it")
            continue
        once_building_data[coordinate_index]=1

        contours_list, hierarchy_list, contours_in_list = get_countours(once_building_data,method=cv2.CHAIN_APPROX_SIMPLE)




        affine_mat=np.array(id_affineparam_dict[str(k)]).reshape(2,3)

        for i, zip_result in enumerate(zip(contours_list, hierarchy_list, contours_in_list)):
            contours, hierarchy, contours_inside = zip_result
            if hierarchy[3] == -1:
                #将像方建筑物的环投到物方
                contours_ring = contours.copy().squeeze(1)

                cv2.drawContours(building_bounary_before, [contours], -1, 255, thickness=1)####################

                contours_ring=np.expand_dims(img_coordinate_to_geo_coordinate_dot_matri_version(trans_2d,contours_ring),axis=1)
                transform_rings = [contours_ring]

                # 有内环的情况
                if hierarchy[2] != -1:
                    j = i
                    break_out = False
                    while not break_out:
                        j += 1
                        if j > len(hierarchy_list) - 1:
                            break
                        if hierarchy_list[j][3] != i:
                            break_out = True
                        else:
                            cv2.drawContours(building_bounary_before, [contours_list[j]], -1, 255, thickness=1)#################
                            t_contours=contours_list[j].copy().squeeze(1)
                            t_contours = np.expand_dims(
                                img_coordinate_to_geo_coordinate_dot_matri_version(trans_2d, t_contours), axis=1)
                            transform_rings.append(t_contours)



                transform_out_rings = []
                # 将外环和内环的点投影至物方
                for ring in transform_rings:
                    # 将像方建筑物环投影至物方---投影坐标
                    out_points = transformer_wrap(ring.squeeze(axis=1), affine_mat)  # m*2
                    out_points = geo_coordinate_to_img_coordinate_dot_matri_version(geo_trans_3d, out_points)

                    index = np.logical_and(out_points[:, 0] > 0, out_points[:, 0] < ref_shape[-1])
                    index2 = np.logical_and(out_points[:, 1] > 0, out_points[:, 1] < ref_shape[-2])
                    index = np.logical_and(index, index2)
                    out_points = out_points[index, :]

                    if out_points.size == 0:
                        continue

                    out_points = np.expand_dims(out_points, axis=1)
                    transform_out_rings.append(out_points)


                cv2.drawContours(building_bounary_after, transform_out_rings, -1, 255, thickness=1)  ####################


                # 获取转换后在物方上的建筑物内部点，并在物方进行边界的扩张
                transform_out_contours_inside_XY = get_contours_in(transform_out_rings, ref_shape,expand_bounary=expand_bounary)  # n*2

                if transform_out_contours_inside_XY.size == 0:
                    continue

                transform_out_contours_inside = transform_out_contours_inside_XY.copy()

                # 将物方像素坐标转到投影坐标上
                transform_out_contours_inside = img_coordinate_to_geo_coordinate_dot_matri_version(geo_trans_3d,
                                                                                                   transform_out_contours_inside)

                # 仿射参数是2*3的矩阵，需转为3*3的矩阵求解逆矩阵
                column = np.ones((transform_out_contours_inside.shape[0], 1))
                transform_out_contours_inside = np.column_stack((transform_out_contours_inside, column)).T
                t_affine_mat = np.row_stack((affine_mat, [0, 0, 1]))
                try:
                    affine_mat_inv = np.linalg.inv(t_affine_mat)
                except:
                    print("矩阵不可逆")
                else:

                    transform_out_2d = np.dot(affine_mat_inv, transform_out_contours_inside)  # 3*n

                    transform_out_2d=geo_coordinate_to_img_coordinate_dot_matri_version(trans_2d,transform_out_2d,True)


                    # 归一化到[-1,1]的范围内
                    transform_out_2d = np.dot(coordiante_trans_mat, transform_out_2d)
                    transform_out_2d = transform_out_2d[0:2, :].T

                    affine_grid[transform_out_contours_inside_XY[:, 1], transform_out_contours_inside_XY[:, 0],
                    :] = transform_out_2d
        print(f"finish buidling {k}")

    fastly_save_image(building_bounary_before, os.path.join(r"/path/to/project/working_man\ToGWX\ppt\finish\key3\before",name),
                      ref_file=raster_building_2d_id)

    fastly_save_image(building_bounary_after, os.path.join(r"/path/to/project/working_man\ToGWX\ppt\finish\key3\after",name),
                      ref_file=raster_ref_3d)
    return affine_grid


def generate_grid_with_groundtruth_v3_dir(raster_building_2d_id_dir,
                                          shpfile_affinebuilding3d_dir,
                                          raster_ref_3d_dir,
                                          out_dir,
                                          xyz_img_map_file:str,
                                          expand_bounary=True):
    xyz_img_dict = read_xyz_img_pair_file(xyz_img_map_file)
    img_files = glob.glob(os.path.join(raster_building_2d_id_dir, "*.tif"))
    for img_file in img_files:
        name = os.path.basename(img_file)
        # if "test" in name:
        xyz_num=os.path.splitext(name)[0].split("_")[-1]
        regx=re.compile(r'\d+')
        num=regx.findall(name)[0]
        xyz_ref_file = os.path.join(raster_ref_3d_dir, xyz_img_dict[xyz_num] + ".tif")
        shpfile_3d = os.path.join(shpfile_affinebuilding3d_dir, num + ".shp")
        out_file = os.path.join(out_dir, name)
        affine_grid=generate_grid_with_groundtruth_v3(img_file,shpfile_3d,xyz_ref_file,expand_bounary=expand_bounary)
        # affine_grid=np.transpose(affine_grid,[2,0,1])
        # fastly_save_image(affine_grid, out_file,
        #                   ref_file=xyz_ref_file,
        #                   gdal_type=gdal.GDT_Float32)


if __name__ == '__main__':

    # points_pair_dir = r"/path/to/project/ChenLabDataset\Penta-Cam-Centre\0_Penta_DeepL\Penta_IMG_MV\train\point_pairs"
    # model_predict_dir = r"/path/to/project/ChenLabDataset\Penta-Cam-Centre\0_Penta_DeepL\Penta_IMG_MV\train\label"
    # dsm_2d_dir = r"/path/to/project/ChenLabDataset\Penta-Cam-Centre\0_Penta_DeepL\Penta_IMG_MV\train\dsm"
    # pair_file = r"/path/to/project/ChenLabDataset\Penta-Cam-Centre\0_Penta_DeepL\XYZ_xy_name_map.txt"
    # ref_dir = r"/path/to/project/ChenLabDataset\Penta-Cam-Centre\0_Penta_DeepL\Penta_XYZ\train\image"
    # out_dir = r"/path/to/project/ChenLabDataset\Penta-Cam-Centre\0_Penta_DeepL\Penta_IMG_MV\train\grid_all"
    # generate_grid_point_all_image_dir(points_pair_dir,
    #                                   model_predict_dir,
    #                                   dsm_2d_dir,
    #                                   pair_file,
    #                                   ref_dir=ref_dir,
    #                                   out_dir=out_dir,
    #                                   none_data=-99999,symbol='D')

    # prob_raster_io=RasterIO("/path/to/user/NAS/gwx/DeepL/Xuzhou/IMG_MV/train/image/x029517_train_01.tif")
    # predict_raster_io=RasterIO("/path/to/user/NAS/gwx/DeepL/Xuzhou/IMG_MV/train/label/x029517_train_01.tif")
    # dsm_2d_raster_io=RasterIO("/path/to/user/NAS/gwx/DeepL/Xuzhou/IMG_MV/train/dsm/x029517_train_01.tif")
    # XY_raster_io=RasterIO("/path/to/user/NAS/gwx/DeepL/Xuzhou/IMG_MV/train/point_pairs/x029517_train_01.tif")
    # ref_raster_io=RasterIO("/path/to/user/NAS/gwx/DeepL/Xuzhou/XYZ/train/label/train_xyz_0.tif")
    # ref_trans=ref_raster_io.get_geo_trans()
    # ref_shape=ref_raster_io.shape()
    # predict_trans=predict_raster_io.get_geo_trans()

    # model_predict=predict_raster_io.read_as_array()
    # dsm_data=dsm_2d_raster_io.read_as_array()
    # dsm_data=np.expand_dims(dsm_data,axis=0)
    # XY_data=XY_raster_io.read_as_array()
    # prob_data=prob_raster_io.read_as_array()
    # prob=np.zeros_like(prob_data,dtype=np.float32)
    # prob[:,:]=prob_data[:,:]
    # prob_data=prob
    # prob_data=np.expand_dims(prob_data, axis=0)
    # point_pair=np.concatenate([dsm_data,XY_data],axis=0)
    # #
    # grid=generate_grid_with_all_building_affine(point_pair,model_predict,ref_shape,ref_trans,predict_trans,"NONE",True,True)

    # # # grid=generate_grid_with_near_building_affine(point_pair,model_predict,ref_shape,ref_trans,predict_trans)
    # #
    # # # grid,_,_=generate_grid(point_pair,model_predict,ref_shape,ref_trans,predict_trans,True)
    # # # grid=generate_grid_point_all_image(point_pair,model_predict,ref_shape,ref_trans,predict_trans)
    # grid=np.expand_dims(grid,axis=0)
    # grid.astype(np.float32)
    # out=F.grid_sample(torch.tensor(prob_data),torch.tensor(grid),align_corners=True)
    # out=out.numpy().squeeze(axis=0)
    # fastly_save_image(out,"/path/to/user/NAS/gwx/experiment_result/x029517_train_01.tif",gdal_type=gdal.GDT_Float32,
    #                   ref_file="/path/to/user/NAS/gwx/DeepL/Xuzhou/XYZ/train/label/train_xyz_0.tif")

    # parser = argparse.ArgumentParser(description='AffineTransform Configs')
    # parser.add_argument('--points_pair_dir', type=str,default="/path/to/user/NAS/gwx/DeepL/Xuzhou/IMG_MV/test/point_pairs/")
    # parser.add_argument('--model_predict_dir', type=str)
    # parser.add_argument('--dsm_2d_dir', type=str,default="/path/to/user/NAS/gwx/DeepL/Xuzhou/IMG_MV/test/dsm/")
    # parser.add_argument('--pair_file', type=str,default="/path/to/user/NAS/gwx/DeepL/Xuzhou/IMG_MV/XYZ_xy_name_map.txt")
    # parser.add_argument('--ref_dir', type=str,default="/path/to/user/NAS/gwx/DeepL/Xuzhou/XYZ/test/label/")
    # parser.add_argument('--out_dir', type=str)
    # parser.add_argument('--ransac', type=False)

    # cfg_args = parser.parse_args()
    # points_pair_dir = cfg_args.points_pair_dir
    # model_predict_dir = cfg_args.model_predict_dir
    # dsm_2d_dir = cfg_args.dsm_2d_dir
    # pair_file = cfg_args.pair_file
    # ref_dir = cfg_args.ref_dir
    # out_dir = cfg_args.out_dir

    # points_pair_dir = r"/path/to/project/ChenLabDataset\Xuzhou-Aerial\0_Xuzhou_DeepL\Xuzhou-IMG_MV\test\point_pairs"
    # model_predict_dir = r"/path/to/project/ChenLabDataset\Xuzhou-Aerial\0_Xuzhou_DeepL\Xuzhou-IMG_MV\test\label"
    # dsm_2d_dir = r"/path/to/project/ChenLabDataset\Xuzhou-Aerial\0_Xuzhou_DeepL\Xuzhou-IMG_MV\test\dsm"
    # pair_file = r"/path/to/project/ChenLabDataset\Xuzhou-Aerial\0_Xuzhou_DeepL\XYZ_xy_name_map.txt"
    # ref_dir = r"/path/to/project/ChenLabDataset\Xuzhou-Aerial\0_Xuzhou_DeepL\Xuzhou-XYZ\test\image"
    # out_dir = r"/path/to/project/ChenLabDataset\Xuzhou-Aerial\0_Xuzhou_DeepL\Xuzhou-XYZ\test\grid_none"
    # ransac=False
    # generate_grid_only_building_otherwithmode_dir(points_pair_dir, model_predict_dir, dsm_2d_dir, pair_file,
    #                                            ref_dir=ref_dir,
    #                                            out_dir=out_dir,
    #                                            mode="NONE",
    #                                            expand_bounary=True,RANSAC=ransac,
    #                                            none_data=-99999,symbol='D')

    # generate_grid_only_building_otherwithmode_dir(points_pair_dir, model_predict_dir, dsm_2d_dir, pair_file,
    #                                         ref_dir=ref_dir,
    #                                         out_dir=out_dir,
    #                                         mode="NONE",
    #                                         expand_bounary=True,RANSAC=ransac,
    #                                         none_data=0)

    # grid_file_dir=r"/path/to/project/ChenLabDataset\Penta-Cam-Centre\0_Penta_DeepL\Penta_IMG_MV\train\grid_none\A002_010_159000264_train_04.tif"
    # img_file_dir=r"/path/to/project/ChenLabDataset\Penta-Cam-Centre\0_Penta_DeepL\Penta_IMG_MV\train\image\A002_010_159000264_train_04.tif"
    # xyz_ref_dir=r"/path/to/project/ChenLabDataset\Penta-Cam-Centre\0_Penta_DeepL\Penta_XYZ\train\image\train_xyz_4.tif"
    # out_file_dir=r'/path/to/project/ChenLabDataset\Penta-Cam-Centre\0_Penta_DeepL\11.tif'
    # xyz_img_map_file=r"/path/to/project/ChenLabDataset\Penta-Cam-Centre\0_Penta_DeepL\XYZ_xy_name_map.txt"
    # # transform_img_to_xyz_dir(grid_file_dir, img_file_dir, xyz_ref_dir, out_file_dir, xyz_img_map_file,True)
    # transform_img_to_xyz(grid_file_dir, img_file_dir, xyz_ref_dir, out_file_dir,False)

    # label_file2d = r"/path/to/project/working_man\ToGWX\data\Affine\shp2\2dshp_raster\clip\x029518_train_21.tif"
    # shpfile_3d = r"/path/to/project/working_man\ToGWX\data\Affine\shp2\transout\029518.shp"
    # ref_file_3d = r"/path/to/project/Xuzhou-Aerial\0_Xuzhou_DeepL\Xuzhou-XYZ\train\label\train_xyz_15.tif"
    # affine_grid = generate_grid_with_groundtruth_v3(label_file2d, shpfile_3d, ref_file_3d)
    # affine_grid = np.transpose(affine_grid, [2, 0, 1])
    # print(np.max(affine_grid))
    # fastly_save_image(affine_grid, r"/path/to/project/working_man\ToGWX\data\Affine\shp2\grid\x029518_train_21.tif",
    #                   ref_file=r"/path/to/project/Xuzhou-Aerial\0_Xuzhou_DeepL\Xuzhou-XYZ\train\label\train_xyz_15.tif",
    #                   gdal_type=gdal.GDT_Float32)
    #


    # transform_img_to_xyz(r"/path/to/project/Users\35416\Desktop\check_data\dataset\grid\0.tif",
    #                      r"/path/to/project/Users\35416\Desktop\check_data\dataset\image\0.tif",
    #                      r"/path/to/project/Xuzhou-Aerial\0_Xuzhou_DeepL\Xuzhou-XYZ\train\label\train_xyz_15.tif",
    #                      r"/path/to/project/Users\35416\Desktop\check_data\dataset\image_trans_with_grid.tif")
    # raster_building_2d_id_dir = r"/path/to/project/Xuzhou-Aerial\0_Xuzhou_DeepL\Xuzhou-IMG_MV\test\label_id"
    # shpfile_affinebuilding3d_dir = r"/path/to/project/Xuzhou-Aerial\AffineTransform_TDOM_Building_Vec"
    # raster_ref_3d_dir = r"/path/to/project/Xuzhou-Aerial\0_Xuzhou_DeepL\Xuzhou-XYZ\test\label"
    # out_dir=r"/path/to/project/Xuzhou-Aerial\0_Xuzhou_DeepL\Xuzhou-IMG_MV\test\grid_colinear"
    # xyz_img_map_file=r"/path/to/project/Xuzhou-Aerial\0_Xuzhou_DeepL\XYZ_xy_name_map.txt"
    # generate_grid_with_groundtruth_v3_dir(raster_building_2d_id_dir, shpfile_affinebuilding3d_dir, raster_ref_3d_dir, out_dir, xyz_img_map_file)

    # transform_img_to_xyz_dir(r"/model/zhangpl/mvmaper/experiment_result_of_val_nadir_20%/MVMapper_Grid_None_DSM_Hrnet32_All_Fusion_Dataset_DSM_Fusion_image_xyz_lr_5.00e-04/predict/grid",
    #                          r"/model/zhangpl/mvmaper/experiment_result_of_val_nadir_20%/MVMapper_Grid_None_DSM_Hrnet32_All_Fusion_Dataset_DSM_Fusion_image_xyz_lr_5.00e-04/predict/img/predict",
    #                          r"/data/stevenchen/mvmaper/Penta_XYZ/MergeData/label",
    #                          r"/model/zhangpl/mvmaper/experiment_result_of_val_nadir_20%/MVMapper_Grid_None_DSM_Hrnet32_All_Fusion_Dataset_DSM_Fusion_image_xyz_lr_5.00e-04/predict/prob_projwithpiecetable",
    #                          r"/data/stevenchen/mvmaper/Penta_IMG_MV/XYZ_xy_name_map.txt")
    
    # transform_img_to_xyz(r"/path/to/user/NAS/gwx/experiment_result/grid.tif",
    #                      r"/path/to/user/NAS/gwx/experiment_result/image.tif",
    #                      r"/path/to/user/NAS/gwx/DeepL/Xuzhou/XYZ/train/samples_none/label/train_xyz_0_0000_0340.tif",
    #                      r"/path/to/user/NAS/gwx/experiment_result/trans.tif",
    #                      False)
    
    transform_img_to_xyz(
        r"/path/to/user/NAS/gwx/DeepL/Melbourne/IMG_MV/samples/grid_none/SV2-02_20240522_L1B0001027886_6042400500010002_01-PAN2_XYZ_5_0150_1200.tif",
        r"/path/to/user/NAS/gwx/DeepL/Melbourne/IMG_MV/samples/image/SV2-02_20240522_L1B0001027886_6042400500010002_01-PAN2_XYZ_5_0150_1200.tif",
        r"/path/to/user/NAS/gwx/DeepL/Melbourne/XYZ/samples/dsm/XYZ_5_0150_1200.tif",
        r"/path/to/user/NAS/gwx/DeepL/Melbourne/trans.tif",False)
    
    
    # transform_img_to_xyz(r"/path/to/user/NAS/gwx/DeepL/Xuzhou/IMG_MV/train/samples_none/grid_none/x029517_train_01_0000_0340.tif",
    #                     r"/path/to/user/NAS/gwx/DeepL/Xuzhou/IMG_MV/train/samples_none/image/x029517_train_01_0000_0340.tif",
    #                     r"/path/to/user/NAS/gwx/DeepL/Xuzhou/XYZ/train/samples_none/label/train_xyz_0_0000_0340.tif",
    #                     r"/path/to/user/NAS/gwx/experiment_result/trans.tif",
    #                     False)
    # ra=RasterIO(r"/path/to/user/NAS/gwx/experiment_result/trans.tif")
    # data=ra.read_as_array_with_scale(0.125)
    # fastly_save_image(data,
    #                   r'/path/to/user/NAS/gwx/experiment_result/resize.tif',
    #                   ref_file=r"/path/to/user/NAS/gwx/experiment_result/trans.tif",
    #                   gdal_type=gdal.GDT_Float32)
    
    # grid_file=r"/path/to/user/NAS/gwx/DeepL/Melbourne/IMG_MV/test/samples/grid_none/IMG_PHR1A_P_201202250025329_SEN_PRG_FC_5110-001_R1C1_test_04_0119_0228.tif"
    # img_file=r"/path/to/user/NAS/gwx/DeepL/Melbourne/IMG_MV/test/samples/image/IMG_PHR1A_P_201202250025329_SEN_PRG_FC_5110-001_R1C1_test_04_0119_0228.tif"
    # xyz_ref=r"/path/to/user/NAS/gwx/DeepL/Melbourne/XYZ/test/samples/image/test_04_0119_0228.tif"
    # out_file=r"/path/to/user/NAS/gwx/DeepL/Melbourne/trans2.tif"
    
    # img_raster_io = RasterIO(img_file)
    # img_data = img_raster_io.read_as_array()
    # prob = np.zeros_like(img_data, dtype=np.float32)
    # prob[:, :] = img_data[:, :]
    # img_data = prob
    # if img_data.ndim<4:
    #     img_data = np.expand_dims(img_data, axis=list(range(4-img_data.ndim)))

    # original_shape = img_data.shape
    # img_data=np.pad(img_data,((0,0),(0,0),(0,100),(0,100)),mode='constant',constant_values=0)
    # padding_shape= img_data.shape
    
    # grid_raster_io = RasterIO(grid_file)
    # grid = grid_raster_io.read_as_array().astype("float32")
    # grid = np.expand_dims(grid, axis=0)
    # grid.astype(np.float32)
    # grid = np.transpose(grid, [0, 2, 3, 1])
    
    # grid[:,:,:,0]=(grid[:,:,:,0]+1)*((original_shape[2]-1)/(padding_shape[2]-1))-1
    # grid[:,:,:,1]=(grid[:,:,:,1]+1)*((original_shape[3]-1)/(padding_shape[3]-1))-1
    # out = F.grid_sample(torch.tensor(img_data), torch.tensor(grid), align_corners=True)
    # out = out.numpy().squeeze(axis=0)


    # fastly_save_image(out, out_file,ref_file=xyz_ref, gdal_type=gdal.GDT_Float32)
