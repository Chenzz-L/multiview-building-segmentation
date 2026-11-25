#!/conda/env python
# -*- coding:utf-8 -*-
import glob
import os

import numpy as np
from osgeo import gdal

__all__=['get_building_map','region_grow']

def region_grow(raster,threshold=0,none_data=0):
    """
    区域生长8邻域，遍历整个图
    Args:
        raster: 输入的原始数据
        threshold:区域生长阈值
        none_data:图像的无效值

    Returns:
        区域生长块
    """
    directs = [(-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0)]
    visited = np.zeros(shape=(raster.shape), dtype=np.uint8)
    res_data=np.zeros(shape=(raster.shape), dtype=np.uint8)
    h,w=raster.shape

    #从整图的第一个像素开始遍历
    nID=1
    lasprocessing=int(np.sum(visited)/(w*h)*100)
    for i in range(h):
        for j in range(w):
            if(raster[i][j]==none_data or visited[i][j]):
                visited[i][j] = 1
                continue
            seeds = [(j,i)]
            while len(seeds):
                seed = seeds.pop(0)
                x = seed[0]
                y = seed[1]
                # visit point (x,y)
                visited[y][x] = 1
                for direct in directs:
                    cur_x = x + direct[0]
                    cur_y = y + direct[1]

                    # 非法
                    if cur_x < 0 or cur_y < 0 or cur_x >= w or cur_y >= h:
                        continue
                    if raster[cur_y][cur_x] == none_data:
                        visited[cur_y][cur_x] = 1
                        continue
                    # 没有访问过且属于同一目标
                    if (not visited[cur_y][cur_x]) and (abs(raster[cur_y][cur_x]-raster[y][x])<=threshold):
                        res_data[cur_y][cur_x] = nID
                        visited[cur_y][cur_x] = 1
                        seeds.append((cur_x, cur_y))
            nID+=1
        if int(np.sum(visited)/(w*h)*100)>lasprocessing:
            print(f'processing:{int(np.sum(visited)/(w*h)*100)}%')
            lasprocessing=int(np.sum(visited)/(w*h)*100)

    return res_data



def get_building_map(building_data,background=0.0):
    map_building={}
    building_data=np.array(building_data)
    max_id=building_data.max()
    img_W=building_data.shape[1]
    img_H=building_data.shape[2]
    # for i in range(1,max_id+1):
        # x,y=np.where(building_data==i)
        # map_building[i]=[*(x+y*img_W)]
    for i in range(img_H):
        for j in range(img_W):
            if building_data[i,j]-background<1e-3:
                continue
            if building_data[i,j] in map_building.keys():
                map_building[building_data[i,j]].append(i*img_H,+j)
            else:
                map_building[building_data[i,j]]=[i*img_H,+j]


def convert_label_to_multi_label(out_dir,dir_list):
    files_list=glob.glob(dir_list[0]+'\*.tif')
    for file in files_list:
        name=os.path.basename(file)
        out_file=os.path.join(out_dir,name)
        poDs=gdal.Open(file,gdal.GA_ReadOnly)
        img_w=poDs.RasterXSize
        img_h=poDs.RasterYSize
        data=poDs.ReadAsArray()
        trans=poDs.GetGeoTransform()
        prj=poDs.GetSpatialRef()
        OutDs = gdal.GetDriverByName("GTiff").Create(out_file, img_w, img_h, 1, gdal.GDT_Float32,
                                                     options=["TILED=YES", "COMPRESS=LZW"])
        i=2
        dir_lists=dir_list[1:]
        for raster_file in dir_lists:
            ds=gdal.Open(os.path.join(raster_file,name),gdal.GA_ReadOnly)
            data_array=ds.ReadAsArray()
            data[data_array==1]=i
            i+=1
            del ds


        band = OutDs.GetRasterBand(1)
        band.WriteArray(data)
        OutDs.SetSpatialRef(prj)
        OutDs.SetGeoTransform(trans)

        del poDs
        del OutDs



if __name__=='__main__':
    # raster_file=r'/path/to/project/working_man\ToGWX\data\Affine\Raster\BRNetv2_epoch280_XYZ\binary\test_xyz_0_binary.tif'
    # raster_out_file=r'/path/to/project/working_man\ToGWX\data\Affine\Raster\BRNetv2_epoch280_XYZ\11111.tif'
    # ds = gdal.Open(raster_file)
    # raster_data=ds.ReadAsArray()
    # res_data=region_grow(raster_data)
    # driver = gdal.GetDriverByName("GTiff")
    # outds = driver.Create( raster_out_file,
    #                          raster_data.shape[1],
    #                          raster_data.shape[0],
    #                          1,
    #                          gdal.GDT_Byte,
    #                          options=['COMPRESS=DEFLATE'])
    # trans=ds.GetGeoTransform()
    # proj=ds.GetProjectionRef()
    # outds.GetRasterBand(1).WriteArray(res_data)
    # outds.SetGeoTransform(trans)
    # outds.SetProjection(proj)
    # outds.FlushCache()
    #
    # del ds
    # ds=None
    # del outds
    # outds=None
    out_dir=r'/path/to/project/Xuzhou-Aerial\0_Xuzhou_DeepL\Xuzhou-IMG_MV\test\label_concate'
    files=[r'/path/to/project/Xuzhou-Aerial\0_Xuzhou_DeepL\Xuzhou-IMG_MV\test\label',r'/path/to/project/Xuzhou-Aerial\0_Xuzhou_DeepL\Xuzhou-IMG_MV\test\wall']
    convert_label_to_multi_label(out_dir,files)
