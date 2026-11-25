import glob
import os

from osgeo import gdal
import numpy as np
import cv2 as cv

from utils.raster_io import *

__all__=["get_countours","get_contours_in",'region_grow','get_building_map','convert_label_to_multi_label',
         "generate_image_index"]

def get_countours(label_data: np.ndarray,mode=cv.RETR_TREE,method=cv.CHAIN_APPROX_NONE,expand_bounary=False):
    """
        contours_in_list包含了contours_list中的点
        获取二值图像每一个建筑物的轮廓、内部点、等级结构
        返回的都是[x,y]格式
    Args:
        label_data: 二值化图像数据

    Returns:
        contours_list：外轮廓
        hierarchy_list：每一个环的层级结构
        每个轮廓contours[i]对应4个hierarchy元素hierarchy[i][0] ~hierarchy[i][3]，分别表示
        后一个轮廓、前一个轮廓、父轮廓、内嵌轮廓的索引编号，如果没有对应项，则该值为负数
        contours_in_list：每一个轮廓内部的点
    """
    label=label_data.copy()
    label=label.astype(np.uint8)
     # 不同版本自行选择
    # _,contours_list, hierarchy_list = cv.findContours(label_data, mode, method)
    contours_list, hierarchy_list = cv.findContours(label_data.astype(np.uint8), mode, method)
    if hierarchy_list is None:
        return None,None,None
    
    hierarchy_list=hierarchy_list.squeeze(axis=0)
    
    #建筑物内部的点
    contours_in_list=[]
    for i,hierarchy in enumerate(hierarchy_list):
        #判断是否是内敛轮廓
        if hierarchy[2]==-1 and hierarchy[3]==-1:
            contours_in_list.append(get_contours_in([contours_list[i]],label_data.shape,expand_bounary))
        elif hierarchy[2]==-1 and hierarchy[3]!=-1:
            contours_in_list.append(np.array([]))
        else:
            #将同一个建筑物的轮廓组合起来
            building_ring_index=[i]
            j=i
            break_out=False
            while not break_out:
                j+=1
                if j>len(hierarchy_list)-1:
                    break
                if hierarchy_list[j][3]!=i:
                  break_out=True
                else:
                  building_ring_index.append(j)
            start_index=building_ring_index[0]
            end_index=building_ring_index[-1]
            contours_in_list.append(get_contours_in(contours_list[start_index:end_index+1],label_data.shape,expand_bounary))


    return contours_list,hierarchy_list,tuple(contours_in_list)

def get_contours_in(contours,shape,expand_bounary=False):
    """
    返回的是[x,y]格式
    Args:
        contours:轮廓的list---contours是一个list,其中包含了多个轮廓，不然无法进行填充
        shape:图像的形状，dim=2

    Returns:
        contours的内部点
    """
    # 第一个参数image表示绘制的目标图像
    # 第二个参数contours表示输入的轮廓组
    # 第三个参数contourIdx指明画第几个轮廓，如果该参数为负值，则画全部轮廓
    # 第四个参数color为轮廓的颜色
    # 第五个参数thickness为轮廓的线宽，如果为负值表示填充轮廓内部
    p=np.zeros(shape=shape,dtype=np.uint8)
    if expand_bounary:
        cv.drawContours(p,contours,-1,255,10)
        cv.drawContours(p,contours,-1,255,-1)
    else:
        cv.drawContours(p,contours,-1,255,-1)
    # a=np.where(p==255)[0].reshape(-1,1)
    # b=np.where(p==255)[1].reshape(-1,1)
    y_x=np.where(p==255)#0：行，1：列
    x_y=list(zip(y_x[1],y_x[0]))

    # fastly_save_image(p, r"/path/to/project/Users\35416\Desktop\test\111.tif")
    # coordinate=np.concatenate([b,a],axis=1).tolist()
    # inside=[tuple(x) for x in coordinate]

    del p
    p=None
    return np.array(x_y)

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

def division_building_with_dsm(building_data,dsm_data,background=0):
    """
    根据dsm分割建筑物
    Args:
        building_data:
        dsm_data:
        background:

    Returns:

    """
    pass

def generate_image_index(data_shape):
    """
    生成二维图像的下标,使用shape=[w*h,2] 表示每一个坐标
    Args:
        data_shape:

    Returns:

    """
    x=np.arange(0,data_shape[1]*data_shape[0])
    x=np.reshape(x,newshape=data_shape)
    x=x%data_shape[1]
    x = np.expand_dims(x, axis=0)

    y=np.arange(0,data_shape[1]*data_shape[0])
    y=np.reshape(y,newshape=data_shape)
    y=y//data_shape[1]
    y=np.expand_dims(y,axis=0)

    x_y=np.concatenate([x,y])
    return x_y

def depart_band(input_file,out_file,stay_band=(1,)):
    ds_raster = gdal.Open(input_file, 0)
    driver = gdal.GetDriverByName("GTiff")
    w = ds_raster.RasterXSize
    h = ds_raster.RasterYSize
    band_count=ds_raster.RasterCount
    groTrans=ds_raster.GetGeoTransform()
    if os.path.exists(out_file):  # 若文件已经存在，则删除它继续重新做一遍
        driver.Delete(out_file)
    print(len(stay_band))
    out_ds = driver.Create(out_file,w,h,len(stay_band), gdal.GDT_Float32,options=["TILED=YES", "COMPRESS=LZW"])

    for i, band in enumerate(stay_band):
        if band>band_count+1:
            continue
        out_band=out_ds.GetRasterBand(i+1)
        out_band.WriteArray(ds_raster.GetRasterBand(band).ReadAsArray().astype(np.float32))
        out_band.FlushCache()
    out_ds.SetGeoTransform(groTrans)
    del out_ds
    del ds_raster

def depart_band_dir(input_dir,out_dir,stay_band=(1,)):
    input_fils=glob.glob(os.path.join(input_dir,"*.tif"))
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    for input_file in input_fils:
        name=os.path.basename(input_file)
        name=name.replace("_prob","")
        out_file=os.path.join(out_dir,name)
        depart_band(input_file,out_file,stay_band)

def roof_wall_2_roof(predict_file,out_file,stay_class=1,nodata=0):
    """
    将多分类中的屋顶提取出来
    Args:
        predict_file:模型预测类别结果路径
        out_file:输出文件路径
    Returns:
        nothing
    """
    ds_raster = gdal.Open(predict_file, 0)
    driver = gdal.GetDriverByName("GTiff")
    if os.path.exists(out_file):  # 若文件已经存在，则删除它继续重新做一遍
        driver.Delete(out_file)
    out_ds=driver.CreateCopy(out_file,ds_raster,options=["TILED=YES", "COMPRESS=LZW"])
    w=out_ds.RasterXSize
    h=out_ds.RasterYSize
    raster_array=out_ds.ReadAsArray()
    raster_array[raster_array!=stay_class]=0
    raster_band=out_ds.GetRasterBand(1)
    raster_band.WriteRaster(0,0,w,h,raster_array)
    raster_band.SetNoDataValue(nodata)
    raster_band.FlushCache()
    del out_ds
    del ds_raster

def roof_wall_2_roof_dir(predict_dir,out_dir,stay_class=1,nodata=0):
    predict_files=glob.glob(os.path.join(predict_dir,"*.tif"))
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    for predict_file in predict_files:
        name=os.path.basename(predict_file)
        name=name.replace("_pred","")
        out_file=os.path.join(out_dir,name)
        roof_wall_2_roof(predict_file,out_file,stay_class,nodata)

if __name__=='__main__':
    # raster_io=RasterIO(r"/path/to/project/working_man\ToGWX\data\Affine\Raster\CNN_MV_test\2dshp\3\roof_pred\x029517_test_05.tif")
    # dsm_2d_raster_io=RasterIO(r"/path/to/project/working_man\ToGWX\data\Affine\Raster\CNN_MV_test\2dshp\3\roof_pred\x029517_test_05.tif")
    # model_predict=raster_io.read_as_array()
    # data=raster_io.read_as_array()
    # contours_list,hierarchy_list,contours_in_list=get_countours(data)
    print("ok")
