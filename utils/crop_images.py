# -*- coding: utf-8 -*-
import argparse
import glob
import os
import re
import sys
from collections import Counter
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# sys.path[0] = os.path.join(sys.path[0], '..')
# sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from osgeo import gdal,ogr,osr

from utils.raster_io import *
from utils.coordinate_convert import *

def filte_difference(sourece_list, target_list):
    sourece_dir=os.path.dirname(sourece_list[0])
    target_dir=os.path.dirname(target_list[0])
    sourece_name=[os.path.basename(file) for file in sourece_list]
    target_name=[os.path.basename(file) for file in target_list]
    intersection = list(set(sourece_name) & set(target_name))
    sourece_list=[os.path.join(sourece_dir,name) for name in intersection]
    tarfect_list=[os.path.join(target_dir,name) for name in intersection]
    return sourece_list, tarfect_list

def filte_exist(sourece_list, target_list):
    source_out=[]
    target_out=[]
    for source,target in zip(sourece_list,target_list):
        if os.path.exists(source) and os.path.exists(target):
           source_out.append(source)
           target_out.append(target)
    return source_out, target_out

def get_crop_coordinate_from_xyz_corner_colinear(coordinate_corners,image_shape):
    """
    根据物方的切图角点获取像方的范围,像方角点通过dem进行共线条件方程投影生成
    Args:
        coordinate_corners:共线条件方程生成的角点列表-----List[left top,left bottom,right top,right bottom]
        image_shape:像方图像shape-----[h,w]、[c,h,w]、[n,c,h,w]
    Returns:

    """
    left_top_x, left_top_y = coordinate_corners[0], coordinate_corners[1]
    left_bottom_x, left_bottom_y = coordinate_corners[2], coordinate_corners[3]

    right_top_x, right_top_y = coordinate_corners[4], coordinate_corners[5]
    right_bottom_x, right_bottom_y = coordinate_corners[6], coordinate_corners[7]

    contour = np.array([[left_top_x, left_top_y], [left_bottom_x, left_bottom_y],
                        [right_top_x, right_top_y], [right_bottom_x, right_bottom_y]])

    point_list=[(left_bottom_x,left_bottom_y),(left_top_x,left_top_y),(right_bottom_x,right_bottom_y),(right_top_x,right_top_y)]
    counts = Counter(point_list)

    # 找到出现次数最多的元素的个数
    if(max(counts.values())>=2):
        return 0,0,0,0

    max_x, min_x = int(np.ceil(np.max(contour[:, 0]))), int(np.floor(np.min(contour[:, 0])))
    max_y, min_y = int(np.ceil(np.max(contour[:, 1]))), int(np.floor(np.min(contour[:, 1])))

    # min_x = min_x if min_x > 0 else 0
    # min_y = min_y if min_y > 0 else 0
    # max_x = max_x if max_x < image_shape[-1] else image_shape[-1]
    # max_y = max_y if max_y < image_shape[-2] else image_shape[-2]

    min_x = max(min_x,0)
    min_y = max(min_y,0)
    max_x = min(max_x,image_shape[-1])
    max_y = min(max_y,image_shape[-2])

    image_sample_width = int(max_x - min_x)
    image_sample_height = int(max_y - min_y)
    return image_sample_width,image_sample_height,min_x,min_y

def crop_image(image_file: str,
               sample_image_file: str,
               xoff: int, yoff: int, xsize: int, ysize: int,
               compress_image: bool, scale: float, resample_mode: str,
               outsize_x: int, outsize_y: int, ot: str = "Byte"):
    r"""
    裁剪单个影像至特定目录，可以将(xsize,ysize)裁切至(outsize_x,outsize_y)大小的切片：
    当切片大小与最终输出切片大小不一样时，会对所给窗口大小内的图像进行缩放到最终输出大小，可根据这两个参数进行缩放
    :param image_file: 需要被裁剪的影像路径
    :param sample_image_file: 裁剪输出结果的影像路径
    :param xoff: 裁切起点的x坐标
    :param yoff: 裁切起点的y坐标
    :param xsize: 裁切大小x
    :param ysize: 裁切大小y
    :param compress_image: 是否压缩
    :param scale: 是否将裁切影像重采样至另外的大小
    :param resample_mode: 重采样的模式：{nearest (default),bilinear,cubic,cubicspline,lanczos,average,rms,mode}
    :param outsize_x: 最终输出的切片大小x
    :param outsize_y: 最终输出的切片大小y
    :param ot: 输出影像的类型 Byte UInt16 Int16 UInt32 Int32 Float32 Float64 CInt16 CInt32 CFloat32 CFloat64
    :return: nothing
    """
    cmd = ["gdal_translate", image_file, sample_image_file, "-srcwin",
           str(xoff), str(yoff), str(xsize), str(ysize),
           "-ot", ot,
           # "-co", "COMPRESS=DEFLATE", #"-co", "PHOTOMETRIC=YCBCR",
           "-co", "TILED=YES",
           "--config", "GDAL_TIFF_INTERNAL_MASK YES"
           ]
    if compress_image:
        cmd.append("-co")
        cmd.append("COMPRESS=DEFLATE")
    if scale != 1:
        cmd.append("-r")
        cmd.append(resample_mode)
        cmd.append("-outsize")
        cmd.append(str(outsize_x))
        cmd.append(str(outsize_y))
    strcmd = " ".join(cmd)
    os.system(strcmd)

def make_cmd(image_file, sample_image_file,
             xoff, yoff, xsize, ysize,
             compress_image, scale, resample_mode,
             outsize_x, outsize_y, ot="Byte"):
    cmd = ["gdal_translate", image_file, sample_image_file, "-srcwin",
           str(xoff), str(yoff), str(xsize), str(ysize),
           "-ot", ot,
           # "-co", "COMPRESS=DEFLATE", #"-co", "PHOTOMETRIC=YCBCR",
           # "-co", "TILED=YES",
           # "--config", "GDAL_TIFF_INTERNAL_MASK YES"
           ]
    if compress_image:
        cmd.append("-co")
        cmd.append("COMPRESS=DEFLATE")
    if scale != 1:
        cmd.append("-r")
        cmd.append(resample_mode)
        cmd.append("-outsize")
        cmd.append(str(outsize_x))
        cmd.append(str(outsize_y))
    return cmd

def crop_images_fixed_names(input_dir: str,
                            output_dir: str,
                            sample_width: int,
                            sample_height: int,
                            sample_stride: int,
                            scale: float,
                            crop_all: bool,
                            none_value: float,
                            make_csv: bool,
                            csv_mode: str,
                            compress_image: bool,
                            data_type: str,
                            name1: str, name2: str, name3: str, name4: str):
    r"""
    将一个目录内，以name1, name2, name3, name4为名的文件夹内的以data_type为类型的文件按照一定大小裁剪至output_dir
    :param input_dir: 输入路径
    :param output_dir: 输出路径
    :param sample_width: 裁切大小x
    :param sample_height: 裁切大小y
    :param sample_stride: 裁剪的步长
    :param scale: 将即将要裁剪的切片按照一定比例重采样至新的大小
    :param crop_all: 是否裁剪所有影像，因为有的部分会全是无效值
    :param none_value: 没有数据的地方的值
    :param make_csv: 是否制作csv记录影像信息
    :param csv_mode: w	打开一个文件只用于写入。a 打开一个文件用于追加。a+	打开一个文件用于读写。
    :param compress_image: 是否压缩文件
    :param data_type: 文件的后缀 tiff tif
    :param name1: 文件夹名，image
    :param name2:
    :param name3:
    :param name4:
    :return:
    """
    outsize_x = sample_width
    outsize_y = sample_height
    sample_width = int(scale * sample_width)
    sample_height = int(scale * sample_height)
    sample_stride = int(scale * sample_stride)

    if name1 != '':
        input_name1_dir = os.path.join(input_dir, name1)
        output_name1_dir = os.path.join(output_dir, name1)
        if not os.path.exists(output_name1_dir):
            os.makedirs(output_name1_dir)
    else:
        print('name1 is None')
        return -1
    if name2 != '':
        input_name2_dir = os.path.join(input_dir, name2)
        output_name2_dir = os.path.join(output_dir, name2)
        if not os.path.exists(output_name2_dir):
            os.makedirs(output_name2_dir)
    if name3 != '':
        input_name3_dir = os.path.join(input_dir, name3)
        output_name3_dir = os.path.join(output_dir, name3)
        if not os.path.exists(output_name3_dir):
            os.makedirs(output_name3_dir)
    if name4 != '':
        input_name4_dir = os.path.join(input_dir, name4)
        output_name4_dir = os.path.join(output_dir, name4)
        if not os.path.exists(output_name4_dir):
            os.makedirs(output_name4_dir)

    if make_csv:
        info_file = open(output_dir + '/indices.csv', csv_mode)
        info_file.write('name,height,width,positive_ratio,mean_value\n')
    for image_file in glob.glob(input_name1_dir + '/*.' + data_type):

        image_name = os.path.basename(image_file)
        if name2 != '':
            name2_file = input_name2_dir + "/" + image_name
        if name3 != '':
            name3_file = input_name3_dir + "/" + image_name
        if name4 != '':
            name4_file = input_name4_dir + "/" + image_name

        ds = gdal.Open(image_file)
        xoffs = list(range(0, ds.RasterXSize - sample_width, sample_stride))
        xoffs.append(ds.RasterXSize - sample_width)
        yoffs = list(range(0, ds.RasterYSize - sample_height, sample_stride))
        yoffs.append(ds.RasterYSize - sample_height)

        if ds.RasterXSize < sample_width or ds.RasterYSize < sample_height:
            continue

        for xoff in xoffs:
            for yoff in yoffs:
                xsize = min(sample_width, ds.RasterXSize - xoff)
                ysize = min(sample_height, ds.RasterYSize - yoff)
                print('xoff=%d, yoff=%d', xoff, yoff)
                sample_image_name = os.path.splitext(image_name)[0] + '_' + str(
                    scale) + '_' + '%.4d' % xoff + '_' + '%.4d' % yoff + '.' + data_type
                sample_image_file = output_name1_dir + "/" + sample_image_name

                image_data = ds.ReadAsArray(xoff, yoff, xsize=xsize, ysize=ysize)
                empty_num = np.sum(image_data == none_value)

                # image
                print('making sample: ' + sample_image_name)
                if crop_all or empty_num < image_data.size * 0.8:
                    # name1
                    crop_image(image_file, sample_image_file,
                               xoff, yoff, xsize, ysize,
                               compress_image, scale, 'cubic',
                               outsize_x, outsize_y)
                    if make_csv:
                        mean_value = np.mean(np.reshape(image_data, (image_data.shape[0], -1)))
                    # name2
                    if name2 != '':
                        sample_name2_file = output_name2_dir + "/" + sample_image_name
                        crop_image(name2_file, sample_name2_file,
                                   xoff, yoff, xsize, ysize,
                                   compress_image, scale, 'nearest',
                                   outsize_x, outsize_y)
                        if make_csv:
                            label_ds = gdal.Open(sample_name2_file)
                            labeldata = label_ds.ReadAsArray()
                            positive_ratio = np.sum(labeldata > 0) / float(outsize_x * outsize_y)
                            info_file.write(
                                '%s,%d,%d,%.3f,%d\n' % (sample_image_name, ysize, xsize, positive_ratio, mean_value))
                    # name3
                    if name3 != '':
                        sample_name3_file = output_name3_dir + "/" + sample_image_name
                        crop_image(name3_file, sample_name3_file,
                                   xoff, yoff, xsize, ysize,
                                   False, scale, 'cubic',
                                   outsize_x, outsize_y, ot='Float32')
                    # name4
                    if name4 != '':
                        sample_name4_file = output_name4_dir + "/" + sample_image_name
                        crop_image(name4_file, sample_name4_file,
                                   xoff, yoff, xsize, ysize,
                                   compress_image, scale, 'nearest',
                                   outsize_x, outsize_y)
    if make_csv:
        info_file.close()

def crop_image_dir(input_dir: str,
                        output_dir: str,
                        data_type: str = 'tif',
                        ot: str = 'Float32',
                        sample_width: int = 512,
                        sample_height: int = 512,
                        sample_stride: int = 170,
                        none_value: float = 0.0,
                        scale: float = 1.0,
                        crop_all: bool = False,
                        compress_image: bool = True,
                        symbol=None):
    """
    裁剪某一个目录下的所有图像
    Args:
        input_dir:
        output_dir:
        data_type:
        ot:
        sample_width:
        sample_height:
        sample_stride:
        none_value:
        scale:
        crop_all:
        compress_image:

    Returns:

    """
    print('start cropping images')
    outsize_x = sample_width
    outsize_y = sample_height
    sample_width = int(scale * sample_width)
    sample_height = int(scale * sample_height)
    sample_stride = int(scale * sample_stride)


    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    for image_file in glob.glob(input_dir + '/*.' + data_type):
        # extent
        if symbol!=None and symbol not in image_file:
            print(f'{symbol} not in the {image_file}')
            continue
        try:
            ds = gdal.Open(image_file)
        except:
            with open('error.log', 'a') as f:
                f.write(image_file + '\n')
            continue
        xoffs = list(range(0, ds.RasterXSize - sample_width, sample_stride))
        xoffs.append(ds.RasterXSize - sample_width)
        l_xoffs = len(xoffs)
        yoffs = list(range(0, ds.RasterYSize - sample_height, sample_stride))
        yoffs.append(ds.RasterYSize - sample_height)
        l_yoffs = len(yoffs)

        if ds.RasterXSize < sample_width or ds.RasterYSize < sample_height:
            continue

        image_name = os.path.basename(image_file)
        for i, xoff in enumerate(xoffs):
            for j, yoff in enumerate(yoffs):
                xsize = min(sample_width, ds.RasterXSize - xoff)
                ysize = min(sample_height, ds.RasterYSize - yoff)
                percent = (i / l_xoffs + j / l_xoffs / l_yoffs) * 100
                print('calculating %.3f' % percent, '%')
                # print('xoff=%d, yoff=%d', xoff, yoff)

                # read image to check if here is no value area
                image_data = ds.ReadAsArray(xoff, yoff, xsize=xsize, ysize=ysize)
                empty_num = np.sum(image_data == none_value)
                if crop_all or empty_num < image_data.size * 0.8:
                    sample_image_name = os.path.splitext(image_name)[0] + '_' + str(
                        scale) + '_' + '%.4d' % xoff + '_' + '%.4d' % yoff + '.' + data_type
                    if scale == 1.0:
                        sample_image_name = os.path.splitext(image_name)[0] + '_' + '%.4d' % xoff \
                                            + '_' + '%.4d' % yoff + '.' + data_type
                    print('making sample: ' + sample_image_name)

                    in_file = os.path.join(input_dir, image_name)
                    out_file = os.path.join(output_dir, sample_image_name)
                    crop_image(image_file=in_file,
                               sample_image_file=out_file,
                               xoff=xoff, yoff=yoff, xsize=xsize, ysize=ysize,
                               compress_image=compress_image, scale=scale, resample_mode='nearest',
                               outsize_x=outsize_x, outsize_y=outsize_y, ot=ot)

def crop_images(input_dir: str,
                output_dir: str,
                data_type: str = 'tif',
                name_list: tuple = (
                'image', 'label_CBO', 'label_dx', 'label_dy', 'label_footprint', 'label_roof', 'label_roof_wall'),
                ot_list: tuple = ('Byte', 'Byte', 'Float32', 'Float32', 'Byte', 'Byte', 'Byte'),
                sample_width: int = 512,
                sample_height: int = 512,
                sample_stride: int = 170,
                none_value: float = 0,
                scale: float = 1.0,
                crop_all: bool = False,
                compress_image: bool = True):
    r"""
    将一个目录内，以name1, name2, name3, name4为名的文件夹内的以data_type为类型的文件按照一定大小裁剪至output_dir
    :param input_dir: 输入路径
    :param output_dir: 输出路径
    :param sample_width: 裁切大小x
    :param sample_height: 裁切大小y
    :param sample_stride: 裁剪的步长
    :param scale: 将即将要裁剪的切片按照一定比例重采样至新的大小
    :param crop_all: 是否裁剪所有影像，因为有的部分会全是无效值
    :param none_value: 没有数据的地方的值
    :param compress_image: 是否压缩文件
    :param data_type: 文件的后缀 tiff tif
    :param name_list: 文件夹名的列表 ['image', 'label'],第一个必须是影像
    :param ot_list: 每种需要裁剪影像的输出格式,Byte,Float32,Int16
    :return:
    """
    if len(name_list) != len(ot_list):
        raise Exception
    print('start cropping images')
    outsize_x = sample_width
    outsize_y = sample_height
    sample_width = int(scale * sample_width)
    sample_height = int(scale * sample_height)
    sample_stride = int(scale * sample_stride)

    input_dirs = {}
    output_dirs = {}

    # create dirs
    for i, name in enumerate(name_list):
        if name != '' and name is not None:
            input_name_dir = os.path.join(input_dir, name)
            output_name_dir = os.path.join(output_dir, name)
            if not os.path.exists(output_name_dir):
                os.makedirs(output_name_dir)
            input_dirs[name] = input_name_dir
            output_dirs[name] = output_name_dir
        else:
            raise Exception('invalid name in name list!')

    # start cropping, image as the first dir in default
    for image_file in glob.glob(input_dirs['image'] + '/*.' + data_type):
        # extent
        ds = gdal.Open(image_file)
        xoffs = list(range(0, ds.RasterXSize - sample_width, sample_stride))
        xoffs.append(ds.RasterXSize - sample_width)
        l_xoffs = len(xoffs)
        yoffs = list(range(0, ds.RasterYSize - sample_height, sample_stride))
        yoffs.append(ds.RasterYSize - sample_height)
        l_yoffs = len(yoffs)

        if ds.RasterXSize < sample_width or ds.RasterYSize < sample_height:
            continue

        image_name = os.path.basename(image_file)
        for i, xoff in enumerate(xoffs):
            for j, yoff in enumerate(yoffs):
                xsize = min(sample_width, ds.RasterXSize - xoff)
                ysize = min(sample_height, ds.RasterYSize - yoff)
                percent = (i / l_xoffs + j / l_xoffs / l_yoffs) * 100
                print('calculating %.3f' % percent, '%')
                # print('xoff=%d, yoff=%d', xoff, yoff)

                # read image to check if here is no value area
                image_data = ds.ReadAsArray(xoff, yoff, xsize=xsize, ysize=ysize)
                empty_num = np.sum(image_data == none_value)
                if crop_all or empty_num < image_data.size * 0.8:
                    sample_image_name = os.path.splitext(image_name)[0] + '_' + str(
                        scale) + '_' + '%.4d' % xoff + '_' + '%.4d' % yoff + '.' + data_type
                    if scale == 1.0:
                        sample_image_name = os.path.splitext(image_name)[0] + '_' + '%.4d' % xoff \
                                            + '_' + '%.4d' % yoff + '.' + data_type
                    print('making sample: ' + sample_image_name)
                    # for k, name in enumerate(name_list):
                    for k in zip(name_list, ot_list):
                        name = k[0]
                        out_type = k[1]
                        in_file = os.path.join(input_dirs[name], image_name)
                        out_file = os.path.join(output_dirs[name], sample_image_name)
                        crop_image(image_file=in_file,
                                   sample_image_file=out_file,
                                   xoff=xoff, yoff=yoff, xsize=xsize, ysize=ysize,
                                   compress_image=compress_image, scale=scale, resample_mode='nearest',
                                   outsize_x=outsize_x, outsize_y=outsize_y, ot=out_type)

def make_multiscale_sample(input_dir,
                           output_dir,
                           sample_width,
                           sample_height,
                           sample_stride,
                           scale_list,
                           crop_all,
                           none_value,
                           make_csv,
                           compress_image,
                           data_type,
                           name1, name2, name3, name4):
    for i, scale in enumerate(scale_list):
        crop_images_fixed_names(input_dir,
                                output_dir,
                                sample_width,
                                sample_height,
                                sample_stride,
                                scale,
                                crop_all,
                                none_value,
                                make_csv,
                                'w' if i == 0 else 'a',
                                compress_image,
                                data_type,
                                name1, name2, name3, name4)

def crop_image_by_ref(image_file: str,
                      ref_image_file: str,
                      out_imgae_file:str,
                      compress_image: bool):
    image_raster_io=RasterIO(image_file)
    image_trans=image_raster_io.get_geo_trans()
    ref_raster_io=RasterIO(ref_image_file)
    ref_trans=ref_raster_io.get_geo_trans()
    ref_shape=ref_raster_io.shape()

    xoff,yoff=geo_coordinate_to_img_coordinate(image_trans,ref_trans[0],ref_trans[3])
    xsize=ref_shape[2]
    ysize=ref_shape[1]
    del image_raster_io
    image_raster_io=None
    del ref_raster_io
    ref_raster_io=None

    cmd = ["gdal_translate", image_file, out_imgae_file, "-srcwin",
           str(xoff), str(yoff), str(xsize), str(ysize),
           # "-co", "COMPRESS=DEFLATE", #"-co", "PHOTOMETRIC=YCBCR",
           "-co", "TILED=YES",
           "--config", "GDAL_TIFF_INTERNAL_MASK YES"
           ]
    if compress_image:
        cmd.append("-co")
        cmd.append("COMPRESS=DEFLATE")
    strcmd = " ".join(cmd)
    os.system(strcmd)

def crop_images_by_refs_dir(image_dir: str,
                            ref_image_dir: str,
                            out_imgae_dir:str,
                            compress_image: bool):

    if not os.path.exists(out_imgae_dir):
        os.mkdir(out_imgae_dir)

    ref_image_files=glob.glob(ref_image_dir+'/*.tif')
    image_files=glob.glob(image_dir+'/*.tif')
    for image_file in image_files:
        # image_name=os.path.basename(image_file)
        # image_name=imgae_name.split(str="_")[0]
        regx=re.compile(r'\d+')
        num=regx.findall(image_file)[0]
        for ref_image in ref_image_files:
            ref_name =os.path.basename(ref_image)
            if num in ref_name:
                out_file=os.path.join(out_imgae_dir,ref_name)
                crop_image_by_ref(image_file,ref_image,out_file,compress_image)

def crop_multi_view_sample(xyz_files_dict,
                           xyz_out_dir_dict,
                           files_dict,
                           output_dir_dict,
                           sample_width: int = 512,
                           sample_height: int = 512,
                           sample_stride: int = 170,
                           none_value: float = 0,
                           compress_image: bool = True):
    """

    Args:
        xyz_files_dict: 物方数据字典{"image":str,"lable":str....}
        xyz_out_dir_dict:物方数据切割之后的输出路径{"image":str,"lable":str....}
        files_dict:与物方对应的像方数据组字典，第一个必须是图像，第二个是grid{"image":[str],"grid":[str],"label":[str]...}
        output_dir_dict:像方输出路径的字典{"image":str,"lable":str....}
        sample_width:切图的宽度
        sample_height:切图的高度
        sample_stride:切图步长
        none_value:无效值
        compress_image:是否压缩图像
    Returns:
        nothing
    """
    for value in output_dir_dict.values():
        if not os.path.exists(value):
            os.mkdir(value)

    for value in xyz_out_dir_dict.values():
        if not os.path.exists(value):
            os.mkdir(value)


    xyz_file=list(xyz_files_dict.values())[0]
    xyz_rasterio=RasterIO(xyz_file)


    xyz_shape=xyz_rasterio.shape()
    xoffs = list(range(0, xyz_shape[2] - sample_width, sample_stride))
    xoffs.append( xyz_shape[2]- sample_width)

    yoffs = list(range(0, xyz_shape[1] - sample_height, sample_stride))
    yoffs.append(xyz_shape[1] - sample_height)

    files_keys=list(files_dict.keys())

    xyz_image_name=os.path.basename(xyz_file)

    for xoff in xoffs:
        for yoff in yoffs:
            image_files=files_dict[files_keys[0]]
            grid_files=files_dict[files_keys[1]]
            for i,files_zip in enumerate(zip(image_files,grid_files)):
                image_file, grid_file=files_zip
                image_rasterio = RasterIO(image_file)
                image_shape = image_rasterio.shape()
                image_name = os.path.basename(image_file)
                sample_image_name = os.path.splitext(image_name)[0] +'_' + '%.4d' % xoff + '_' + '%.4d' % yoff + '.' + "tif"

                grid_rasterio = RasterIO(grid_file)
                grid_trans=grid_rasterio.get_geo_trans()
                grid_data = grid_rasterio.read_as_array()
                grid_data[0,:, :] = (image_shape[2] - 1) * (grid_data[0,:, :] + 1) / 2
                grid_data[1,:, :] = (image_shape[1] - 1) * (grid_data[1,:, :] + 1) / 2

                left_top_x,left_top_y=grid_data[0,yoff,xoff],grid_data[1,yoff,xoff]
                left_bottom_x,left_bottom_y=grid_data[0,yoff+sample_height,xoff],grid_data[1,yoff+sample_height,xoff]

                right_top_x,right_top_y=grid_data[0,yoff,xoff+sample_width],grid_data[1,yoff,xoff+sample_width]
                right_bottom_x,right_bottom_y=grid_data[0,yoff+sample_height,xoff+sample_width],\
                                              grid_data[1,yoff+sample_height,xoff+sample_width]

                contour=np.array([[left_top_x,left_top_y],[left_bottom_x,left_bottom_y],
                                       [right_top_x,right_top_y],[right_bottom_x,right_bottom_y]])

                # max_x,min_x=np.ceil(np.max(contour[:,0])),np.floor(np.min(contour[:,0]))
                # max_y,min_y=np.ceil(np.max(contour[:,1])),np.floor(np.min(contour[:,1]))
                #
                # image_sample_width=int(max_x-min_x)
                # image_sample_height=int(max_y-min_y)
                #
                # min_x = min_x if min_x > 0 else 0
                # min_y = min_y if min_y > 0 else 0
                # #
                # # max_x=max_x if max_x<image_shape[2] else image_shape[2]
                # # max_y=max_y if max_y<image_shape[1] else image_shape[1]
                # max_x = min_x + image_sample_width
                # max_y = min_y + image_sample_height
                # # crop grid
                # crop_grid_data = grid_data[:, yoff:yoff + sample_height, xoff:xoff + sample_width]
                # crop_grid_data[0, :, :] = 2 * (crop_grid_data[0, :, :] - min_x) / (max_x - min_x - 1) - 1
                # crop_grid_data[1, :, :] = 2 * (crop_grid_data[1, :, :] - min_y) / (max_y - min_y - 1) - 1
                #
                # geo_x, geo_y = grid_rasterio.proj_image_to_geo([[xoff, yoff]])[0]
                # crop_grid_trans = [geo_x, grid_trans[1], grid_trans[2], geo_y, grid_trans[4], grid_trans[5]]
                # fastly_save_image(crop_grid_data, os.path.join(output_dir_dict[files_keys[1]], sample_image_name),
                #                   geo_trans=crop_grid_trans,
                #                   gdal_type=grid_rasterio.get_gdal_data_type())
                #
                #
                # #crop image
                # out_crop_image=os.path.join(output_dir_dict[files_keys[0]],sample_image_name)
                # crop_image(image_file,out_crop_image,min_x,min_y,
                #            image_sample_width,image_sample_height,compress_image,1,"",0,0,ot=image_rasterio.get_data_type())

                max_x, min_x = np.ceil(np.max(contour[:, 0])), np.floor(np.min(contour[:, 0]))
                max_y, min_y = np.ceil(np.max(contour[:, 1])), np.floor(np.min(contour[:, 1]))

                # left_top_x=min_x
                # left_top_y=min_y
                min_x = min_x if min_x > 0 else 0
                min_y = min_y if min_y > 0 else 0
                max_x=max_x if max_x<image_shape[2] else image_shape[2]
                max_y=max_y if max_y<image_shape[1] else image_shape[1]
                image_sample_width = int(max_x - min_x)
                image_sample_height = int(max_y - min_y)

                # crop grid
                crop_grid_data = grid_data[:, yoff:yoff + sample_height, xoff:xoff + sample_width]
                crop_grid_data[0, :, :] = 2 * (crop_grid_data[0, :, :] - min_x) / (image_sample_width - 1) - 1
                crop_grid_data[1, :, :] = 2 * (crop_grid_data[1, :, :] - min_y) / (image_sample_height - 1) - 1

                geo_x, geo_y = grid_rasterio.proj_image_to_geo([[xoff, yoff]])[0]
                crop_grid_trans = [geo_x, grid_trans[1], grid_trans[2], geo_y, grid_trans[4], grid_trans[5]]
                fastly_save_image(crop_grid_data, os.path.join(output_dir_dict[files_keys[1]], sample_image_name),
                                  geo_trans=crop_grid_trans,
                                  gdal_type=grid_rasterio.get_gdal_data_type())

                # crop image
                out_crop_image = os.path.join(output_dir_dict[files_keys[0]], sample_image_name)
                crop_image(image_file, out_crop_image, min_x, min_y,
                           image_sample_width, image_sample_height, compress_image, 0.5, "bilinear",
                           sample_width, sample_height,
                           ot=image_rasterio.get_data_type())

                if len(files_keys)>2:
                    for key in files_keys[2:]:
                        temp_rasterio=RasterIO(files_dict[key][i])
                        out_file=os.path.join(output_dir_dict[key], sample_image_name)
                        crop_image(files_dict[key][i], out_file,
                                   min_x,min_y,image_sample_width, image_sample_height,
                                   compress_image, 0.5, "bilinear", sample_width, sample_height,
                                   ot=temp_rasterio.get_data_type())


            sample_image_name = os.path.splitext(xyz_image_name)[
                                    0] + '_' + '%.4d' % xoff + '_' + '%.4d' % yoff + '.' + "tif"
            for key,value in xyz_files_dict.items():
                xyz_sample_out_file=os.path.join(xyz_out_dir_dict[key],sample_image_name)
                temp_rasterio=RasterIO(value)
                crop_image(value,xyz_sample_out_file , xoff, yoff,
                        sample_width, sample_height, compress_image, 1, "", 0, 0,
                        ot=temp_rasterio.get_data_type())

def crop_multi_view_sample_dir(xyz_dir:str,
                               xyz_out_dir:str,
                               image_dir:str,
                               image_out_dir:str,
                               pair_file:str,
                               xyz_name_list=("image","dsm","label"),
                               image_name_list=("image","grid","dsm","label"),
                               sample_width: int = 512,
                               sample_height: int = 512,
                               sample_stride: int = 170,
                               none_value: float = 0,
                               compress_image: bool = True):
    """

    Args:
        xyz_dir: 物方数据目录
        xyz_out_dir: 物方输出结果目录
        image_dir: 像方数据目录
        image_out_dir: 像方输出目录
        pair_file: 物方和像方对应关系txt文件
        xyz_name_list: 物方需要切割的文件目录名字
        image_name_list: 像方需要切割的文件目录名字
        sample_width: 切图宽度
        sample_height: 切图高度
        sample_stride: 切图步长
        none_value: 无效值
        compress_image: 是否压缩图像

    Returns:

    """
    with open(pair_file, "r") as f:
        scans = f.readlines()
        scans = [line.replace("\t", " ").strip().split(" ") for line in scans]
        scans = np.array(scans)
        scans_dict = dict(zip(scans[:, 1], scans[:, 0]))



    output_dir_dict = {}
    for image_out_name in image_name_list:
        output_dir_dict[image_out_name]=os.path.join(image_out_dir,image_out_name)

    xyz_out_dir_dict  = {}
    for xyz_out_name in xyz_name_list:
        xyz_out_dir_dict[xyz_out_name] = os.path.join(xyz_out_dir, xyz_out_name)

    model_predict_dir=os.path.join(image_dir,image_name_list[0])
    model_predict_files = glob.glob(os.path.join(model_predict_dir, "*.tif"))

    model_predict_pair={}

    for model_predict_file in model_predict_files:
        name=os.path.basename(model_predict_file)
        model_predict_name=os.path.splitext(os.path.basename(model_predict_file))[0]
        num=model_predict_name.split("_")[-1]
        if num in model_predict_pair.keys():
            model_predict_pair[num].append(name)
        else:
            model_predict_pair[num]=[name]


    for key,value in model_predict_pair.items():
        xyz_name=scans_dict[key]+".tif"
        xyz_files_dict = {}
        for xyz_input_name in xyz_name_list:
            xyz_files_dict[xyz_input_name] = os.path.join(xyz_dir,xyz_input_name, xyz_name)

        files_dict = {}
        for image_name in image_name_list:
            files_dict[image_name] = [ os.path.join(image_dir,image_name,x) for x in value]

        crop_multi_view_sample(xyz_files_dict, xyz_out_dir_dict, files_dict, output_dir_dict,sample_width,
                               sample_height, sample_stride, none_value, compress_image)

def crop_multi_view_sample_with_colinear(xyz_files_dict,
                                         xyz_out_dir_dict,
                                         files_dict,
                                         corner_coordination_files,
                                         output_dir_dict,
                                         sample_width: int = 512,
                                         sample_height: int = 512,
                                         sample_stride: int = 170,
                                         none_value: float = 0,
                                         compress_image: bool = True,
                                         crop_all : bool = False,
                                         symbol=None,
                                         resize: bool=True):
    """

    Args:
        xyz_files_dict: 物方数据字典{"image":str,"lable":str....}
        xyz_out_dir_dict:物方数据切割之后的输出路径{"image":str,"lable":str....}
        files_dict:与物方对应的像方数据组字典，第一个必须是图像，第二个是grid{"image":[str],"grid":[str],"label":[str]...}
        corner_coordination_files:物方角点投影后在像方上的坐标的多个txt文件
        output_dir_dict:像方输出路径的字典{"image":str,"lable":str....}
        sample_width:切图的宽度
        sample_height:切图的高度
        sample_stride:切图步长
        none_value:无效值
        compress_image:是否压缩图像
    Returns:
        nothing
    """
    for value in output_dir_dict.values():
        if not os.path.exists(value):
            os.mkdir(value)

    for value in xyz_out_dir_dict.values():
        if not os.path.exists(value):
            os.mkdir(value)

    if resize:
        scale=-1
    else:
        scale=1
    xyz_file=list(xyz_files_dict.values())[0]
    xyz_rasterio=RasterIO(xyz_file)


    xyz_shape=xyz_rasterio.shape()
    xoffs = list(range(0, xyz_shape[2] - sample_width, sample_stride))
    xoffs.append( xyz_shape[2]- sample_width)

    yoffs = list(range(0, xyz_shape[1] - sample_height, sample_stride))
    yoffs.append(xyz_shape[1] - sample_height)

    files_keys=list(files_dict.keys())

    xyz_image_name=os.path.basename(xyz_file)

    #reading corner coordinationk
    coordinate_list=[]
    for corner_coordination_file in corner_coordination_files:
        with open(corner_coordination_file, "r") as f:
            scans = f.readlines()
            scans=scans[1:]
            # scans = [line.replace("\n", " ").strip().split(" ") for line in scans]
            scans=[line.rstrip().strip().split(" ")  for line in scans if line.strip()]
            for i,line in enumerate(scans):
                line=[ float(l) for l in line]
                scans[i]=line
        coordinate_list.append(scans)

    image_files = files_dict[files_keys[0]]
    grid_files = files_dict[files_keys[1]]

    image_files,grid_files=filte_exist(image_files,grid_files)

    for i, files_zip in enumerate(zip(image_files, grid_files)):
        image_file, grid_file = files_zip
        image_rasterio = RasterIO(image_file)
        image_shape = image_rasterio.shape()
        image_name = os.path.basename(image_file)

        if symbol!=None:
            if symbol not in image_name:
                continue
        
        grid_rasterio = RasterIO(grid_file)
        grid_trans = grid_rasterio.get_geo_trans()
        grid_data = grid_rasterio.read_as_array()
        #转换到以图像左上角为原点的坐标系下
        grid_data[0, :, :] = (image_shape[2] - 1) * (grid_data[0, :, :] + 1) / 2
        grid_data[1, :, :] = (image_shape[1] - 1) * (grid_data[1, :, :] + 1) / 2

        n=0
        for xoff in xoffs:
            for yoff in yoffs:
                sample_image_name = os.path.splitext(image_name)[
                                        0] + '_' + '%.4d' % xoff + '_' + '%.4d' % yoff + '.' + "tif"
                out_crop_image = os.path.join(output_dir_dict[files_keys[0]], sample_image_name)
                
                if os.path.exists(out_crop_image):
                    n+=1
                    continue
                
                image_sample_width,image_sample_height,\
                min_x,min_y=get_crop_coordinate_from_xyz_corner_colinear(coordinate_corners=coordinate_list[i][n],
                                                                         image_shape=image_shape)

                if min_x==min_y==image_sample_width==image_sample_height:
                    n+=1
                    continue
                if min_x<0 or min_y<0 or image_sample_width<0 or image_sample_height<0:
                    n+=1
                    continue
                image_data = image_rasterio.read_as_given_xyhw(min_x, min_y,image_sample_height,image_sample_width)
                # crop image

                empty_num = np.sum(image_data == none_value)
                if crop_all or empty_num < image_data.size * 0.8:

                    
                    crop_image(image_file, out_crop_image, min_x, min_y,
                               image_sample_width, image_sample_height, compress_image, scale, "bilinear",
                               sample_width, sample_height,
                               ot=image_rasterio.get_data_type())

                    # crop grid
                    geo_x, geo_y = grid_rasterio.proj_image_to_geo([[xoff, yoff]])[0]
                    crop_grid_trans = [geo_x, grid_trans[1], grid_trans[2], geo_y, grid_trans[4], grid_trans[5]]

                    crop_grid_data = grid_data[:, yoff:yoff + sample_height, xoff:xoff + sample_width].copy()
                    crop_grid_data[0, :, :] = 2 * (crop_grid_data[0, :, :] - min_x) / (image_sample_width - 1) - 1
                    crop_grid_data[1, :, :] = 2 * (crop_grid_data[1, :, :] - min_y) / (image_sample_height - 1) - 1

                    fastly_save_image(crop_grid_data, os.path.join(output_dir_dict[files_keys[1]], sample_image_name),
                                    geo_trans=crop_grid_trans,
                                    gdal_type=grid_rasterio.get_gdal_data_type())

                    if len(files_keys) > 2:
                        for key in files_keys[2:]:
                            temp_rasterio = RasterIO(files_dict[key][i])
                            out_file = os.path.join(output_dir_dict[key], sample_image_name)
                            crop_image(files_dict[key][i], out_file,
                                    min_x, min_y, image_sample_width, image_sample_height,
                                    compress_image, scale, "bilinear", sample_width, sample_height,
                                    ot=temp_rasterio.get_data_type())


                    sample_image_name = os.path.splitext(xyz_image_name)[
                                            0] + '_' + '%.4d' % xoff + '_' + '%.4d' % yoff + '.' + "tif"
                    for key,value in xyz_files_dict.items():
                        xyz_sample_out_file=os.path.join(xyz_out_dir_dict[key],sample_image_name)
                        if os.path.exists(xyz_sample_out_file):
                            break
                        temp_rasterio=RasterIO(value)
                        crop_image(value,xyz_sample_out_file , xoff, yoff,
                                sample_width, sample_height, compress_image, 1, "", 0, 0,
                                ot=temp_rasterio.get_data_type())
                n+=1

def crop_multi_view_sample_with_colinear_dir(xyz_dir:str,
                                             xyz_out_dir:str,
                                             image_dir:str,
                                             image_out_dir:str,
                                             pair_file:str,
                                             corner_coordination_file_dir:str,
                                             xyz_name_list=("image","dsm","label"),
                                             image_name_list=("image","grid","dsm","label"),
                                             sample_width: int = 512,
                                             sample_height: int = 512,
                                             sample_stride: int = 170,
                                             none_value: float = 0,
                                             compress_image: bool = True,
                                             crop_all : bool = False,
                                             symbol=None,
                                             resize:bool=True):
    """

    Args:
        xyz_dir: 物方数据目录
        xyz_out_dir: 物方输出结果目录
        image_dir: 像方数据目录
        image_out_dir: 像方输出目录
        pair_file: 物方和像方对应关系txt文件
        corner_coordination_file_dir：物方角点投影后在像方上的坐标的txt文件目录
        xyz_name_list: 物方需要切割的文件目录名字
        image_name_list: 像方需要切割的文件目录名字
        sample_width: 切图宽度
        sample_height: 切图高度
        sample_stride: 切图步长
        none_value: 无效值
        compress_image: 是否压缩图像

    Returns:

    """
    with open(pair_file, "r") as f:
        scans = f.readlines()
        scans = [line.replace("\t", " ").strip().split(" ") for line in scans]
        scans = np.array(scans)
        scans_dict = dict(zip(scans[:, 1], scans[:, 0]))

    if not os.path.exists(xyz_out_dir):
        os.makedirs(xyz_out_dir)
    if not os.path.exists(image_out_dir):
        os.makedirs(image_out_dir)

    output_dir_dict = {}
    for image_out_name in image_name_list:
        output_dir_dict[image_out_name]=os.path.join(image_out_dir,image_out_name)

    xyz_out_dir_dict  = {}
    for xyz_out_name in xyz_name_list:
        xyz_out_dir_dict[xyz_out_name] = os.path.join(xyz_out_dir, xyz_out_name)

    model_predict_dir=os.path.join(image_dir,image_name_list[0])
    model_predict_files = glob.glob(os.path.join(model_predict_dir, "*.tif"))

    model_predict_pair={}

    for model_predict_file in model_predict_files:
        name=os.path.basename(model_predict_file)
        model_predict_name=os.path.splitext(os.path.basename(model_predict_file))[0]
        num=model_predict_name.split("_")[-1]
        if num in model_predict_pair.keys():
            model_predict_pair[num].append(name)
        else:
            model_predict_pair[num]=[name]


    for key,value in model_predict_pair.items():
        xyz_name=scans_dict[key]+".tif"
        xyz_files_dict = {}
        for xyz_input_name in xyz_name_list:
            xyz_files_dict[xyz_input_name] = os.path.join(xyz_dir,xyz_input_name, xyz_name)

        files_dict = {}
        for image_name in image_name_list:
            files_dict[image_name] = [ os.path.join(image_dir,image_name,x) for x in value]

        corner_coordination_files=[ os.path.join(corner_coordination_file_dir,x.split(".")[0]+".txt") for x in value]


        crop_multi_view_sample_with_colinear(xyz_files_dict, xyz_out_dir_dict, files_dict,corner_coordination_files,
                                             output_dir_dict,sample_width,sample_height, sample_stride,
                                             none_value, compress_image,crop_all,symbol,resize)

if __name__ == "__main2__":
    parser = argparse.ArgumentParser(description='Sample Generator')
    parser.add_argument('-input_dir', type=str, default='/path/to/user/data3/1Data_back/Huawei/Hangzhou/',
                        help='working directory with data prepared')
    parser.add_argument('-output_dir', type=str, default='/path/to/user/data3/Huawei_Hangzhou//train_2_samples',
                        help='sample output path')
    parser.add_argument('-sample_width', type=int, default=512,
                        help='sample_width of output data')
    parser.add_argument('-sample_height', type=int, default=512,
                        help='sample_height of output data')
    parser.add_argument('-sample_stride', type=int, default=256,
                        help='sample_stride of output data')
    parser.add_argument('-scale_list', type=int, default=[1],
                        help='')
    parser.add_argument('-crop_all', type=int, default=False,
                        help='')
    parser.add_argument('-none_value', type=int, default=0,
                        help='none data value')
    parser.add_argument('-make_csv', type=int, default=False,
                        help='')
    parser.add_argument('-compress_image', type=int, default=True,
                        help='')
    parser.add_argument('-data_type', type=str, default='tif',
                        help='')
    parser.add_argument('-name_list', type=tuple,
                        default=('image', 'label_CBO', 'label_dx', 'label_dy', 'label_footprint', 'label_roof',
                                 'label_roof_wall'), help='')
    parser.add_argument('-name1', type=str, default='TDOM',
                        help='file name1')
    parser.add_argument('-name2', type=str, default='label',
                        help='file name2')
    parser.add_argument('-name3', type=str, default='DSM',
                        help='file name3')
    parser.add_argument('-name4', type=str, default='',
                        help='file name4')
    args = parser.parse_args()
    # make_multiscale_sample(args.input_dir,
    #                        args.output_dir,
    #                        args.sample_width,
    #                        args.sample_height,
    #                        args.sample_stride,
    #                        args.scale_list,
    #                        args.crop_all,
    #                        args.none_value,
    #                        args.make_csv,
    #                        args.compress_image,
    #                        args.data_type,
    #                        args.name1, args.name2, args.name3, args.name4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sample Generator')
    parser.add_argument('--xyz_dir', type=str, default=r"/path/to/project/ChenLabDataset\Penta-Cam-Centre\0_Penta_DeepL\Penta_XYZ\train")
    parser.add_argument('--xyz_out_dir', type=str, default=r"/path/to/project/ChenLabDataset\Penta-Cam-Centre\0_Penta_DeepL\Penta_XYZ\train\samples_dsm_512_512_512")
    parser.add_argument('--image_dir', type=str, default=r"/path/to/project/ChenLabDataset\Penta-Cam-Centre\0_Penta_DeepL\Penta_IMG_MV\train")
    parser.add_argument('--image_out_dir', type=str, default=r"/path/to/project/ChenLabDataset\Penta-Cam-Centre\0_Penta_DeepL\Penta_IMG_MV\train\samples_dsm_512_512_512")
    parser.add_argument('--pair_file', type=str, default=r"/path/to/project/ChenLabDataset\Penta-Cam-Centre\0_Penta_DeepL\XYZ_xy_name_map.txt")
    parser.add_argument('--corner_coordination_file_dir', type=str, default=r"/path/to/project/ChenLabDataset\Penta-Cam-Centre\0_Penta_DeepL\Penta_IMG_MV\train\connertxt_Dsm_512_512_512")
    parser.add_argument('--sample_width', type=int, default=170)
    parser.add_argument('--sample_height', type=int, default=170)
    parser.add_argument('--sample_stride', type=int, default=170)
    parser.add_argument('--symbol', default=None)
    parser.add_argument('--resize', default=True)
    args = parser.parse_args()
    # crop_images_by_refs_dir(r"/path/to/project/Xuzhou-Aerial\Point_pairs",r"/path/to/project/Xuzhou-Aerial\0_Xuzhou_DeepL\Xuzhou-IMG_MV\test\dsm",
    #                         r"/path/to/project/Xuzhou-Aerial\0_Xuzhou_DeepL\Xuzhou-IMG_MV\test\point_pairs",True)

    # crop_image(r"/path/to/project/Xuzhou-Aerial\0_Xuzhou_DeepL\Xuzhou-XYZ\train\image\train_xyz_0.tif",
    #            r"/path/to/project/Xuzhou-Aerial\0_Xuzhou_DeepL\Xuzhou-XYZ\train\image\111.tif",0,0,512,512,True,1,"",0,0)

    # crop_image(r"/path/to/project/Xuzhou-Aerial\0_Xuzhou_DeepL\Xuzhou-IMG_MV\train\image\x029517_train_01.tif",
    #            r"/path/to/project/Xuzhou-Aerial\0_Xuzhou_DeepL\Xuzhou-XYZ\train\image\222.tif",0,0,542,534,True,1,"",0,0)
    # xyz_files_dict={"image":r"/path/to/project/Xuzhou-Aerial\0_Xuzhou_DeepL\Xuzhou-XYZ\train\image\train_xyz_0.tif",
    #                  "dsm":r"/path/to/project/Xuzhou-Aerial\0_Xuzhou_DeepL\Xuzhou-XYZ\train\dsm\train_xyz_0.tif",
    #                  "label":r"/path/to/project/Xuzhou-Aerial\0_Xuzhou_DeepL\Xuzhou-XYZ\train\label\train_xyz_0.tif"}
    #
    # xyz_out_dir_dict={"image":r"/path/to/project/Xuzhou-Aerial\0_Xuzhou_DeepL\Xuzhou-XYZ\train\samples\image",
    #                  "dsm":r"/path/to/project/Xuzhou-Aerial\0_Xuzhou_DeepL\Xuzhou-XYZ\train\samples\dsm",
    #                  "label":r"/path/to/project/Xuzhou-Aerial\0_Xuzhou_DeepL\Xuzhou-XYZ\train\samples\label"}
    #
    # files_dict={"image":[r"/path/to/project/Xuzhou-Aerial\0_Xuzhou_DeepL\Xuzhou-IMG_MV\train\image\x029517_train_01.tif",],
    #             "grid":[r"/path/to/project/Xuzhou-Aerial\0_Xuzhou_DeepL\Xuzhou-IMG_MV\train\grid\x029517_train_01.tif"],
    #             "dsm":[r"/path/to/project/Xuzhou-Aerial\0_Xuzhou_DeepL\Xuzhou-IMG_MV\train\dsm\x029517_train_01.tif"],
    #             "label":[r"/path/to/project/Xuzhou-Aerial\0_Xuzhou_DeepL\Xuzhou-IMG_MV\train\label\x029517_train_01.tif"]}
    # output_dir_dict={"image":r"/path/to/project/Xuzhou-Aerial\0_Xuzhou_DeepL\Xuzhou-IMG_MV\train\samples_colinear\image",
    #                  "grid":r"/path/to/project/Xuzhou-Aerial\0_Xuzhou_DeepL\Xuzhou-IMG_MV\train\samples_colinear\grid",
    #                  "dsm":r"/path/to/project/Xuzhou-Aerial\0_Xuzhou_DeepL\Xuzhou-IMG_MV\train\samples_colinear\dsm",
    #                  "label":r"/path/to/project/Xuzhou-Aerial\0_Xuzhou_DeepL\Xuzhou-IMG_MV\train\samples_colinear\label"}
    # # crop_multi_view_sample( xyz_files_dict,xyz_out_dir_dict,files_dict, output_dir_dict)
    #
    #
    # corner_coordination_file=[r"/path/to/project/Users\35416\Desktop\cornertxts\x029517_train_01.txt"]
    # crop_multi_view_sample_with_colinear( xyz_files_dict,xyz_out_dir_dict,files_dict, corner_coordination_file,output_dir_dict)

    # xyz_dir="/path/to/user/NAS/gwx/DeepL/Xuzhou/XYZ/test/"
    # xyz_out_dir="/path/to/user/NAS/gwx/DeepL/Xuzhou/XYZ/test/samples/"
    # image_dir="/path/to/user/NAS/gwx/DeepL/Xuzhou/IMG_MV/test/"
    # image_out_dir="/path/to/user/NAS/gwx/DeepL/Xuzhou/IMG_MV/test/sample_spatial_trans/"
    # pair_file="/path/to/user/NAS/gwx/DeepL/Xuzhou/IMG_MV/XYZ_xy_name_map.txt"

    # crop_multi_view_sample_dir(xyz_dir, xyz_out_dir, image_dir, image_out_dir, pair_file)


    # xyz_dir=r"/path/to/project/ChenLabDataset\Penta-Cam-Centre\0_Penta_DeepL\Penta_XYZ\train"
    # xyz_out_dir=r"/path/to/project/ChenLabDataset\Penta-Cam-Centre\0_Penta_DeepL\Penta_XYZ\train\samples"
    # image_dir=r"/path/to/project/ChenLabDataset\Penta-Cam-Centre\0_Penta_DeepL\Penta_IMG_MV\train"
    # image_out_dir=r"/path/to/project/ChenLabDataset\Penta-Cam-Centre\0_Penta_DeepL\Penta_IMG_MV\train\samples_512_512_512"
    # pair_file=r"/path/to/project/ChenLabDataset\Penta-Cam-Centre\0_Penta_DeepL\XYZ_xy_name_map.txt"
    # corner_coordination_file_dir=r"/path/to/project/ChenLabDataset\Penta-Cam-Centre\0_Penta_DeepL\Penta_IMG_MV\train\cornertxt_512_512_512"

    xyz_dir = r"/path/to/project/ChenLabDataset\JKW\0_DeepLearning\XYZ\Crop"
    xyz_out_dir = r"/path/to/project/ChenLabDataset\JKW\0_DeepLearning\XYZ\Crop\samples_170x170_170"
    image_dir = r"/path/to/project/ChenLabDataset\JKW\0_DeepLearning\IMG_MV\Crop"
    image_out_dir = r"/path/to/project/ChenLabDataset\JKW\0_DeepLearning\IMG_MV\Crop\samples_crop_with_corner_noresize_170x170_170"
    pair_file = r"/path/to/project/ChenLabDataset\JKW\0_DeepLearning\XYZ_xy_name_map.txt"
    corner_coordination_file_dir=r"/path/to/project/ChenLabDataset\JKW\0_DeepLearning\IMG_MV\Crop\corner_global2local_170x170_170"

    # xyz_dir=args.xyz_dir
    # xyz_out_dir=args.xyz_out_dir
    # image_dir=args.image_dir
    # image_out_dir=args.image_out_dir
    # pair_file=args.pair_file
    # corner_coordination_file_dir=args.corner_coordination_file_dir
    sample_width = args.sample_width
    sample_height = args.sample_height
    sample_stride = args.sample_stride
    symbol = args.symbol
    resize = False
    crop_multi_view_sample_with_colinear_dir(xyz_dir, xyz_out_dir, image_dir, image_out_dir, pair_file,
                                             corner_coordination_file_dir,image_name_list=("image","grid_none","dsm","label"),
                                             sample_width=sample_width,sample_height=sample_height,sample_stride=sample_stride,
                                             symbol = symbol,resize = resize)

    # input_dir ='/path/to/user/NAS/gwx/DeepL/Xuzhou/IMG_MV/train/grid_ransac'
    # output_dir ='/path/to/user/NAS/gwx/DeepL/Xuzhou/IMG_MV/train/samples_none/grid_ransac'
    # data_type = 'tif'
    # ot = 'Float32'
    # sample_width = 512
    # sample_height = 512
    # sample_stride = 170
    # none_value = 0.0
    # scale = 1.0
    # crop_all = True
    # compress_image = True
    # crop_image_dir(input_dir,output_dir,data_type,ot,
    #             sample_width,sample_height,sample_stride,
    #             none_value,scale,crop_all,compress_image)

    # parser = argparse.ArgumentParser(description='Sample Generator')
    # parser.add_argument('--input_dir', type=str, default=r"/model/zhangpl/mvmaper/experiment_result/grid_all_point/")
    # parser.add_argument('--output_dir', type=str, default=r"/model/zhangpl/mvmaper/experiment_result/grid_all_point_sample/")

    # parser.add_argument('--sample_width', type=int, default=512)
    # parser.add_argument('--sample_height', type=int, default=512)
    # parser.add_argument('--sample_stride', type=int, default=512)
    # parser.add_argument('--symbol', default=None)
    # args = parser.parse_args()
    # input_dir=args.input_dir
    # output_dir=args.output_dir
    # sample_width=args.sample_width
    # sample_height=args.sample_height
    # sample_stride=args.sample_stride
    # symbol=args.symbol
    # crop_image_dir(input_dir=input_dir,output_dir=output_dir,
    #             sample_height=sample_height,sample_width=sample_width,sample_stride=sample_stride,symbol=symbol)
