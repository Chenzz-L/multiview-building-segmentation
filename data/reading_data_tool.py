import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from typing import Optional,Union,List,Tuple

import pandas as pd
import numpy as np

from utils.raster_io import RasterIO

__all__=['read_stat_dsm','read_stat_dom','image_open']
def read_stat_dsm(stat_file):
    assert os.path.isfile(stat_file)
    with open(stat_file, "r") as f:
        lines = f.readlines()
        if len(lines) != 3 or len(lines[0].split(',')) != 3 or len(lines[1].split(',')) != 3 \
                or len(lines[2].split(',')) != 4:
            raise Exception("stat_file error")
        rgb_mean = (float(lines[0].split(',')[0].strip()),
                         float(lines[0].split(',')[1].strip()),
                         float(lines[0].split(',')[2].strip()))
        rgb_std = (float(lines[1].split(',')[0].strip()),
                        float(lines[1].split(',')[1].strip()),
                        float(lines[1].split(',')[2].strip()))
        dsm_stat = (float(lines[2].split(',')[0].strip()),
                         float(lines[2].split(',')[1].strip()),
                         float(lines[2].split(',')[2].strip()),
                         float(lines[2].split(',')[3].strip()))
        return rgb_mean,rgb_std,dsm_stat

def read_stat_dom(stat_file):
    assert os.path.isfile(stat_file)
    with open(stat_file, "r") as f:
        lines = f.readlines()
        if len(lines) < 2 or len(lines[0].split(',')) < 3 or len(lines[1].split(',')) < 3:
            raise Exception("stat_file error")
        rgb_mean = (float(lines[0].split(',')[0].strip()),
                         float(lines[0].split(',')[1].strip()),
                         float(lines[0].split(',')[2].strip()))
        rgb_std = (float(lines[1].split(',')[0].strip()),
                        float(lines[1].split(',')[1].strip()),
                        float(lines[1].split(',')[2].strip()))
        return rgb_mean,rgb_std

def calculate_stat_dsm(indices:pd.DataFrame,stat_file):
    r"""
    calculate mean value and standard deviation of rgb channel, and save stat_file
    :return: nothing
    """
    rgb_mean = (indices['r_mean'].mean(), indices['g_mean'].mean(), indices['b_mean'].mean())
    rgb_std = (indices['r_std'].mean(), indices['g_std'].mean(), indices['b_std'].mean())
    dsm_stat = (indices['d_mean'].mean(), indices['d_std'].mean(),
                     max(indices['d_max']), min(indices['d_min']))
    info_file = open(stat_file, 'w')
    info_file.write('%f, %f, %f\n' % (rgb_mean[0], rgb_mean[1], rgb_mean[2]))
    info_file.write('%f, %f, %f\n' % (rgb_std[0], rgb_std[1], rgb_std[2]))
    info_file.write('%f, %f, %f ,%f\n' % (dsm_stat[0], dsm_stat[1], dsm_stat[2], dsm_stat[3]))

    return rgb_mean,rgb_std,dsm_stat

def calculate_stat_dom(indices:pd.DataFrame,stat_file):
    r"""
    calculate mean value and standard deviation of rgb channel, and save stat_file
    :return: nothing
    """
    rgb_mean = (indices['r_mean'].mean(), indices['g_mean'].mean(), indices['b_mean'].mean())
    rgb_std = (indices['r_std'].mean(), indices['g_std'].mean(), indices['b_std'].mean())
    info_file = open(stat_file, 'w')
    info_file.write('%f, %f, %f\n' % (rgb_mean[0], rgb_mean[1], rgb_mean[2]))
    info_file.write('%f, %f, %f\n' % (rgb_std[0], rgb_std[1], rgb_std[2]))

    return rgb_mean,rgb_std

def calculate_stat_PAN_DSM(indices:pd.DataFrame,stat_file):
    mean = indices['mean'].mean()
    std = indices['std'].mean()
    dsm_stat = (indices['d_mean'].mean(), indices['d_std'].mean(),
                     max(indices['d_max']), min(indices['d_min']))
    info_file = open(stat_file, 'w')
    info_file.write('%f, %f\n' % (mean,std))
    info_file.write('%f, %f, %f ,%f\n' % (dsm_stat[0], dsm_stat[1], dsm_stat[2], dsm_stat[3]))


def image_open(data_path,channel_sum=None,scale:Optional[float]=None,
             standardization=False,mean:Optional[Union[Tuple[float],float]]=None,
             std:Optional[Union[Tuple[float],float]]=None,
             mask=False,mask_value:Optional[float]=None,none_value:Optional[float]=None,
             xoff:Optional[int]=None,yoff:Optional[int]=None,
             w:Optional[int]=None,h:Optional[int]=None):        
        raster_io = RasterIO()
        raster_io.open(data_path)
        resize=False
        if scale:
            image_data = raster_io.read_as_array_with_scale(scale).astype('float32')
        if xoff!=None and yoff!=None and w!=None and h!=None:
            image_data = raster_io.read_as_given_xyhw(xoff,yoff,h,w)
        else:
            image_data = raster_io.read_as_array().astype('float32')
            
        if image_data.ndim == 2:
            image_data = np.expand_dims(image_data, 0)
        if not channel_sum is None:
            image_data = image_data[:min(image_data.shape[0],channel_sum), :, :]
        
        if mask and mask_value!=None:
            nodata_mask = np.zeros_like(image_data)
            nodata_mask[image_data == mask_value] = 0
            image_data[image_data == mask_value] = 0
            image_data = np.concatenate((image_data, nodata_mask), axis=0)
        
        if none_value is not None and none_value!=0:
            image_data[image_data == none_value] = 0
        
        if standardization and not mean is None and not std is None:
            if isinstance(mean,Tuple):
                assert len(mean)==image_data.shape[0],'len of mean {} not equal len image {}'.format(len(mean),image_data.shape[0])
                for i in range(image_data.shape[0]):
                    image_data[i] = ((image_data[i] - mean[i]) / std[i]).astype('float32')
            else:
                mean_vec = np.array(mean).reshape(-1, 1, 1)
                std_vec = np.array(std).reshape(-1, 1, 1)
                image_data=((image_data - mean_vec) / std_vec ).astype('float32')
                
        return image_data

if __name__ == '__main__':
    indices = pd.read_csv(r"/path/to/nas/mvmapper_datamake/Melbourne_v3/IMG_MV/Crop/indices.csv")
    # calculate_stat_PAN_DSM(indices,r'/path/to/nas/mvmapper_datamake/Melbourne_v3/IMG_MV/Crop/stat_binary.txt')

    indices = pd.read_csv(r"/path/to/nas/mvmapper_datamake/Melbourne_v3/XYZ_PANSharpen/Crop/indices.csv")
    calculate_stat_dsm(indices,r'/path/to/nas/mvmapper_datamake/Melbourne_v3/XYZ_PANSharpen/Crop/stat_binary.txt')
