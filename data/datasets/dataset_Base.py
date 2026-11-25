# -*- coding: utf-8 -*-
import os
from typing import Tuple,Optional,Union,List
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import pandas as pd
import numpy as np
import torch.utils.data as tdata
from data.transforms.augmentation import build_agument_pipeline
from utils.make_samples import make_pair_with_indices
from utils.raster_io import RasterIO

__all__ = ['BaseDataset']


class BaseDataset(tdata.Dataset):
    def __init__(self,
                 data_dir: str,
                 augmentation: bool = True,
                 standardization: bool = True,
                #  mean: list = None,
                #  std: list = None,
                 clip: list = None,
                 data_pipeline=None,
                 indices_name=None):
        r"""
        note : data_pipeline生成的时候已经进行了normalize，RGB不需要在外面进行normalize
        :param data_dir: root dir of dataset, including image, label inside
        :param augmentation: if need data augmentation
        :param clip: clip indices if needed
        usage:
            train_set = SegmentationDataset(data_dir=trainset_dir)
            # directly set statics from outside
            train_set.set_stat(rgb_mean, rgb_std)
            # read stat if statics not given
                if os.path.exists(txt_file):
                    train_set.read_stat(txt_file)
            # calculate stat
            train_set.calculate_stat()
            # balance_samples
            self.add_positive_samples(positive_ratio_thresh=0.2, balance_rate=0.7, str_pr='positive_ratio')
        """
        assert os.path.exists(data_dir)
        self.data_dir = data_dir
        self._image_dir = os.path.join(data_dir, 'image')
        self._label_dir = os.path.join(data_dir, 'label')
        # self.data_dirs={'image':'image','label':'label'}
        self.augmentation = augmentation
        # get indices, return pd.DataFrame, a python dictionary
        if indices_name!=None:
            self.indices = pd.read_csv(os.path.join(data_dir, f'{indices_name}.csv'))
            self.ids = self.indices['name'].tolist()
        # else:
        #     self.indices = pd.read_csv(os.path.join(data_dir, 'indices.csv'))
        # calculate statics: mean value, standard
        self.standardization = standardization

        # 当标准化文件不存在时，计算该参数
        # 若存在则从中读取该参数
        # if self.standardization:
        #     # self.stat_file = self.data_dir + '/' + 'stat_binary.txt'
        #     # if not os.path.exists(self.stat_file):
        #     #     self.stat_data=self.calculate_stat()
        #     #     # self.stat_data=self.read_stat(self.stat_file)
        #     # else:
        #     #     self.stat_data=self.read_stat(self.stat_file)
        #     self.mean=mean
        #     self.std=std
        # pd.Series to python list

        # clip
        if clip is not None:
            assert len(clip) == 2
            self.ids = self.ids[clip[0]:clip[1]]
            
        self.data_pipeline=None
        if augmentation and data_pipeline:
            if not 'ToFloat' in data_pipeline[0].values():# 自定义ToFloat
                data_pipeline.insert(0,{'type':'ToFloat','max_value':1})
            self.data_pipeline=build_agument_pipeline(data_pipeline)
        else:
            self.augmentation=False
    def open(self,data_path,channel_sum=None,scale:Optional[float]=None,
             standardization=False,mean:Optional[Union[Tuple[float],float]]=None,standardization_method:Optional[str]='Z-Score',
             std:Optional[Union[Tuple[float],float]]=None,
             mask=False,mask_value:Optional[float]=None,none_value:Optional[float]=None):
        raster_io = RasterIO()
        raster_io.open(data_path)
        if scale:
            image_data = raster_io.read_as_array_with_scale(scale).astype('float32')
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

        # if standardization and not mean is None and not std is None:
        #     if isinstance(mean,(Tuple,List)):
        #         assert len(mean)==image_data.shape[0],'len of mean {} not equal len image {}'.format(len(mean),image_data.shape[0])
        #         for i in range(image_data.shape[0]):
        #             image_data[i] = ((image_data[i] - mean[i]) / std[i]).astype('float32')
        #     else:
        #             image_data=((image_data - mean) / std ).astype('float32')
        if standardization and not mean is None and not std is None:
            # mean=np.array(mean)[:,None,None]
            if isinstance(mean, (int, float)):
                mean = np.array(mean)[None, None, None]
            else:
                mean=np.array(mean)[:,None,None]
            if isinstance(std, (int, float)):
                std = np.array(std)[None, None, None]
            else:
                std=np.array(std)[:,None,None]
            if standardization_method=='Z-Score':
                image_data=((image_data-mean)/std).astype('float32')
            elif standardization_method=='Min-Max':
                image_data=((image_data-mean)/(std-mean)).astype('float32')
        return image_data
        
    @staticmethod
    def get_pn_pixel_num(indices: pd.DataFrame, str_pr: str = 'positive_ratio') -> Tuple[int, int, float, float]:
        """
        function: get positive pixels num and negative pixels num in the whole samples
        return: positive_pixel_num, negative_pixel_num, positive_pixel_rate, negative_pixel_rate
        """
        p = 0
        n = 0
        # print()
        for i in range(len(indices[str_pr])):
            p += indices[str_pr][i]  # indices['height'][i] * indices['width'][i] *
            n += (1 - indices[str_pr][i])  # indices['height'][i] * indices['width'][i] *
        return p, n, p / (p + n), n / (p + n)

    @staticmethod
    def get_pn_sample_num(indices: pd.DataFrame, thresh: float = 0, str_pr: str = 'positive_ratio'):
        """
        function: if positive_ratio of one sample is higher than thresh, this sample is positive sample
        return: positive_sample_num, negative_sample_num, positive_sample_rate, negative_sample_rate
        """
        p = sum(i > thresh for i in indices[str_pr])
        n = len(indices[str_pr]) - p
        # return p/(p+n), n/(p+n)
        return p, n, p / (p + n), n / (p + n)

    def balance_samples(self,
                        positive_ratio_thresh: float = 0.20,
                        balance_rate: float = 0.70,
                        str_pr: str = 'positive_ratio') -> None:
        """
        add positive samples whose positive_ratio is higher than 'positive_ratio_thresh'
        until total positive sample rate is higher than 'balance_rate'.
        self.indices will be updated.
        :param positive_ratio_thresh: positive pixel rate thresh, judge as positive sample if higher
        :param balance_rate: add positive samples until positive sample ratio is higher than balance_rate
        :param str_pr: name of csv column which represent label positive ratio
        """
        p_sample_num, n_sample_num, p_sample_rate, n_sample_rate = \
            self.get_pn_sample_num(self.indices, positive_ratio_thresh, str_pr)
        print('get_pn_sample_num:', p_sample_num, n_sample_num, p_sample_rate, n_sample_rate)
        p, n, pr, nr = self.get_pn_pixel_num(self.indices, str_pr)
        print('get_pn_pixel_num:', p, n, pr, nr)
        print('origin length of indices:', len(self.indices))
        if p_sample_rate < balance_rate:
            positive_indices = self.indices.loc[self.indices[str_pr] > positive_ratio_thresh].copy()
            len_positive_indices = len(positive_indices)
            print('length of positive indices', len_positive_indices)
            # supplement
            sup_num = n_sample_num * balance_rate / (1 - balance_rate) - p_sample_num
            n = sup_num / len_positive_indices  #
            n_int = int(n)  # n round up, int(4.5) = 4
            # print(n)
            if n > 1:
                for i in range(n_int):
                    self.indices = self.indices.append(positive_indices, ignore_index=True)
                self.indices = self.indices.append(positive_indices.head(int((n - n_int) * len_positive_indices)),
                                                   ignore_index=True)
            else:
                self.indices = self.indices.append(positive_indices.head(int(n * len_positive_indices)),
                                                   ignore_index=True)
            # validate
            print('after balance_samples')
            p, n, pr, nr = self.get_pn_sample_num(self.indices, positive_ratio_thresh, str_pr)
            print('get_pn_sample_num:', p, n, pr, nr)
            p, n, pr, nr = self.get_pn_pixel_num(self.indices, str_pr)
            print('get_pn_pixel_num:', p, n, pr, nr)
            print('length of indices after adding samples', len(self.indices))
        else:
            print('need not balance')

    def set_dir(self, image_dir_name: str, label_dir_name: str):
        self._image_dir = os.path.join(self.data_dir, image_dir_name)
        self._label_dir = os.path.join(self.data_dir, label_dir_name)
        assert os.path.isdir(self.data_dir)
        assert os.path.isdir(self._image_dir)
        assert os.path.isdir(self._label_dir)

    def set_stat(self, stat_data):
        r"""
        :param rgb_mean: Tuple[Any,Any,Any]  # rgb_mean = [R_mean, G_mean, B_mean]
        :param rgb_std: Tuple[Any,Any,Any], # rgb_std = [R_std, G_std, B_std]
        """
        self.stat_data=stat_data

    def read_stat(self, stat_file):
        assert os.path.isfile(stat_file)
        def clear_fix(strs):
            return [float(s.strip()) for s in strs]
        with open(stat_file, "r") as f:
            lines = f.readlines()
            stat_data=tuple([clear_fix(line.split(',')) for line in lines])
        return stat_data
    
    def calculate_stat(self):
        r"""
        calculate mean value and standard deviation of rgb channel, and save stat_file
        :return: nothing
        """
        info_file = open(self.stat_file, 'w')
        stat_data=[[self.indices['r_mean'].mean(), self.indices['g_mean'].mean(), self.indices['b_mean'].mean()],
                        [self.indices['r_std'].mean(), self.indices['g_std'].mean(), self.indices['b_std'].mean()]]
        
        info_file.write('%f, %f, %f\n' % (stat_data[0][0],stat_data[0][1],stat_data[0][2]))
        info_file.write('%f, %f, %f\n' % (stat_data[1][0],stat_data[1][1],stat_data[1][2]))
        if 'd_mean' in self.indices:
            stat_data.append([self.indices['d_mean'].mean(), self.indices['d_std'].mean(),
                                    max(self.indices['d_max']), min(self.indices['d_min'])])
            info_file.write('%f, %f, %f ,%f\n' % (stat_data[2][0],stat_data[2][1],
                                                    stat_data[2][2],stat_data[2][3]))
        stat_data=tuple(stat_data)
        return stat_data
    
    # def calculate_stat(self,stat_file):
    #         r"""
    #         calculate mean value and standard deviation of rgb channel, and save stat_file
    #         :return: nothing
    #         """
    #         info_file = open(self.stat_file, 'w')
    #         self.stat_data=[[self.indices['r_mean'].mean(), self.indices['g_mean'].mean(), self.indices['b_mean'].mean()],
    #                         [self.indices['r_std'].mean(), self.indices['g_std'].mean(), self.indices['b_std'].mean()]]
            
    #         info_file.write('%f, %f, %f\n' % (self.stat_data[0][0],self.stat_data[0][1],self.stat_data[0][2]))
    #         info_file.write('%f, %f, %f\n' % (self.stat_data[1][0],self.stat_data[1][1],self.stat_data[1][2]))
    #         if 'd_mean' in self.indices:
    #             self.stat_data.append([self.indices['d_mean'].mean(), self.indices['d_std'].mean(),
    #                                    max(self.indices['d_max']), min(self.indices['d_min'])])
    #             info_file.write('%f, %f, %f ,%f\n' % (self.stat_data[2][0],self.stat_data[2][1],
    #                                                   self.stat_data[2][2],self.stat_data[2][3]))
    #         self.stat_data=tuple(self.stat_data)

    def build_pair(self):
        metas = make_pair_with_indices(self.ids)
        return metas

    def read_xyz_img_pair_file(self, xyz_img_pair_file):
        with open(xyz_img_pair_file, "r") as f:
            scans = f.readlines()
            scans = [line.replace("\t", " ").strip().split(" ") for line in scans]
            scans = np.array(scans)
            scans_dict = dict(zip(scans[:, 1], scans[:, 0]))
        return scans_dict

    def __len__(self):
        return len(self.ids)
