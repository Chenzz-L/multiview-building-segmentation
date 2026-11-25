import sys
from typing import Dict,Tuple,List,Union
import inspect
import copy
import numpy as np
import torch
import albumentations
# from albumentations import (
#     OneOf, Compose, SomeOf,ImageOnlyTransform,
#     HorizontalFlip,VerticalFlip, RandomRotate90,Transpose, Flip,
#     GaussianBlur, MotionBlur, MedianBlur, Blur,Cutout,ToFloat,
#     ShiftScaleRotate,Affine, PiecewiseAffine, OpticalDistortion, GridDistortion, HueSaturationValue,
#     GaussNoise, Emboss, RandomBrightnessContrast, Sharpen,ColorJitter,RandomResizedCrop
# )
from albumentations import (
    OneOf, Compose,ImageOnlyTransform,
    HorizontalFlip,VerticalFlip, RandomRotate90,Transpose, Flip,
    MotionBlur, MedianBlur, Blur,Cutout,ToFloat,
    ShiftScaleRotate, OpticalDistortion, GridDistortion, HueSaturationValue,
    GaussNoise, RandomBrightnessContrast,
)
from utils.utils_tool import *

# import imgaug.augmenters as iaa
# from imgaug.augmentables.segmaps import SegmentationMapsOnImage

__all__=["train_data_augmentation_edge","train_data_augmentation_dsm",
         "train_data_augmentation","train_data_augmentation3d_multi_label",
         "data_augmentation",'data_augumentation_group_views',
         'seg_augmentation','build_agument_pipeline','seg_augmentation_imageonly','build_albumentations_pipeline']

support_aug=dict(inspect.getmembers(albumentations,inspect.isclass))

def build_agument_pipeline(aug_cfgs:List[Dict],image_only=False):
    """
    albumentations针对像素级变换的函数不支持手动转换的float32数据类型，
    如要对Float32数据进行处理，需要使用ToFloat进行转换

    Args:
        aug_cfgs (List[Dict]): _description_

    Returns:
        _type_: _description_
    """
    _aug_cfgs=copy.deepcopy(aug_cfgs)
    _=[]
    for aug_cfg in _aug_cfgs:
        aug_type=aug_cfg.pop("type")
        if image_only:
            if isinstance(support_aug[aug_type](),ImageOnlyTransform):
                _.append(support_aug[aug_type](**aug_cfg))
            else:
                continue
        else:
            _.append(support_aug[aug_type](**aug_cfg))
    return Compose(_)

# def seg_augmentation(aug=None,**kwargs):
#     # aug_squ = SomeOf([
#     #     RandomRotate90(),
#     #     Flip(),
#     #     OneOf([GaussianBlur(),
#     #            GaussNoise()
#     #            ],
#     #           p=.2),
#     #     OneOf([
#     #         MotionBlur(),
#     #         MedianBlur(blur_limit=3),
#     #         Blur(blur_limit=3),
#     #     ]),
#     #     ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
#     #     OneOf([
#     #         Sharpen(),
#     #         Emboss(),
#     #         RandomBrightnessContrast(),
#     #     ], p=0.3),
#     #     ],n=2)
#     # aug_squ = Compose([
#     #     OneOf([
#     #         RandomRotate90(p=0.5),
#     #         HorizontalFlip(p=0.5),
#     #         VerticalFlip(p=0.5),
#     #         Transpose(p=0.5)
#     #     ]),
#     #     GaussNoise(p=0.3),
#     #     OneOf([
#     #         RandomResizedCrop(512,512),
#     #     ]),
#     #     OneOf([
#     #         ShiftScaleRotate(p=0.3),
#     #     ]),
#     #     OneOf([
#     #         GaussianBlur(),
#     #         MotionBlur(),
#     #         MedianBlur(),
#     #         Blur()])
#     # ])

#     if aug:
#         aug_squ=aug
#     else:
#         aug_squ = Compose([
#             RandomRotate90(p=0.5),
#             HorizontalFlip(p=0.5),
#             VerticalFlip(p=0.5),
#             ])

#     #将(C,H,W)转为(H,W,C)

#     for key,val in kwargs.items():
#         kwargs[key]=conver_chw_and_hwc(val,'hwc')
           
#     if 'dsm' in kwargs.keys():
#         data={'image':kwargs['image'],'masks':[kwargs['label'],kwargs['dsm']]}
#     else:
#         data = {'image': kwargs['image'], 'mask': kwargs['label']}

#     auged=aug_squ(**data)
#     # out_data=convert_dicts_2_dict(auged)
    
  
#     for key,val in auged.items():
#         if isinstance(val,(Tuple,List)):
#             for i,v in enumerate(val):
#                 auged[key][i]=conver_chw_and_hwc(v,'chw')
#         else:
#             auged[key]=conver_chw_and_hwc(val,'chw')
    


#     if 'dsm' in kwargs.keys():
#         return auged['image'],auged['masks'][0],auged['masks'][1]
#     else:
#          return auged['image'],auged['mask']
    
    
    
def build_albumentations_pipeline(pipeline_cfg):
    """
    回归旧逻辑：把 DSM (image0) 视为 mask，从而避开 Normalize
    """
    transforms = []
    _cfg_list = copy.deepcopy(pipeline_cfg)
    # ... (中间的解析循环完全不用变，直接复制你之前的逻辑) ...
    for cfg in pipeline_cfg:
        t_type = cfg['type']
        p = cfg.get('p', 1.0)
        
        if t_type == 'Flip':
            transforms.extend([albumentations.HorizontalFlip(p=p), albumentations.VerticalFlip(p=p)])
        elif t_type == 'RandomResizedCrop':
            transforms.append(albumentations.RandomResizedCrop(
                height=cfg['height'], width=cfg['width'],
                scale=tuple(cfg['scale']), ratio=tuple(cfg['ratio']), p=p))
        elif t_type == 'RandomBrightnessContrast':
            transforms.append(albumentations.RandomBrightnessContrast(p=p))
        elif t_type == 'ColorJitter':
            transforms.append(albumentations.ColorJitter(
                brightness=cfg.get('brightness', 0.2),
                contrast=cfg.get('contrast', 0.2),
                saturation=cfg.get('saturation', 0.2),
                hue=cfg.get('hue', 0.2),
                p=p))
        elif t_type == 'Normalize':
            transforms.append(albumentations.Normalize(
                mean=cfg['mean'], std=cfg['std'],
                max_pixel_value=cfg.get('max_pixel_value', 1.0), p=1.0))

    # ====================================================
    # 【核心修改】这里！把 image0 标记为 'mask'
    # ====================================================
    # 这样 Albumentations 就会认为 DSM 也是一种 "标签"，
    # 从而在做 Normalize 时自动跳过它，只处理 RGB。
    return albumentations.Compose(
        transforms, 
        additional_targets={'image0': 'mask', 'masks': 'mask'}
    )



def seg_augmentation(image,mask:Union[np.ndarray,List[np.ndarray]],aug):
    aug_squ=aug
    #将(C,H,W)转为(H,W,C)
    mask_or_masks = 'masks' if isinstance(mask,List) else 'mask'
    data={'image':image,
          mask_or_masks:mask}
    
    for key,val in data.items():    
        if key=='masks':
            for i,v in enumerate(val):
                data[key][i]=conver_chw_and_hwc(v,'hwc')
        else:
            data[key]=conver_chw_and_hwc(val,'hwc')

    auged=aug_squ(**data)

    for key,val in auged.items():
        if isinstance(val,(Tuple,List)):
            for i,v in enumerate(val):
                auged[key][i]=conver_chw_and_hwc(v,'chw')
        else:
            auged[key]=conver_chw_and_hwc(val,'chw')
    

    if mask_or_masks=='masks':
        return [auged['image']]+auged['masks']
    else:
        return auged['image'],auged['mask']

def seg_augmentation_imageonly(image,aug):
    
    data={'image':image}
    
    for key,val in data.items():    
        data[key]=conver_chw_and_hwc(val,'hwc')

    auged=aug(**data)

    for key,val in auged.items():
        auged[key]=conver_chw_and_hwc(val,'chw')
    return auged['image']

def train_data_augmentation(image_data, label_data):
    # check augmentation result
    # plt.imshow(np.transpose(np.squeeze(image_data),(1,2,0)).astype('uint8'))
    # plt.show()
    # plt.imshow(np.squeeze(label_data)*255)
    # plt.show()

    # image_data = image_data.copy()
    # label_data = label_data.copy()
    aug_seed_1 = np.random.randint(0, 2)  # [0,1]
    aug_seed_2 = np.random.randint(0, 2)  # [0,1]
    aug_seed_3 = np.random.randint(0, 5)  # [0,4]
    if aug_seed_1 == 1:  # rotation
        rotation_seed = np.random.randint(1, 4)  # [1,3]
        image_data = np.rot90(image_data, rotation_seed, (1, 2))
        label_data = np.rot90(label_data, rotation_seed, (1, 2))
    if aug_seed_2 == 1:  # flip
        flip_seed = np.random.randint(0, 2)  # [0,1] 1:水平翻转 0:垂直翻转
        for i in range(image_data.shape[0]):
            image_data[i] = np.flip(image_data[i], axis=flip_seed)
        for i in range(label_data.shape[0]):
            label_data[i] = np.flip(label_data[i], axis=flip_seed)
    if aug_seed_3 == 1:  # blur
        # image_data = cv2.GaussianBlur(image_data, (3,3), 0)
        pass

    # check augmentation result
    # plt.imshow(np.transpose(np.squeeze(image_data),(1,2,0)).astype('uint8'))
    # plt.show()
    # plt.imshow(np.squeeze(label_data)*255)
    # plt.show()
    return image_data.copy(), label_data.copy()

def data_augmentation(*args,**kwargs):
    aug_seed_1 = np.random.randint(0, 2)  # [0,1]
    aug_seed_2 = np.random.randint(0, 2)  # [0,1]
    out_data=[]

    if aug_seed_1==1:
        rotation_seed = np.random.randint(1, 4)  # [1,3]
        for data in args:
            data = np.rot90(data, rotation_seed, (1, 2))
            out_data.append(data)
    else:
        for data in args:
            out_data.append(data)

    if aug_seed_2 == 1:  # flip
        flip_seed = np.random.randint(0, 2)  # [0,1] 1:水平翻转 0:垂直翻转
        for data in out_data:
            for i in range(data.shape[0]):
                data[i] = np.flip(data[i], axis=flip_seed)

    # for key,value in kwargs.items():
    #     if aug_seed_1 == 1:  # rotation
    #         rotation_seed = np.random.randint(1, 4)  # [1,3]
    #         value=np.rot90(value, rotation_seed, (1, 2))
    #     if aug_seed_2 == 1:  # flip
    #         flip_seed = np.random.randint(0, 2)  # [0,1] 1:水平翻转 0:垂直翻转
    #         for i in range(value.shape[0]):
    #             value[i] = np.flip(value[i], axis=flip_seed)
    #     out_data.append(value)

    return out_data

def data_augumentation_group_views(xyz_data,grids,*args):
    aug_seed_1 = np.random.randint(0, 2)  # [0,1]
    aug_seed_2 = np.random.randint(0, 2)  # [0,1]
    out_data = []

    if aug_seed_1 == 1:  # rotation
        rotation_seed = np.random.randint(1, 4)  # [1,3]

        xyz_data=np.rot90(xyz_data, rotation_seed, (1, 2))
        for datas in args:
            tdata=[]
            for data in datas:
                data = np.rot90(data, rotation_seed, (1, 2))
                tdata.append(data)
            out_data.append(tdata)


        for i,grid in enumerate(grids):
            grid=np.rot90(grid, rotation_seed, (1, 2))
            if rotation_seed == 2:
                #[0.5,0]----->[-0.5,0]
                grid[0,:,:]=-grid[0,:,:]
            elif rotation_seed == 1:
                #[0.5,0]----->[0,0.5]
                grid = grid[::-1,:, :]
            elif rotation_seed == 3:
                #[0.5,0]----->[0,-0.5]
                grid = grid[::-1,:, :]
                grid[1,:,:]=-grid[1,:,:]
            grids[i]=grid
    else:
        for data in args:
            out_data.append(data)

    if aug_seed_2 == 1:  # flip
        flip_seed = np.random.randint(0, 2)  # [0,1] 1:水平翻转 0:垂直翻转
        xyz_data[0]=np.flip(xyz_data[0], axis=flip_seed)
        for datas in out_data:
            for data in datas:
                for i in range(data.shape[0]):
                    data[i] = np.flip(data[i], axis=flip_seed)


        for i,grid in enumerate(grids):
            for j in range(grid.shape[0]):
                grid[j]=np.flip(grid[j], axis=flip_seed)

            # grid的数值坐标系与上述操作的坐标系不一样，需要额外对数值进行处理
            if flip_seed:
                grid[0,:,:]=-grid[0,:,:]
            else:
                grid[1,:, :] = -grid[1,:, :]
            grids[i]=grid
    return xyz_data,grids,out_data

def train_data_augmentation_edge(image_data, label_data, edge_data):
    aug_seed_1 = np.random.randint(0, 2)  # [0,1]
    aug_seed_2 = np.random.randint(0, 2)  # [0,1]
    aug_seed_3 = np.random.randint(0, 5)  # [0,4]
    if aug_seed_1 == 1:  # rotation
        rotation_seed = np.random.randint(1, 4)  # [1,3]
        image_data = np.rot90(image_data, rotation_seed, (1, 2))
        label_data = np.rot90(label_data, rotation_seed, (1, 2))
        edge_data = np.rot90(edge_data, rotation_seed, (1, 2))
    if aug_seed_2 == 1:  # flip
        flip_seed = np.random.randint(0, 2)  # [0,1] 1:水平翻转 0:垂直翻转
        for i in range(image_data.shape[0]):
            image_data[i] = np.flip(image_data[i], axis=flip_seed)
        for i in range(label_data.shape[0]):
            label_data[i] = np.flip(label_data[i], axis=flip_seed)
        for i in range(edge_data.shape[0]):
            edge_data[i] = np.flip(edge_data[i], axis=flip_seed)
    if aug_seed_3 == 1:  # blur
        # image_data = cv2.GaussianBlur(image_data, (3,3), 0)
        pass
    return image_data.copy(), label_data.copy(), edge_data.copy()

def train_data_augmentation_dsm(image_data, label_data, DSM_data):
    # check augmentation result
    # plt.imshow(np.transpose(np.squeeze(image_data),(1,2,0)).astype('uint8'))
    # plt.show()
    # plt.imshow(np.squeeze(DSM_data)*255)
    # plt.show()
    # plt.imshow(np.squeeze(label_data)*255)
    # plt.show()
    aug_seed_1 = np.random.randint(0, 2)  # [0,1]
    aug_seed_2 = np.random.randint(0, 2)  # [0,1]
    aug_seed_3 = np.random.randint(0, 5)  # [0,4]
    if aug_seed_1 == 1:  # rotation
        rotation_seed = np.random.randint(1, 4)  # [1,3]
        image_data = np.rot90(image_data, rotation_seed, (1, 2))
        label_data = np.rot90(label_data, rotation_seed, (1, 2))
        DSM_data = np.rot90(DSM_data, rotation_seed, (1, 2))
    if aug_seed_2 == 1:  # flip
        flip_seed = np.random.randint(0, 2)  # [0,1] 1:水平翻转 0:垂直翻转
        for i in range(image_data.shape[0]):
            image_data[i] = np.flip(image_data[i], axis=flip_seed)
        for i in range(label_data.shape[0]):
            label_data[i] = np.flip(label_data[i], axis=flip_seed)
        for i in range(DSM_data.shape[0]):
            DSM_data[i] = np.flip(DSM_data[i], axis=flip_seed)
    if aug_seed_3 == 1:  # blur
        # image_data = cv2.GaussianBlur(image_data, (3,3), 0)
        pass
    # check augmentation result
    # plt.imshow(np.transpose(np.squeeze(image_data),(1,2,0)).astype('uint8'))
    # plt.show()
    # plt.imshow(np.squeeze(DSM_data)*255)
    # plt.show()
    # plt.imshow(np.squeeze(label_data)*255)
    # plt.show()
    return image_data.copy(), label_data.copy(), DSM_data.copy()

def train_data_augmentation3d_multi_label(image_data, label_data_cbo, label_data_dx, label_data_dy,
                                          label_data_foot, label_data_roof, label_data_rf):
    if image_data.ndim != 3 or label_data_cbo.ndim != 3 or label_data_dx.ndim != 3 or label_data_dy.ndim != 3 or \
            label_data_foot.ndim != 3 or label_data_roof.ndim != 3 or label_data_rf.ndim != 3:
        raise Exception("data.ndim error")
    # set seeds
    aug_seed_1 = np.random.randint(0, 2)  # [0,1]
    aug_seed_2 = np.random.randint(0, 2)  # [0,1]
    aug_seed_3 = np.random.randint(0, 5)  # [0,4]
    if aug_seed_1 == 1:  # rotation, anticlockwise rotate 90, 180, 270
        rotation_seed = np.random.randint(1, 4)  # [1,3]
        image_data = np.rot90(image_data, rotation_seed, (1, 2))
        label_data_cbo = np.rot90(label_data_cbo, rotation_seed, (1, 2))
        label_data_dx = np.rot90(label_data_dx, rotation_seed, (1, 2))
        label_data_dy = np.rot90(label_data_dy, rotation_seed, (1, 2))
        label_data_foot = np.rot90(label_data_foot, rotation_seed, (1, 2))
        label_data_roof = np.rot90(label_data_roof, rotation_seed, (1, 2))
        label_data_rf = np.rot90(label_data_rf, rotation_seed, (1, 2))
        # process dx, dy
        if rotation_seed == 1:  # anticlockwise rotate 90, dy=-dy , then swap dx and dy,
            label_data_dx, label_data_dy = swap_dx_dy(label_data_dx, -label_data_dy)
        elif rotation_seed == 2:  # anticlockwise rotate 180, dx=-dx , dy=-dy
            label_data_dx = -label_data_dx
            label_data_dy = -label_data_dy
        elif rotation_seed == 3:  # anticlockwise rotate 270, dx=-dx , then swap dx and dy,
            label_data_dx, label_data_dy = swap_dx_dy(-label_data_dx, label_data_dy)
        else:
            raise Exception("rotation_seed error")
    if aug_seed_2 == 1:  # flip
        flip_seed = np.random.randint(0, 2)  # [0,1] 1:水平翻转 0:垂直翻转
        for i in range(image_data.shape[0]):
            image_data[i] = np.flip(image_data[i], axis=flip_seed)
        for i in range(label_data_cbo.shape[0]):
            label_data_cbo[i] = np.flip(label_data_cbo[i], axis=flip_seed)
        for i in range(label_data_dx.shape[0]):
            label_data_dx[i] = np.flip(label_data_dx[i], axis=flip_seed)
        for i in range(label_data_dy.shape[0]):
            label_data_dy[i] = np.flip(label_data_dy[i], axis=flip_seed)
        for i in range(label_data_foot.shape[0]):
            label_data_foot[i] = np.flip(label_data_foot[i], axis=flip_seed)
        for i in range(label_data_roof.shape[0]):
            label_data_roof[i] = np.flip(label_data_roof[i], axis=flip_seed)
        for i in range(label_data_rf.shape[0]):
            label_data_rf[i] = np.flip(label_data_rf[i], axis=flip_seed)
        # process dx, dy
        if flip_seed == 0:  # 上面的翻到下面
            label_data_dy = -label_data_dy
        elif flip_seed == 1:  # 左边的翻到右边
            label_data_dx = -label_data_dx
        else:
            raise Exception("flip_seed error")
    if aug_seed_3 == 1:  # blur
        # image_data = cv2.GaussianBlur(image_data, (3,3), 0)
        pass

    return image_data.copy(), label_data_cbo.copy(), label_data_dx.copy(), label_data_dy.copy(), \
           label_data_foot.copy(), label_data_roof.copy(), label_data_rf.copy()

def calculate_moved_map_2d(dx_data: np.ndarray, dy_data: np.ndarray, ix: float = 1.0, iy: float = 1.0):  # 1.486秒
    r"""
    输出一个移动后的标签图，逐个disp图的元素的值，加上该值对应的索引，即: [x',y'] = [x+dx,y-dy] = 1
    :param dx_data: 2d array, 记录该点的x方向的偏移量，向右为正，向左为负
    :param dy_data: 2d array, 记录该点的y方向的偏移量，向上为正，向下为负
    :param ix: 1/空间分辨率x，由于偏移标签的值为地理坐标的偏移，在计算像素坐标的过程中，需要乘以地理坐标到像素坐标的转换系数ix
    :param iy: 1/空间分辨率y
    :return: 移动后的标签图: 2d array
    """
    if dx_data.ndim != 2 or dy_data.ndim != 2:
        raise Exception("data.ndim != 2")
    roof_moved = np.zeros_like(dx_data, dtype=np.uint8)
    x_list = np.where(np.logical_or(dx_data < -0.5 / ix, dx_data > 0.5 / ix))
    y_list = np.where(np.logical_or(dy_data < -0.5 / iy, dy_data > 0.5 / iy))
    xy_list_2 = (np.concatenate((x_list[0], y_list[0]), 0), np.concatenate((x_list[1], y_list[1]), 0))
    # 从影像的左下角开始读，GDAL读取影像的坐标系为左上角坐标系，左上角为(0,0),右下角为(h,w)
    # 偏移图的方向为左下角坐标系，向上为+，向右为+
    # 在计算偏移时，需要将y方向翻转
    for ij in set(zip(xy_list_2[0], xy_list_2[1])):
        i = ij[0]
        j = ij[1]
        x = int(j + dx_data[i, j] * ix)
        y = int(i - dy_data[i, j] * iy)
        # 移动范围在0.5个像素以内的不会移动
        if 0 <= x < dx_data.shape[1] and 0 <= y < dx_data.shape[0]:
            roof_moved[y, x] = 1
    return roof_moved

def calculate_moved_map_2d_slow(dx_data: np.ndarray, dy_data: np.ndarray, ix: float = 1.0, iy: float = 1.0):  # 1.507秒
    r"""
    输出一个移动后的标签图，逐个disp图的元素的值，加上该值对应的索引，即: [x',y'] = [x+dx,y-dy] = 1
    :param dx_data: 2d array, 记录该点的x方向的偏移量，向右为正，向左为负
    :param dy_data: 2d array, 记录该点的y方向的偏移量，向上为正，向下为负
    :param ix: 1/空间分辨率x，由于偏移标签的值为地理坐标的偏移，在计算像素坐标的过程中，需要乘以地理坐标到像素坐标的转换系数ix
    :param iy: 1/空间分辨率y
    :return: 移动后的标签图: 2d array
    """
    if dx_data.ndim != 2 or dy_data.ndim != 2:
        raise Exception("data.ndim != 2")
    roof_moved = np.zeros_like(dx_data, dtype=np.uint8)
    x_list = np.where(dx_data != 0)
    y_list = np.where(dy_data != 0)
    xy_list_2 = (np.concatenate((x_list[0], y_list[0]), 0), np.concatenate((x_list[1], y_list[1]), 0))
    b_list = [[], []]
    # 从影像的左下角开始读，GDAL读取影像的坐标系为左上角坐标系，左上角为(0,0),右下角为(h,w)
    # 偏移图的方向为左下角坐标系，向上为+，向右为+
    # 在计算偏移时，需要将y方向翻转
    for ij in set(zip(xy_list_2[0], xy_list_2[1])):
        i = ij[0]
        j = ij[1]
        x = int(j + dx_data[i, j] * ix)
        y = int(i - dy_data[i, j] * iy)
        # 移动范围在0.5个像素以内的不会移动
        if 0.5 <= x < dx_data.shape[1] and 0.5 <= y < dx_data.shape[0]:
            b_list[0].append(y)
            b_list[1].append(x)
    roof_moved[tuple(b_list)] = 1
    return roof_moved

def calculate_moved_map_2d_slowest(dx_data: np.ndarray, dy_data: np.ndarray, ix: float = 1.0,
                                   iy: float = 1.0):  # 2.379秒
    r"""
    输出一个移动后的标签图，逐个disp图的元素的值，加上该值对应的索引，即: [x',y'] = [x+dx,y-dy] = 1
    :param dx_data: 2d array, 记录该点的x方向的偏移量，向右为正，向左为负
    :param dy_data: 2d array, 记录该点的y方向的偏移量，向上为正，向下为负
    :param ix: 1/空间分辨率x，由于偏移标签的值为地理坐标的偏移，在计算像素坐标的过程中，需要乘以地理坐标到像素坐标的转换系数ix
    :param iy: 1/空间分辨率y
    :return: 移动后的标签图: 2d array
    """
    if dx_data.ndim != 2 or dy_data.ndim != 2:
        raise Exception("data.ndim != 2")
    roof_moved = np.zeros_like(dx_data, dtype=np.uint8)
    # 从影像的左下角开始读，GDAL读取影像的坐标系为左上角坐标系，左上角为(0,0),右下角为(h,w)
    # 偏移图的方向为左下角坐标系，向上为+，向右为+
    # 在计算偏移时，需要将y方向翻转
    for i in range(dx_data.shape[0]):  # row, 第y行
        for j in range(dx_data.shape[1]):  # column， 第x列
            if dx_data[i, j] != 0 or dy_data[i, j] != 0:
                x = int(j + dx_data[i, j] * ix)
                y = int(i - dy_data[i, j] * iy)
                # 移动范围在0.5个像素以内的不会移动
                if 0.5 <= x < dx_data.shape[1] and 0.5 <= y < dx_data.shape[0]:
                    roof_moved[y, x] = 1
    return roof_moved

def calculate_moved_map_2d_tensor(dx_data: torch.Tensor, dy_data: torch.Tensor, roof_moved: torch.Tensor,
                                  ix: float = 1.0, iy: float = 1.0):
    r"""
    直接修改roof_moved
    :return: nothing
    """
    for i in range(dx_data.shape[0]):  # row, 第y行
        for j in range(dx_data.shape[1]):  # column， 第x列
            if dx_data[i, j] != 0 or dy_data[i, j] != 0:
                x = int(j + dx_data[i, j].item() * ix)
                y = int(i - dy_data[i, j].item() * iy)
                # 移动范围在0.5个像素以内的不会移动
                if 0.5 <= x < dx_data.shape[1] and 0.5 <= y < dx_data.shape[0]:
                    roof_moved[y, x] = 1.0

def swap_dx_dy(dx_data: np.ndarray, dy_data: np.ndarray):
    return dy_data.copy(), dx_data.copy()


if __name__ == "__main__":
    pass
    # out_dir = '/path/to/project/1Code\\zyy-segmentation-roof\\dataset_augmentation_test_result\\'
    # dx_file = "/path/to/project/1Data\\shanghai_mini\\test\\samples\\label_dx\\test_0256_5120.tif"
    # dy_file = "/path/to/project/1Data\\shanghai_mini\\test\\samples\\label_dy\\test_0256_5120.tif"
    # foot_file = "/path/to/project/1Code/zyy-segmentation-RE/datasets/augmentation_result/footprint.tif"
    #
    # io = raster_io.RasterIO()
    # io.open(foot_file)
    # label_data_ = io.read_as_array().astype('float32')
    # mask_to_binary_edges_2d(label_data_, radius=1)

    # io=raster_io.RasterIO()
    # io.open(r"/path/to/project/Xuzhou-Aerial\0_Xuzhou_DeepL\Xuzhou-IMG_MV\test\samples_none\label\x029517_test_05_0000_0000.tif")
    # label_data=io.read_as_array().astype(np.float32)
    # io.open(r"/path/to/project/Xuzhou-Aerial\0_Xuzhou_DeepL\Xuzhou-IMG_MV\test\samples_none\image\x029517_test_05_0000_0000.tif")
    # image_data=io.read_as_array().astype(np.float32)
    # image_data=np.transpose(image_data,[1,2,0])
    # io.open(r"/path/to/project/Xuzhou-Aerial\0_Xuzhou_DeepL\Xuzhou-IMG_MV\test\samples_none\dsm\x029517_test_05_0000_0000.tif")
    # dsm_data=io.read_as_array().astype(np.float32)

    # data={"image": image_data, "masks": [label_data,dsm_data]}
    # augmented=seg_augmentation(image=image_data,dsm=dsm_data,label=label_data)
    # augmented['image']=np.transpose(augmented['image'],[2,0,1])
    # raster_io.fastly_save_image(augmented['image'],r"/path/to/project/Users\35416\Desktop\image.tif",gdal_type=gdal.GDT_Float32)
    # raster_io.fastly_save_image(augmented['masks'][0],r"/path/to/project/Users\35416\Desktop\label.tif",gdal_type=gdal.GDT_Float32)
    # raster_io.fastly_save_image(augmented['masks'][1],r"/path/to/project/Users\35416\Desktop\dsm.tif",gdal_type=gdal.GDT_Float32)
    # print("ok")
    # io = raster_io.RasterIO()
    # io.open(dx_file)
    # dx_data_ = io.read_as_array().astype('float32')
    # io.open(dy_file)
    # dy_data_ = io.read_as_array().astype('float32')
    #
    # dx_data_ = np.ones_like(dx_data_)
    # dy_data_ = np.ones_like(dy_data_)
    # t = time.time()
    # k = np.where(np.logical_or(dx_data_ < -0.5 / 10, dx_data_ > 0.5 / 10))
    # print(time.time() - t)
    # t = time.time()
    # m = np.where(dx_data_ != 0)
    # print(time.time() - t)
    # t = time.time()
    # roof = calculate_moved_map_2d(dx_data_, dy_data_, 10, 10)
    # print(time.time() - t)
    # # plt.imshow(roof)
    # # plt.show()
    # t = time.time()
    # roof = calculate_moved_map_2d_slow(dx_data_, dy_data_, 10, 10)
    # print(time.time() - t)
    # # plt.imshow(roof)
    # # plt.show()
    # t = time.time()
    # roof = calculate_moved_map_2d(dx_data_, dy_data_, 10, 10)
    # print(time.time() - t)
    # # plt.imshow(roof)
    # # plt.show()

    # '''移动'''
    # t = time.time()
    # roof = calculate_moved_map_2d(dx_data_, dy_data_, 10, 10)
    # print(time.time() - t)
    # eval_io = raster_io.RasterIO()
    # eval_io.create(out_dir + "moved.tif", 768, 768, 1, options=["COMPRESS=DEFLATE"])
    # eval_io.write_band(roof, 0)
    # eval_io.copy_projection(dx_file)
    #
    # '''翻转'''
    # '''水平'''
    # dx_data_2 = np.flip(dx_data_, axis=0)
    # dy_data_2 = np.flip(dy_data_, axis=0)
    # roof = calculate_moved_map_2d(dx_data_2, -dy_data_2, 10, 10)
    # eval_io = raster_io.RasterIO()
    # eval_io.create(out_dir + "moved_flip_h.tif", 768, 768, 1, options=["COMPRESS=DEFLATE"])
    # eval_io.write_band(roof, 0)
    # eval_io.copy_projection(dx_file)
    # '''垂直'''
    # dx_data_2 = np.flip(dx_data_, axis=1)
    # dy_data_2 = np.flip(dy_data_, axis=1)
    # roof = calculate_moved_map_2d(-dx_data_2, dy_data_2, 10, 10)
    # eval_io = raster_io.RasterIO()
    # eval_io.create(out_dir + "moved_flip_v.tif", 768, 768, 1, options=["COMPRESS=DEFLATE"])
    # eval_io.write_band(roof, 0)
    # eval_io.copy_projection(dx_file)
    # '''保存footprint翻转的结果'''
    # '''水平'''
    # io = raster_io.RasterIO()
    # io.open(foot_file)
    # f_data = io.read_as_array().astype('float32')
    # f_data = np.flip(f_data, axis=0)
    # eval_io = raster_io.RasterIO()
    # eval_io.create(out_dir + "footprint_flip_h.tif", 768, 768, 1, options=["COMPRESS=DEFLATE"])
    # eval_io.write_band(f_data, 0)
    # eval_io.copy_projection(dx_file)
    # '''垂直'''
    # io = raster_io.RasterIO()
    # io.open(foot_file)
    # f_data = io.read_as_array().astype('float32')
    # f_data = np.flip(f_data, axis=1)
    # eval_io = raster_io.RasterIO()
    # eval_io.create(out_dir + "footprint_flip_v.tif", 768, 768, 1, options=["COMPRESS=DEFLATE"])
    # eval_io.write_band(f_data, 0)
    # eval_io.copy_projection(dx_file)
    #
    # '''旋转'''
    # '''90'''
    # dx_data_2 = np.rot90(dx_data_, 1, (0, 1))
    # dy_data_2 = np.rot90(dy_data_, 1, (0, 1))
    # dx_data_2, dy_data_2 = swap_dx_dy(dx_data_2, -dy_data_2)
    # roof = calculate_moved_map_2d(dx_data_2, dy_data_2, 10, 10)
    # eval_io = raster_io.RasterIO()
    # eval_io.create(out_dir + "moved_rotate_90.tif", 768, 768, 1, options=["COMPRESS=DEFLATE"])
    # eval_io.write_band(roof, 0)
    # eval_io.copy_projection(dx_file)
    # '''180'''
    # dx_data_2 = -np.rot90(dx_data_, 2, (0, 1))
    # dy_data_2 = -np.rot90(dy_data_, 2, (0, 1))
    # roof = calculate_moved_map_2d(dx_data_2, dy_data_2, 10, 10)
    # eval_io = raster_io.RasterIO()
    # eval_io.create(out_dir + "moved_rotate_180.tif", 768, 768, 1, options=["COMPRESS=DEFLATE"])
    # eval_io.write_band(roof, 0)
    # eval_io.copy_projection(dx_file)
    # '''270'''
    # dx_data_2 = np.rot90(dx_data_, 3, (0, 1))
    # dy_data_2 = np.rot90(dy_data_, 3, (0, 1))
    # dx_data_2, dy_data_2 = swap_dx_dy(-dx_data_2, dy_data_2)
    # roof = calculate_moved_map_2d(dx_data_2, dy_data_2, 10, 10)
    # eval_io = raster_io.RasterIO()
    # eval_io.create(out_dir + "moved_rotate_270.tif", 768, 768, 1, options=["COMPRESS=DEFLATE"])
    # eval_io.write_band(roof, 0)
    # eval_io.copy_projection(dx_file)
    # '''保存footprint旋转的结果'''
    # '''90'''
    # io = raster_io.RasterIO()
    # io.open(foot_file)
    # f_data = io.read_as_array().astype('float32')
    # f_data = np.rot90(f_data, 1, (0, 1))
    # eval_io = raster_io.RasterIO()
    # eval_io.create(out_dir + "footprint_rotate_90.tif", 768, 768, 1, options=["COMPRESS=DEFLATE"])
    # eval_io.write_band(f_data, 0)
    # eval_io.copy_projection(dx_file)
    # '''180'''
    # f_data = np.rot90(f_data, 1, (0, 1))
    # eval_io = raster_io.RasterIO()
    # eval_io.create(out_dir + "footprint_rotate_180.tif", 768, 768, 1, options=["COMPRESS=DEFLATE"])
    # eval_io.write_band(f_data, 0)
    # eval_io.copy_projection(dx_file)
    # '''270'''
    # f_data = np.rot90(f_data, 1, (0, 1))
    # eval_io = raster_io.RasterIO()
    # eval_io.create(out_dir + "footprint_rotate_270.tif", 768, 768, 1, options=["COMPRESS=DEFLATE"])
    # eval_io.write_band(f_data, 0)
    # eval_io.copy_projection(dx_file)
