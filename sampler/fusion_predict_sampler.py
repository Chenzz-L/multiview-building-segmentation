from typing import List,Union

import numpy as np
import torch
import torch.nn.functional as F
import cv2 as cv


from utils.crop_images import get_crop_coordinate_from_xyz_corner_colinear

# def fusion_sampler(batch, crop_width, crop_height, crop_stride):

#     def cat_sample_with_list(datas:List[Union[np.ndarray, torch.Tensor]]):
#         data=datas[0]
#         if data.ndim<4:
#             data=np.expand_dims(data,axis=list(range(4-data.ndim)))
#         for tdata in datas[1:]:
#             if tdata.ndim < 4:
#                 tdata = np.expand_dims(tdata, axis=list(range(4 - tdata.ndim)))
#             data=np.concatenate([data,tdata],axis=0)
#         return data
#     xyz_shape = batch['xyz_image'].shape

#     xoffs = list(range(0, xyz_shape[-1] - crop_width, crop_stride))
#     xoffs.append(xyz_shape[-1] - crop_width)
#     xoffs = sorted(list(set(xoffs)))
#     yoffs = list(range(0, xyz_shape[-2] - crop_height, crop_stride))
#     yoffs.append(xyz_shape[-2] - crop_height)
#     yoffs = sorted(list(set(yoffs)))

#     corner_coordination = batch['corners']

#     input_data_dict = {
#         'image': batch['image'],
#         'grid': batch['grid'],
#         'xyz_image': batch['xyz_image']}

#     grid_scale = batch['grid_scale']

#     n = 0
#     for xoff in xoffs:
#         for yoff in yoffs:
#             output_data_dict = {}

#             for i,files_data_zip in enumerate(zip(input_data_dict['image'],input_data_dict['grid'])):
#                 image_data, grid_data = files_data_zip
#                 image_data, grid_data = image_data.clone(), grid_data.clone()
#                 image_shape = image_data.shape

#                 if grid_data.shape[-1]==2:
#                     grid_data=np.transpose(grid_data,[2,0,1])
#                 grid_data[0,:, :] = (image_shape[-1] - 1) * (grid_data[0,:, :] + 1) / 2
#                 grid_data[1,:, :] = (image_shape[-2] - 1) * (grid_data[1,:, :] + 1) / 2

#                 image_sample_width, image_sample_height, \
#                 min_x, min_y = get_crop_coordinate_from_xyz_corner_colinear(coordinate_corners = corner_coordination[i][n],
#                                                                             image_shape=image_shape)

#                 # crop grid and transformer to [image_sample_width,image_sample_height] coordination
#                 crop_grid_data = grid_data[:, yoff:yoff + crop_height, xoff:xoff + crop_width]
#                 crop_grid_data[0, :, :] = 2 * (crop_grid_data[0, :, :] - min_x) / (image_sample_width - 1) - 1
#                 crop_grid_data[1, :, :] = 2 * (crop_grid_data[1, :, :] - min_y) / (image_sample_height - 1) - 1

#                 crop_grid_data = np.transpose(crop_grid_data,[1,2,0])
#                 crop_grid_data = cv.resize(crop_grid_data,
#                                          dsize=(int(crop_height * grid_scale),int(crop_width * grid_scale)))


#                 crop_image_data = image_data[:, min_y:min_y + image_sample_height, min_x:min_x + image_sample_width]
#                 crop_image_data = np.transpose(crop_image_data,[1,2,0])

#                 crop_image_data = cv.resize(crop_image_data,dsize = [crop_height,crop_width], interpolation = cv.INTER_LINEAR)
#                 crop_image_data = np.transpose(crop_image_data,[2,0,1])


#                 if 'image' in output_data_dict.keys():
#                     output_data_dict['image'].append(crop_image_data)
#                 else:
#                     output_data_dict['image'] = [crop_image_data]

#                 if 'grid' in output_data_dict.keys():
#                     output_data_dict['grid'].append(crop_grid_data)
#                 else:
#                     output_data_dict['grid'] = [crop_grid_data]

#             for key,value in output_data_dict.items():
#                 output_data_dict[key] = cat_sample_with_list(value)


#             output_data_dict['image'] = [output_data_dict['image'],batch['xyz_image'][:, yoff:yoff + crop_height, xoff:xoff + crop_width]]

#             n += 1
#             yield output_data_dict, xoff, yoff

def fusion_sampler(batch, crop_width, crop_height, crop_stride):

    xyz_shape = batch['xyz_image'].shape

    xoffs = list(range(0, xyz_shape[-1] - crop_width, crop_stride))
    xoffs.append(xyz_shape[-1] - crop_width)
    xoffs = sorted(list(set(xoffs)))
    yoffs = list(range(0, xyz_shape[-2] - crop_height, crop_stride))
    yoffs.append(xyz_shape[-2] - crop_height)
    yoffs = sorted(list(set(yoffs)))

    corner_coordination = batch['corners']

    input_data_dict = {
        'image': batch['image'],
        'grid': batch['grid'],
        'xyz_image': batch['xyz_image']}

    grid_scale = batch['grid_scale']

    n = 0
    for xoff in xoffs:
        for yoff in yoffs:
            output_data_dict = {}

            for i,files_data_zip in enumerate(zip(input_data_dict['image'],input_data_dict['grid'])):
                image_data, grid_data = files_data_zip
                image_data, grid_data = image_data.clone(), grid_data.clone()
                image_shape = image_data.shape

                if grid_data.shape[-1] == 2:
                    grid_data = torch.permute(grid_data,[0, 3, 1, 2])
                grid_data[:, 0, :, :] = (image_shape[-1] - 1) * (grid_data[:, 0, :, :] + 1) / 2
                grid_data[:, 1, :, :] = (image_shape[-2] - 1) * (grid_data[:, 1, :, :] + 1) / 2

                image_sample_width, image_sample_height, \
                min_x, min_y = get_crop_coordinate_from_xyz_corner_colinear(coordinate_corners = corner_coordination[i][n],
                                                                            image_shape = image_shape)

                # crop grid and transformer to [image_sample_width,image_sample_height] coordination
                crop_grid_data = grid_data[:, :, yoff:yoff + crop_height, xoff:xoff + crop_width]
                crop_grid_data[:, 0, :, :] = 2 * (crop_grid_data[:, 0, :, :] - min_x) / (image_sample_width - 1) - 1
                crop_grid_data[:, 1, :, :] = 2 * (crop_grid_data[:, 1, :, :] - min_y) / (image_sample_height - 1) - 1

                crop_grid_data = F.interpolate(crop_grid_data, size = [int(crop_height*grid_scale), int(crop_width*grid_scale)])
                crop_grid_data = torch.permute(crop_grid_data,[0, 2, 3, 1])

                crop_image_data = image_data[..., min_y:min_y + image_sample_height, min_x:min_x + image_sample_width]
                crop_image_data = F.interpolate(crop_image_data, size = [crop_height, crop_width], mode='bilinear')


                if 'image' in output_data_dict.keys():
                    output_data_dict['image'].append(crop_image_data)
                else:
                    output_data_dict['image'] = [crop_image_data]

                if 'grid' in output_data_dict.keys():
                    output_data_dict['grid'].append(crop_grid_data)
                else:
                    output_data_dict['grid'] = [crop_grid_data]

            for key, value in output_data_dict.items():
                output_data_dict[key] = torch.cat(value, dim=0)


            output_data_dict['image'] = [output_data_dict['image'],batch['xyz_image'][..., yoff:yoff + crop_height, xoff:xoff + crop_width]]
            output_data_dict["batch_info"] = batch["batch_info"]
            n += 1
            yield output_data_dict, xoff, yoff

def fusion_stitch(pred_output, offsets, batch, class_num):
    image_shape=batch['image'].shape
    # pred_sum = np.zeros([class_num, image_shape[-2], image_shape[-1]], dtype=np.float32)
    # count = np.zeros([image_shape[-2], image_shape[-1]], dtype=np.uint8)
    pred_sum = torch.zeros([class_num, image_shape[-2], image_shape[-1]], dtype=torch.float32)
    count = torch.zeros([image_shape[-2], image_shape[-1]], dtype=torch.uint8)
    crop_height=pred_output[0].shape[-2]
    crop_width=pred_output[0].shape[-1]
    for out_put, offset in zip(pred_output, offsets):
        pred_sum[:, offset[1]:offset[1] + crop_height, offset[0]:offset[0] + crop_width]+= out_put.cpu().squeeze_(dim=0)
        count[offset[1]:offset[1] + crop_height, offset[0]:offset[0] + crop_width]+= 1
    count[count==0]=1
    pred_sum /= count
    return pred_sum.numpy()
