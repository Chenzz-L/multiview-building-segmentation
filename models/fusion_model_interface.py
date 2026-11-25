import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np

from sampler.normal_predict_sampler import normal_sampler,normal_stitch,streaming_stitch_step
from sampler.fusion_predict_sampler import fusion_sampler
from .model_interface import MInterface
from utils.raster_io import RasterIO
from utils.basic_func import get_param_from_config_func
from AffineTransform.affine_transform import *
from metric import generate_metric, cal_eval_image


class FusionMInterface(MInterface):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
    
    def on_predict_start(self):
        device = next(self.parameters()).device
        self.img_predict_metrics.to(device)

    def configure_metric(self):
        metrics_cfg = getattr(self.hparams, 'metric', None)
        if metrics_cfg is None:
            return
        metrics = generate_metric(metrics_cfg)
        self.train_metrics = metrics.clone(prefix='train_')
        self.val_metrics = metrics.clone(prefix='val_')
        # self.img_predict_metrics = metrics.clone(prefix='img_predict_')
        # self.xyz_predict_metrics = metrics.clone(prefix='xyz_predict_')
        self.img_predict_metrics = metrics.clone()
        self.xyz_predict_metrics = metrics.clone()

    # def configure_metric(self):
    #     metrics_cfg=getattr(self.hparams, 'metric', None)
    #     if metrics_cfg is None:
    #         return
    #     metrics = generate_metric(metrics_cfg)
    #     self.train_metrics = metrics.clone(prefix='train_')
    #     self.val_metrics = metrics.clone(prefix='val_')
    #     self.img_predict_metrics = metrics.clone(prefix='predict_')
    #     self.predict_metrics = metrics.clone(prefix='predict_')
    def training_step(self, batch, batch_idx):
        x, y1, grid, y2, batch_info = batch[0],batch[1],batch[2],batch[4],batch[5]
        pred = self({'image':[x,y1],'grid':grid,'batch_info':batch_info})
        loss,_ = self.loss(pred[1], y2)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        metrics=self.train_metrics(pred[1], y2)
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y1, grid, y2, batch_info = batch[0],batch[1],batch[2],batch[4],batch[5]
        pred = self({'image':[x,y1],'grid':grid,'batch_info':batch_info})
        loss,_ = self.loss(pred, y2)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        metrics=self.val_metrics(pred, y2)
        self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return metrics

    def predict_step(self, batch, batch_idx):
        x, y1, grid, y2, xyz_image_name, batch_info = batch[0], batch[1], batch[2], batch[4], batch[5],batch[6]
        pred = self({'image':[x,y1],'grid':grid,'batch_info':batch_info})
        out_class = pred.shape[1]
        if out_class > 1:
            pred = torch.softmax(pred, dim = 1)
        else:
            pred = torch.sigmoid(pred)
        base = xyz_image_name[0].split('.')[0]
        _, xoff, yoff = base.split('_')
        xoff, yoff = int(xoff), int(yoff)
        H, W = y1.shape[-2], x.shape[-1]  # patch大小，或者 batch_info 里有的话也行
        return {
            "pred": pred.detach().cpu(),
            "offset": (xoff, yoff),
            "patch_size": (H, W),
        }

    # def predict_step(self, batch, batch_idx, dataloader_idx=0):
    #     image_output_prob, image_output_class, image_output_eval, image_metric = self.img_predict_step(batch, batch_idx, dataloader_idx)
    #     grids = self.generate_grid_step(image_output_class, batch, batch_idx, dataloader_idx)
    #     batch['grid'] = grids

    #     xyz_output_prob, xyz_output_class, xyz_output_eval, xyz_metric = self.xyz_predict_step(batch, batch_idx, dataloader_idx)

    #     return image_output_class, xyz_output_class, image_output_eval, xyz_output_eval, image_metric, xyz_metric
    def img_predict_step(self, batch, batch_idx, dataloader_idx=0):
        # self.model.set_stage('stage1')

        image_output_prob = []
        image_output_class = []
        image_output_eval = []
        metrics = []
        # out_class = self.hparams.model['image_branch']['out_class']

        for image_data, label_data in zip(batch['image'],batch['label']):
            
            # prob_output = []
            # offsets = []
            # image_shape = image_data.shape
            # for input, xoff, yoff in normal_sampler(image_data,
            #                                         batch['sampler_info'][0],
            #                                         batch['sampler_info'][1],
            #                                         batch['sampler_info'][2]):
            #     prob = self(input)
            #     out_class = prob.shape[1]
            #     if out_class > 1:
            #         prob = torch.softmax(prob, dim = 1)
            #     else:
            #         prob = torch.sigmoid(prob)
            #     prob_output.append(prob)
            #     offsets.append([xoff,yoff])
            # prob = normal_stitch(prob_output,
            #                      offsets,
            #                      image_shape[-1],
            #                      image_shape[-2],
            #                      out_class)
            # img_class_prediction = torch.argmax(prob, dim = 0) if out_class > 1 else (prob > batch['threshold']).squeeze().to(torch.int64)
            prob, img_class_prediction = streaming_stitch_step(
                model=self, 
                image_data=image_data,  
                sampler_info=batch['sampler_info'],  
                threshold=batch['threshold'], 
                device=image_data.device
            )
            label_data = label_data.to(prob.device)
            image_output_prob.append(prob)
            image_output_class.append(img_class_prediction)
            metrics.append(self.img_predict_metrics(img_class_prediction, label_data.squeeze().to(torch.int64)))

            eval_image = cal_eval_image(img_class_prediction, label_data.squeeze().to(torch.int64))
            image_output_eval.append(eval_image)

        return image_output_prob, image_output_class, image_output_eval, metrics

    @torch.no_grad()
    def generate_grid_step(self, image_output_calss:list[torch.Tensor], batch, batch_idx, dataloader_idx = 0):
        """
            这一部分只能在CPU上生成，但是batch的数据如果是GPU那么全都是GPU，必须转到CPU才能使用，
            由于物方像方的分辨率差距太大，因此生成gird时需要对小图进行
        """
        grids = []
        grid_func = globals()[self.hparams.predict['grid_func']['type']]
        func_param = get_param_from_config_func(grid_func, self.hparams.predict['grid_func'])

        for image_output, image_path,point_pair in zip(image_output_calss, batch['image_path'], batch['points_pair']):

            # grids.append(torch.tensor(torch.rand(1, 2, image_output.shape[-2], image_output.shape[-1]*4-2, device = image_output_calss[0].device)))
            # continue

            predict_raster_io = RasterIO(image_path)
            predict_trans = predict_raster_io.get_geo_trans()

            ref_raster_io = RasterIO(batch['ref_path'])
            ref_trans = ref_raster_io.get_geo_trans()
            ref_shape = ref_raster_io.shape()

            ## 这里修改为利用globals()通过cfg生成对应的grid生成函数，从当前环境中获取输入参数
            # img_pred_sum=img_pred_sum.astype(np.uint8)

            func_param_cfg = {'points_pair': point_pair.cpu().numpy(),
                              'model_predict': image_output.cpu().numpy(),
                              'ref_shape':ref_shape,
                              'geo_trans':ref_trans,
                              'predict_trans':predict_trans}

            func_param.update(func_param_cfg)
            grid = grid_func(**func_param)

            grids.append(torch.tensor(grid[0], device = image_output_calss[0].device).unsqueeze(0))

        return grids

    def xyz_predict_step(self, batch, batch_idx, dataloader_idx=0):
        # self.model.set_stage('stage2')
        # "image" "label" 'points_pair' 'xyz_label' 'xyz_image' "batch_info" "name" "corners"
        prob_output = []
        offsets = []
        # out_class = self.hparams.model['xyz_branch']['out_class']
        for input, xoff, yoff in fusion_sampler(batch,
                                                batch['sampler_info'][0],
                                                batch['sampler_info'][1],
                                                batch['sampler_info'][2]):
            prob = self(input)
            out_class = prob.shape[1]
            if out_class > 1:
                prob = torch.softmax(prob, dim = 1)
            else:
                prob = torch.sigmoid(prob)
            prob_output.append(prob)
            offsets.append([xoff, yoff])
        prob = normal_stitch(prob_output,
                             offsets,
                             batch['xyz_image'].shape[-1],
                             batch['xyz_image'].shape[-2],
                             out_class)
        class_prediction = torch.argmax(prob, dim = 0) if out_class > 1 else (prob > batch['threshold']).squeeze().to(torch.int64)

        label_data = batch['xyz_label'].squeeze().to(torch.int64)
        metrics = self.xyz_predict_metrics(class_prediction, label_data)
        eval_image = cal_eval_image(class_prediction, label_data)
        return prob, class_prediction, eval_image, metrics

    # def on_predict_epoch_end(self):
    #     # output = self.img_predict_metrics.compute()
    #     # self.log_dict(output)
    #     self.img_predict_metrics.reset()

    #     # output = self.xyz_predict_metrics.compute()
    #     # self.log_dict(output)
    #     self.xyz_predict_metrics.reset()
    def predict_epoch_end(self, outputs):
        prob_list = []
        offsets = []
        patch_sizes = []

        for o in outputs:
            prob_list.append(o["pred"])
            offsets.append(o["offset"])
            patch_sizes.append(o["patch_size"])

        # 假设所有patch大小一致
        H, W = patch_sizes[0]

        # 根据offset计算完整图大小（示例）
        max_x = max([x + W for (x, y) in offsets])
        max_y = max([y + H for (x, y) in offsets])

        # 也可以用参考影像大小，二选一
        full_H, full_W = max_y, max_x

        out_class = prob_list[0].shape[1]

        stitched_prob = normal_stitch(prob_list, offsets, full_W, full_H, out_class)

        if out_class > 1:
            class_prediction = torch.argmax(stitched_prob, dim=0)
        else:
            class_prediction = (stitched_prob.squeeze() > 0.5).long()   #这里的0.5只能是手动设置了

        return {
            "stitched_prob": stitched_prob,
            "class_prediction": class_prediction,
        }
