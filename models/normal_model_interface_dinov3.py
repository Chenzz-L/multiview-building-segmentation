import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time
import torch

from sampler.normal_predict_sampler import normal_sampler, normal_stitch, normal_sampler_with_dsm,stitch_neck_features_dict
from .model_interface import MInterface
from metric.tool import cal_eval_image

class NormalMInterface_DINOv3(MInterface):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)

    def training_step(self, batch, batch_idx):
        x, dsm, y = batch
        pred = self([x,dsm])
        loss, _ = self.loss(pred, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        metrics = self.train_metrics(pred, y)
        self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, dsm, y = batch
        pred = self([x,dsm])
        loss, _ = self.loss(pred, y)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        metrics = self.val_metrics(pred, y)
        self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return metrics

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        crop_stride = self.hparams.predict['crop_stride']
        crop_width = self.hparams.predict['crop_width']
        crop_height = self.hparams.predict['crop_height']
        threshold = self.hparams.predict['threshold']
        image_shape=batch['image'].shape
        prob_output = []
        offsets = []
        with torch.no_grad():
            for image, dsm, xoff, yoff in normal_sampler_with_dsm(batch, crop_width, crop_height, crop_stride):
                image = image.unsqueeze(0)
                dsm = dsm.unsqueeze(0)
                prob = self([image,dsm])
                out_class = prob.shape[1]
                if out_class > 1:
                    prob = torch.softmax(prob, dim = 1)
                else:
                    prob = torch.sigmoid(prob)
                prob_output.append(prob)
                offsets.append([xoff,yoff])

        # stitch output the torch tensor
        prob = normal_stitch(prob_output,
                             offsets,
                             image_shape[-1],
                             image_shape[-2],
                             out_class)
        class_prediction = torch.argmax(prob, dim = 0) if out_class > 1 else (prob > threshold).unsqueeze(0).to(torch.int64)

        # calculate the metrics
        label_data = batch['label'].unsqueeze(0).to(torch.int64)
        metrics = self.predict_metrics(class_prediction, label_data)

        # eval
        eval_image = cal_eval_image(class_prediction, label_data)
        return class_prediction, eval_image, metrics
    
        # ------------保存特征--------------------------
    # def predict_step(self, batch, batch_idx):
    #     image, dsm = batch[0], batch[1]
    #     _, _, H, W = image.shape

    #     crop_size = 512
    #     stride = 360

    #     all_neck_feats = []

    #     for y0 in range(0, H, stride):
    #         for x0 in range(0, W, stride):
    #             yy0 = min(max(y0, 0), H - crop_size)
    #             xx0 = min(max(x0, 0), W - crop_size)
    #             yy1 = yy0 + crop_size
    #             xx1 = xx0 + crop_size

    #             img_patch = image[:, :, yy0:yy1, xx0:xx1]
    #             dsm_patch = dsm[:, :, yy0:yy1, xx0:xx1] if dsm is not None else None

    #             with torch.no_grad():
    #                 outputs = self([img_patch, dsm_patch])

    #             # neck_feats: Dict[str, Tensor[B, C, h, w]]
    #             neck_feats = outputs

    #             # ⭐ 去掉 batch 维：B=1
    #             neck_feats_cpu = {
    #                 k: v.squeeze(0).detach().half().cpu()
    #                 for k, v in neck_feats.items()
    #             }

    #             all_neck_feats.append({
    #                 "offset": (xx0, yy0),
    #                 "feat": neck_feats_cpu,
    #             })

    #     full_neck_feats = stitch_neck_features_dict(
    #         all_neck_feats,
    #         image_shape=(H, W),
    #         strides_dict={
    #             "res2": 4,
    #             "res3": 8,
    #             "res4": 16,
    #             "res5": 32,
    #         }
    #     )

    #     return full_neck_feats
    
    
    def on_predict_epoch_end(self):
        self.predict_metrics.reset()
