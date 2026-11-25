import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time
import torch

from sampler.normal_predict_sampler import normal_sampler, normal_stitch
from .model_interface import MInterface
from metric.tool import cal_eval_image

class NormalMInterface(MInterface):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss, _ = self.loss(pred, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        metrics = self.train_metrics(pred, y)
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
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
            for input, xoff, yoff in normal_sampler(batch, crop_width, crop_height, crop_stride):
                input = input.unsqueeze(0)
                prob = self(input)
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
    
    # def predict_step(self, batch, batch_idx, dataloader_idx=0):
    #     crop_stride = self.hparams.predict['crop_stride']
    #     crop_width = self.hparams.predict['crop_width']
    #     crop_height = self.hparams.predict['crop_height']
    #     threshold = self.hparams.predict['threshold']

    #     images = batch['image']      # (B, C, H, W)
    #     labels = batch['label']      # (B, H, W)
    #     if images.ndim == 3:
    #         images = images.unsqueeze(0)
    #     if labels.ndim == 3:
    #         labels = labels.unsqueeze(0)
    #     B = images.shape[0]

    #     preds, evals, metrics_list = [], [], []

    #     for b in range(B):
    #         image = images[b]
    #         label = labels[b]
    #         image_shape = image.shape
    #         prob_output, offsets = [], []

    #         for input, xoff, yoff in normal_sampler({"image": image}, crop_width, crop_height, crop_stride):
    #             input = input.unsqueeze(0)
    #             prob = self(input)
    #             out_class = prob.shape[1]
    #             if out_class > 1:
    #                 prob = torch.softmax(prob, dim=1)
    #             else:
    #                 prob = torch.sigmoid(prob)
    #             prob_output.append(prob)
    #             offsets.append([xoff, yoff])

    #         prob = normal_stitch(prob_output, offsets, image_shape[-1], image_shape[-2], out_class)
    #         class_prediction = (
    #             torch.argmax(prob, dim=0)
    #             if out_class > 1
    #             else (prob > threshold).unsqueeze(0).to(torch.int64)
    #         )

    #         metrics = self.predict_metrics(class_prediction, label.unsqueeze(0).to(torch.int64))
    #         eval_image = cal_eval_image(class_prediction, label.unsqueeze(0).to(torch.int64))

    #         preds.append(class_prediction)
    #         evals.append(eval_image)
    #         metrics_list.append(metrics)

    #     return preds, evals, metrics_list
    # def on_predict_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
    #     # 在这里重置，清空内存
    #     self.predict_metrics.reset()
        
    def on_predict_epoch_end(self):
        self.predict_metrics.reset()
