import os
import sys

from torch import Tensor


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from optimizer import *
from loss.criterion import *
from builder import generate_model
from metric import generate_metric
from callback import generate_callback
from loggers import generate_logger
from lightning.pytorch.core.module import LightningModule
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from torch.optim.optimizer import Optimizer
class MInterface(LightningModule):
    def __init__(self,**kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.load_model()
        self.configure_loss()
        self.configure_metric()

    def forward(self, img):
        return self.model(img)
    def configure_optimizers(self):

        opeitmizer_cfg=getattr(self.hparams, 'optimizer', None)
        if opeitmizer_cfg is None:
            return None
        optimizer=build_optimizer(self.model,opeitmizer_cfg)

        if self.hparams.lr_scheduler is None:
            return optimizer
        else:
            lr_scheduler_cfg = self.hparams.lr_scheduler
            scheduler=build_scheduler(optimizer,lr_scheduler_cfg)
            return [optimizer], [scheduler]
    def configure_loss(self):
        loss = getattr(self.hparams, 'loss', None)
        if loss is None:
            return
        self.loss = SegmentationCriterion(loss)
    def load_model(self):
        self.model = generate_model(self.hparams.model)
    def configure_metric(self):
        metrics_cfg=getattr(self.hparams, 'metric', None)
        if metrics_cfg is None:
            return
        metrics = generate_metric(metrics_cfg)

        self.train_metrics = metrics.clone(prefix='train_')
        self.val_metrics = metrics.clone(prefix='val_')
        self.predict_metrics = metrics.clone()

    def configure_callbacks(self):
        callbacks_cfg = getattr(self.hparams, 'callback', None)
        if callbacks_cfg is None:
            return []
        return generate_callback(self.hparams.callback)

    # def configure_loggers(self):
    #     loggers_cfg = getattr(self.hparams, 'logger', None)
    #     if loggers_cfg is None:
    #         return
    #     self.loggers=generate_logger(loggers_cfg)

    # def on_train_epoch_end(self):
    # def on_train_epoch_start(self):
    # def on_before_zero_grad(self, optimizer: Optimizer):
    # def on_before_backward(self, loss: Tensor):
    #     tensorboard_log = next((logger.experiment for logger in self.loggers if isinstance(logger, TensorBoardLogger)), None)
    #     if tensorboard_log is None:
    #         return
    #     # optimizer.param_groups[0].
    #     for name, param in self.model.named_parameters():
    #         if param.grad is None:
    #             continue
    #         tensorboard_log.add_histogram(name, param.clone().cpu().data.numpy(), self.current_epoch)
    #         tensorboard_log.add_histogram(name + '/grad', param.grad.clone().cpu().data.numpy(), self.current_epoch)

    def on_validation_epoch_end(self):
        # use log_dict instead of log
        # metrics are logged with keys: val_Accuracy, val_Precision and val_Recall
        output = self.val_metrics.compute()
        self.log_dict(output)
        # reset metrics at the end of the epoch
        self.val_metrics.reset()

    # def on_predict_epoch_end(self):
    #     output = self.img_predict_metrics.compute()
    #     self.log_dict(output)
    #     self.img_predict_metrics.reset()

    #     output = self.xyz_predict_metrics.compute()
    #     self.log_dict(output)
    #     self.xyz_predict_metrics.reset()

if __name__ == '__main__':
    cfg=load_yaml(r'/path/to/project/GraduateCareer\code\mvmapper_reconstruct\cfgs\test_for_pl.yaml')
    model=MInterface(**cfg)
    opt,lr_scheduler=model.configure_optimizers()
    callback=model.configure_callbacks()
    print(model)
