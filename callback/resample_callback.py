# -*- coding: utf-8 -*-
import logging
from lightning.pytorch.callbacks import Callback

# 获取 logger，方便在终端输出标准格式的日志
logger = logging.getLogger(__name__)

__all__ = ['ResampleDatasetCallback']

class ResampleDatasetCallback(Callback):
    """
    自定义数据集重采样回调 (Dataset Resampling Callback)
    
    功能:
        在每个训练 Epoch 开始前 (on_train_epoch_start)，调用 Dataset 的 `resample_views` 方法。
        用于 MVS/多视角任务中，动态改变每个 Epoch 参与训练的视角组合。
        
    Args:
        dataset: 需要进行重采样的 Dataset 实例。该实例必须包含 `resample_views` 方法。
    """
    def __init__(self, dataset):
            super().__init__()
            self.dataset = dataset

    def on_train_epoch_start(self, trainer, pl_module):
        """
        在每个训练 Epoch 开始时触发
        """
        if hasattr(self.dataset, 'resample_views'):
            # 获取当前 epoch
            current_epoch = trainer.current_epoch
            
            # 只在主进程打印日志
            if trainer.is_global_zero:
                logger.info(f"🔄 [ResampleDatasetCallback] Epoch {current_epoch}: Resampling with seed {current_epoch}...")
            
            # 【关键修改】将 epoch 传进去作为 seed！
            # 确保 GPU 6 和 GPU 7 在第 N 轮选出的数据是完全一模一样的
            self.dataset.resample_views(seed=current_epoch)
            
        else:
            if trainer.is_global_zero and trainer.current_epoch == 0:
                logger.warning(f"⚠️ Warning: Dataset {type(self.dataset).__name__} missing 'resample_views'.")
