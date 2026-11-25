from typing import Union,List,Tuple
from pytorch_lightning.callbacks import BaseFinetuning
class FreezeLayerCallback(BaseFinetuning):
    def __init__(self, freezelayers_cfg: Union[List[str], Tuple[str, ...]],free_bn = True):
         super().__init__()
         self._freezelayers_cfg = freezelayers_cfg
         self._freeze_bn = free_bn

    def freeze_before_training(self, pl_module):
        for layer_name in self._freezelayers_cfg:
            # self.freeze(getattr(pl_module._modules['model'],layer_name),train_bn= not self._freeze_bn)
            self.freeze(getattr(pl_module.model,layer_name),train_bn= not self._freeze_bn)

    def finetune_function(self, pl_module, current_epoch, optimizer):
        #  # When `current_epoch` is 10, feature_extractor will start training.
        #  if current_epoch == self._unfreeze_at_epoch:
        #      self.unfreeze_and_add_param_group(
        #          modules=pl_module.feature_extractor,
        #          optimizer=optimizer,
        #          train_bn=True,
        #      )
        return None
