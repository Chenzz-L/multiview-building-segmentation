"""
MIT License

Copyright (c) 2019 Ildoo Kim

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
# class WarmupLR(_LRScheduler):
#     """The WarmupLR scheduler
#
#     This scheduler is almost same as NoamLR Scheduler except for following
#     difference:
#
#     NoamLR:
#         lr = optimizer.lr * model_size ** -0.5
#              * min(step ** -0.5, step * warmup_step ** -1.5)
#     WarmupLR:
#         lr = optimizer.lr * warmup_step ** 0.5
#              * min(step ** -0.5, step * warmup_step ** -1.5)
#
#     Note that the maximum lr equals to optimizer.lr in this scheduler.
#
#     """
#     def __init__(
#         self,
#         optimizer: torch.optim.Optimizer,
#         warmup_steps: Union[int, float] = 25000,
#         last_epoch: int = -1,
#     ):
#         assert check_argument_types()
#         self.warmup_steps = warmup_steps
#
#         # __init__() must be invoked before setting field
#         # because step() is also invoked in __init__()
#         super().__init__(optimizer, last_epoch)
#
#     def __repr__(self):
#         return f"{self.__class__.__name__}(warmup_steps={self.warmup_steps})"
#
#     def get_lr(self):
#         step_num = self.last_epoch + 1
#         return [
#             lr
#             * self.warmup_steps ** 0.5
#             * min(step_num ** -0.5, step_num * self.warmup_steps ** -1.5)
#             for lr in self.base_lrs
#         ]
#
#     def set_step(self, step: int):
#         self.last_epoch = step
from functools import partial
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import ExponentialLR

# class GradualWarmupScheduler(_LRScheduler):
#     """ Gradually warm-up(increasing) learning rate in optimizer.
#     Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
#     Args:
#         optimizer (Optimizer): Wrapped optimizer.
#         multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
#         total_epoch: target learning rate is reached at total_epoch, gradually
#         after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
#     """
#
#     def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
#         self.multiplier = multiplier
#         if self.multiplier < 1.:
#             raise ValueError('multiplier should be greater thant or equal to 1.')
#         self.total_epoch = total_epoch
#         self.after_scheduler = after_scheduler
#         self.finished = False
#         super(GradualWarmupScheduler, self).__init__(optimizer)
#
#     def get_lr(self):
#         if self.last_epoch > self.total_epoch:
#             if self.after_scheduler:
#                 if not self.finished:
#                     self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
#                     self.finished = True
#                 return self.after_scheduler.get_last_lr()
#             return [base_lr * self.multiplier for base_lr in self.base_lrs]
#
#         if self.multiplier == 1.0:
#             return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
#         else:
#             return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]
#
#     def step_ReduceLROnPlateau(self, metrics, epoch=None):
#         if epoch is None:
#             epoch = self.last_epoch + 1
#         self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
#         if self.last_epoch <= self.total_epoch:
#             warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]
#             for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
#                 param_group['lr'] = lr
#         else:
#             if epoch is None:
#                 self.after_scheduler.step(metrics, None)
#             else:
#                 self.after_scheduler.step(metrics, epoch - self.total_epoch)
#
#     def step(self, epoch=None, metrics=None):
#         if type(self.after_scheduler) != ReduceLROnPlateau:
#             if self.finished and self.after_scheduler:
#                 if epoch is None:
#                     self.after_scheduler.step(None)
#                 else:
#                     self.after_scheduler.step(epoch - self.total_epoch)
#                 self._last_lr = self.after_scheduler.get_last_lr()
#             else:
#                 return super(GradualWarmupScheduler, self).step(epoch)
#         else:
#             self.step_ReduceLROnPlateau(metrics, epoch)

# class GradualWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
#     """ Gradually warm-up(increasing) learning rate in optimizer.
#     Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
#     Args:
#         optimizer (Optimizer): Wrapped optimizer.
#         multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
#         total_epoch: target learning rate is reached at total_epoch, gradually
#         after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
#     """

#     def __init__(self, optimizer, multiplier, warmup_epoch,last_epoch=-1,after_scheduler=None):
#         self.multiplier = multiplier
#         if self.multiplier < 1.:
#             raise ValueError('multiplier should be greater thant or equal to 1.')
#         self.warmup_epoch = warmup_epoch
#         self.finished=False
#         # self.after_scheduler = after_scheduler if after_scheduler else ExponentialLR(self.optimizer,gamma=0.9)
#         self.after_scheduler = after_scheduler if after_scheduler else partial(ExponentialLR,gamma=0.9)
#         super(GradualWarmupScheduler, self).__init__(optimizer,last_epoch)


#     def get_lr(self):
#         #在warmup之外
#         if self.last_epoch > self.warmup_epoch:
#             #使用after_scheduler
#             if not self.finished:
#                 # self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
#                 # self.after_scheduler._last_lr=[base_lr ** 1.0001 for base_lr in self.base_lrs]
#                 self.after_scheduler=self.after_scheduler(optimizer=self.optimizer)
#                 self.finished = True
#                 # self.after_scheduler.step()

#             return self.after_scheduler._last_lr
#             # return self.after_scheduler.get_lr

#         # 在warmup之内
#         out_lr=[base_lr * ((self.multiplier - 1.) * self.last_epoch / self.warmup_epoch + 1.) for base_lr in self.base_lrs]
#         # self.after_scheduler.base_lrs=out_lr
#         return out_lr

#     def get_last_lr(self):
#         return self._last_lr

#     def step_ReduceLROnPlateau(self, metrics, epoch=None):
#         if epoch is None:
#             epoch = self.last_epoch + 1
#         self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
#         if self.last_epoch <= self.total_epoch:
#             warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]
#             for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
#                 param_group['lr'] = lr
#         else:
#             self.after_scheduler.step(metrics)

#     def step(self,metrics=None):
#         if self.finished:
#             if type(self.after_scheduler) != ReduceLROnPlateau:
#                 self.after_scheduler.step()
#                 self._last_lr = self.after_scheduler._last_lr
#             else:
#                 self.after_scheduler.step(metrics)
#         else:
#             return super(GradualWarmupScheduler, self).step()


class GradualWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer,max_lr, warmup_epoch,last_epoch=-1,after_scheduler=None):
        if not isinstance(max_lr,(list,tuple)):
            if len(optimizer.param_groups)!=1:
                max_lr=[max_lr,max_lr]
            else:
                max_lr=[max_lr]
            
        self.max_lr=max_lr
        self.warmup_epoch = warmup_epoch
        self.finished=False
        # self.after_scheduler = after_scheduler if after_scheduler else ExponentialLR(self.optimizer,gamma=0.9)
        self.after_scheduler = after_scheduler if after_scheduler else partial(ExponentialLR,gamma=0.9)
        super(GradualWarmupScheduler, self).__init__(optimizer,last_epoch)


    def get_lr(self):
        #在warmup之外
        if self.last_epoch == self.warmup_epoch:
            #使用after_scheduler
            if not self.finished:
                for i, data in enumerate(zip(self.optimizer.param_groups, self.max_lr)):
                    param_group, lr = data
                    param_group['lr'] = lr

                self._last_lr = [group['lr'] for group in self.optimizer.param_groups]
                self.finished = True
            self.after_scheduler=self.after_scheduler(optimizer=self.optimizer)
            return self.after_scheduler._last_lr

        elif self.last_epoch>self.warmup_epoch:
            return self.after_scheduler._last_lr
        
        # 在warmup之内
        out_lr=[base_lr + ((self.max_lr[i] - base_lr) * self.last_epoch / self.warmup_epoch ) for i,base_lr in enumerate(self.base_lrs)]
        return out_lr

    def get_last_lr(self):
        return self._last_lr

    def step(self,metrics=None):
        if self.finished:
            if type(self.after_scheduler) != ReduceLROnPlateau:
                self.after_scheduler.step()
                self._last_lr = self.after_scheduler._last_lr
            else:
                self.after_scheduler.step(metrics)
        else:
            return super(GradualWarmupScheduler, self).step()
