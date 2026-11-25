from typing import List,Tuple
import copy
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn.functional as F
from torch import nn

from loss import build_loss

class SegmentationCriterion(nn.Module):
    """
    support single task and multi task
    dont support aux loss now
    """
    def __init__(self,losses_cfg):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories
        """
        super().__init__()
        #output_losses :
        # multi task : [[loss1,loss2...],[loss1,loss2]] or [[loss1],[loss2]] or [[loss1]] is multi task same lossfunc
        # single task : [[loss1]] or [[loss1,loss2]]
        #
        self.output_losses=build_loss(losses_cfg)

    def cal_loss(self,output,label,losses):
        return sum(loss_func(output,label) for loss_func in losses)
        # if isinstance(losses,(Tuple,List)):
        #     loss=sum(loss_func(output,label) for loss_func in losses)
        # else:
        #     loss=losses(output,label)
        # return loss

    def loss_masks(self, outputs, labels):
        assert type(outputs)==type(labels)
        out_loss=[]
        log_loss=[]
        # multi task
        if isinstance(outputs,(Tuple,List)):
            if len(outputs)!=len(self.output_losses):
                self.output_losses=tuple(copy.copy(self.output_losses[0]) for i in range(len(outputs)))
            else:
                self.output_losses=self.output_losses

            for output,label,losses in zip(outputs,labels,self.output_losses):
            # upsample predictions to the target size
                output = F.interpolate(
                    output, size=label.shape[-2:], mode="bilinear", align_corners=False
                )
                loss=self.cal_loss(output,label,losses)
                out_loss.append(loss)
                log_loss.append(loss.detach().clone())


        else:
            outputs = F.interpolate(
                    outputs, size=labels.shape[-2:], mode="bilinear", align_corners=False
                )
            loss=self.cal_loss(outputs,labels,self.output_losses[0])
            out_loss.append(loss)
            log_loss.append(loss.detach().clone())

        return sum(out_loss),log_loss


    def forward(self, outputs, targets):
        return self.loss_masks(outputs,targets)

# if __name__=='__main__':
#     criterion=SegmentationCriterion(1,'SigmoidDiceLoss,focalloss')
#     x=[torch.rand([16,1,512,512],device='cuda:0') for _ in range(2)]
#     y=[torch.rand([16,1,512,512],device='cuda:0') for _ in range(2)]
#     loss,log_loss=criterion(x,y)
#     print(loss)
