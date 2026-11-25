import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch import Tensor
import numpy as np
import cv2
from torchmetrics import Metric

from utils.boundary_process import get_edge_data
from metric.tool import calc_confusion_matrix, calc_tp_fp_fn_from_conf_matrix, calculate_acc_from_tp_tn_fp_cuda
class Boundary_Metric(Metric):
    def __init__(self, edge_buffers = (0, 1, 2, 3, 4), **kwargs):
        super().__init__(**kwargs)
        self.edge_buffers = edge_buffers

        # self.add_state("tp", default = [], dist_reduce_fx="sum")
        # self.add_state("fp", default = [], dist_reduce_fx="sum")
        # self.add_state("fn", default = [], dist_reduce_fx="sum")

        self.add_state("tp", default = torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("fp", default = torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("fn", default = torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        # preds, target = self._input_format(preds, target)
        if preds.shape != target.shape:
            raise ValueError("preds and target must have the same shape")

        binary_data = preds.detach().cpu().to(torch.uint8).numpy()
        label_data = target.detach().cpu().to(torch.uint8).numpy()

        # binary_data = cv2.morphologyEx(binary_data, cv2.MORPH_OPEN, kernel = np.ones((5, 5), np.uint8))  # 开运算
        binary_data = get_edge_data(binary_data)
        label_data = get_edge_data(label_data)
        for i, buffer in enumerate(self.edge_buffers):
            binary_buff_bd_data = cv2.dilate(binary_data, kernel=np.ones((2 * buffer + 1, 2 * buffer + 1), np.uint8))
            label_buff_bd_data = cv2.dilate(label_data, kernel=np.ones((2 * buffer + 1, 2 * buffer + 1), np.uint8))
            confusion_matrix = calc_confusion_matrix(binary_buff_bd_data.flatten(),
                                                        label_buff_bd_data.flatten(), class_number = 2)
            tp, fp, fn = calc_tp_fp_fn_from_conf_matrix(confusion_matrix, class_id=1)
            self.tp += torch.tensor(tp).to(preds.device)
            self.fp += torch.tensor(fp).to(preds.device)
            self.fn += torch.tensor(fn).to(preds.device)
            # self.tp.append(torch.Tensor(tp, device = preds.device))
            # self.fp.append(torch.Tensor(fp, device = preds.device))
            # self.fn.append(torch.Tensor(fn, device = preds.device))
    def compute(self) -> Tensor:
        return calculate_acc_from_tp_tn_fp_cuda(self.tp, self.fp, self.fn)[0]

class Boundary_IoU(Boundary_Metric):
    def __init__(self, edge_buffers = (0, 1, 2, 3, 4), **kwargs):
        super().__init__(edge_buffers , **kwargs)
    def compute(self) -> Tensor:
        return calculate_acc_from_tp_tn_fp_cuda(self.tp, self.fp, self.fn)[0]
class Boundary_Precision(Boundary_Metric):
    def __init__(self, edge_buffers = (0, 1, 2, 3, 4), **kwargs):
        super().__init__(edge_buffers , **kwargs)
    def compute(self) -> Tensor:
        return calculate_acc_from_tp_tn_fp_cuda(self.tp, self.fp, self.fn)[1]

class Boundary_Recall(Boundary_Metric):
    def __init__(self, edge_buffers = (0, 1, 2, 3, 4), **kwargs):
        super().__init__(edge_buffers , **kwargs)
    def compute(self) -> Tensor:
        return calculate_acc_from_tp_tn_fp_cuda(self.tp, self.fp, self.fn)[2]


class Boundary_F1Score(Boundary_Metric):
    def __init__(self, edge_buffers = (0, 1, 2, 3, 4), **kwargs):
        super().__init__(edge_buffers , **kwargs)
    def compute(self) -> Tensor:
        return calculate_acc_from_tp_tn_fp_cuda(self.tp, self.fp, self.fn)[3]



if __name__=='__main__':
    x = torch.randint(0, 2,(512, 512), device='cuda:0')
    y = x
    # y = torch.randint(0, 2,(512,512), device='cuda:0')
    mymetric = Boundary_Precision().to(x.device)
    a = mymetric(x, y)
    a = mymetric.compute()
    print(a)
