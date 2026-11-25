from typing import Any,Literal
import torch
import torch.nn.functional as F
from torchmetrics.segmentation import MeanIoU

class IoU(MeanIoU):
    def __init__(
        self,
        num_classes: int,
        threshold: float = 0.5,
        # task: Literal["binary", "multiclass"] = "binary",
        include_background: bool = False,
        per_class: bool = True,
        input_format: Literal["one-hot", "index"] = "index",
        **kwargs: Any,
    ) -> None:
        # super().__init__(num_classes, include_background, per_class, input_format, **kwargs)
        super().__init__(num_classes, include_background, per_class, **kwargs)
        # self.task = task
        self.threshold = threshold
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        if preds.is_floating_point():
            if not torch.all((preds >= 0) * (preds <= 1)):
                # preds is logits, convert with sigmoid or softmax
                if preds.shape[1] > 1:
                    preds = torch.softmax(preds, dim=1)
                    _, preds = torch.max(preds, dim=1, keepdim=True)
                    # one_hot_preds = F.one_hot(predicted_classes, num_classes=self.num_classes).permute(0, 3, 1, 2).contiguous()
                elif preds.shape[1] == 1:
                    preds = torch.sigmoid(preds)
                    preds = (preds > self.threshold).to(torch.int64)
                else:
                    raise ValueError("Unsupported dimension for predictions.")

        super().update(preds.to(torch.int64), target.to(torch.int64))

if __name__=='__main__':
    # 假设 num_classes 是类别的数量
    num_classes = 2
    miou = IoU(num_classes=num_classes, per_class=True,include_background=False)
    # miou = MeanIoU(num_classes=3,per_class=True)
    # 假设 outputs 是模型的 logits 输出，shape 为 (N, C, H, W)
    # 其中 N 是批次大小，C 是类别数，H 和 W 是特征图的高度和宽度
    outputs = torch.randn(10, 1, 256, 256)
    # outputs = torch.randint(0, 3, (10, 1, 256, 256))

    # 假设 targets 是类别索引，shape 为 (N, H, W)
    targets = torch.randint(0, 2, (10, 1, 256, 256))

    # 更新度量
    miou.update(preds=outputs, target = targets)

    # 计算度量
    mean_iou = miou.compute()
    print(f"Mean IoU: {mean_iou}")

    # miou = MeanIoU(num_classes = 4, per_class=True,input_format='index')
    # preds = torch.randint(0, 4, (10, 1, 128, 128))
    # target = torch.randint(0, 4, (10, 1, 128, 128))
    # # target = preds
    # print(miou(preds, target))
