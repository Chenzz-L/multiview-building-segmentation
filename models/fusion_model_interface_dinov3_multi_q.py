"""One-pass fixed-Q sweep prediction for SAIL image memory."""

from __future__ import annotations

import torch

from .fusion_model_interface_dinov3_crossattn import (
    FusionMInterface_DINOv3_CrossAttn,
)


class FusionMInterface_DINOv3_MultiQ(FusionMInterface_DINOv3_CrossAttn):
    """Load one max-Q tail batch and emit several prefix-Q predictions.

    The encoder/neck runs once per patch and the already loaded tail tensors
    are sliced in memory for each Q.  This is intentionally different from
    global Top-K: the view order comes from the deterministic nested manifest.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        values = getattr(self.hparams, "predict_tail_q_values", None)
        self.predict_tail_q_values = tuple(
            sorted({int(value) for value in (values or ())})
        )
        if not self.predict_tail_q_values:
            raise ValueError("predict_tail_q_values must not be empty")

    def predict_step(self, batch, batch_idx):
        data = self._unpack_fusion_batch(batch)
        image, dsm, xyz_name = data["image"], data["dsm"], data["names"]
        if xyz_name is None:
            raise ValueError("multi-Q prediction requires sample names")
        if data.get("all_tail_paths") is not None:
            raise ValueError(
                "multi-Q fixed-prefix prediction requires loaded tail tensors, "
                "not all-tail path streaming"
            )
        if data.get("tail_features") is None:
            raise ValueError("multi-Q prediction requires tail_features")

        logits_by_q = self.model(
            x=(image, dsm),
            output_attn=False,
            tail_multi_q_eval_choices=self.predict_tail_q_values,
            **self._model_fusion_kwargs(data),
        )
        probabilities = {}
        for q, logits in logits_by_q.items():
            probability = (
                torch.softmax(logits, dim=1)
                if logits.shape[1] > 1
                else torch.sigmoid(logits)
            )
            probabilities[str(int(q))] = probability.detach().cpu()

        _, xoff, yoff = xyz_name[0].split("_")
        height, width = image.shape[-2:]
        return {
            "pred_by_topk": probabilities,
            "offset": (int(xoff), int(yoff)),
            "patch_size": (height, width),
            "xyz_name": xyz_name[0],
        }
