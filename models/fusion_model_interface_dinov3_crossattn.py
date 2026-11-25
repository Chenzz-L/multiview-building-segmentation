"""
FusionMInterface_DINOv3_CrossAttn
=================================
鍩轰簬 FusionMInterface_DINOv3 淇敼锛屼娇鐢?Cross-Attention 铻嶅悎
5 涓浉鏈虹殑 per-camera 鐗瑰緛涓?TDOM 缂栫爜鍣ㄨ緭鍑虹壒寰併€?"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import math
import torch.nn.functional as F
import numpy as np
import tifffile
from lightning.fabric.utilities.apply_func import move_data_to_device

from .model_interface import MInterface
from metric import generate_metric
import cv2


class FusionMInterface_DINOv3_CrossAttn(MInterface):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        bg_cfg = getattr(self.hparams, "background_consistency", None)
        self.use_background_consistency = bool(
            self._cfg_get(bg_cfg, "enabled", False)
        )
        self.background_consistency_weight = float(
            self._cfg_get(bg_cfg, "loss_weight", 0.05)
        )
        self.tdom_background_threshold = float(
            self._cfg_get(bg_cfg, "tdom_bg_threshold", 0.1)
        )
        view_cfg = getattr(self.hparams, "view_supervision", None)
        self.use_view_supervision = bool(
            self._cfg_get(view_cfg, "enabled", False)
        )
        self.view_supervision_weight = float(
            self._cfg_get(view_cfg, "loss_weight", 0.001)
        )
        fusion_module = getattr(self.model, "cross_attn_fusion", None)
        off_cfg = getattr(self.hparams, "tail_off_supervision", None)
        self.use_tail_off_supervision = bool(self._cfg_get(off_cfg, "enabled", False))
        self.tail_off_loss_weight = float(self._cfg_get(off_cfg, "loss_weight", 1.0))
        self.normalize_tail_off_loss = bool(self._cfg_get(off_cfg, "normalize", True))
        if self.use_tail_off_supervision:
            if not math.isfinite(self.tail_off_loss_weight) or self.tail_off_loss_weight <= 0:
                raise ValueError("tail_off_supervision.loss_weight must be finite and > 0")
            if (not getattr(self.model, "_uses_sail_image_memory", False)
                    or getattr(self.model, "tail_logit_correction_enabled", False)
                    or getattr(fusion_module, "memory_output_mode", None) != "anchor_plus_tail"
                    or self.use_background_consistency):
                raise ValueError("tail-off supervision requires ordinary dense SAIL; disable background consistency/logit correction")
            # These are the same model's OFF metrics, not an independent
            # pretrained R-only teacher. Lightning resets/reduces objects.
            self.tail_off_train_metrics = self.train_metrics.clone(prefix="train_tail_off_")
            self.tail_off_val_metrics = self.val_metrics.clone(prefix="val_tail_off_")
        self.log_swin_r_stats = bool(
            getattr(fusion_module, "swin_r_enabled", False)
            or getattr(fusion_module, "multi_latent_swin_r_enabled", False)
            or bool(getattr(fusion_module, "single_r_swin_set_scales", ()))
            or getattr(fusion_module, "tail_local_context_enabled", False)
            or bool(getattr(fusion_module, "tail_dual_memory_context_scales", ()))
        )
        debug_cfg = getattr(self.hparams, "cross_attn_debug", None)
        self.cross_attn_debug_enabled = bool(
            self._cfg_get(debug_cfg, "enabled", False)
        )
        self.cross_attn_debug_every_n_steps = int(
            self._cfg_get(debug_cfg, "every_n_steps", 100)
        )
        self._cross_attn_initial_params = {}

    @staticmethod
    def _cfg_get(cfg, key, default=None):
        if cfg is None:
            return default
        if isinstance(cfg, dict):
            return cfg.get(key, default)
        return getattr(cfg, key, default)

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        """Keep global-TopK tail candidates on CPU until after selection.

        Lightning normally transfers the complete nested batch recursively.
        For 32 multi-scale candidates that defeats Top-K's memory/computation
        benefit.  Anchors and ordinary raster inputs still follow Lightning's
        normal transfer, while the frozen precomputed tail tensors are moved
        selectively inside ``SAILImageMemoryFusion.forward``.
        """
        fusion = getattr(self.model, "cross_attn_fusion", None)
        lazy_global_topk = (
            getattr(fusion, "tail_aggregation", None) == "global_tail_topk"
        )
        if not lazy_global_topk or not isinstance(batch, (tuple, list)):
            return super().transfer_batch_to_device(batch, device, dataloader_idx)
        if len(batch) < 8:
            # All-tail prediction already retains paths rather than tensors.
            return super().transfer_batch_to_device(batch, device, dataloader_idx)

        anchors, anchor_exist, tails, tail_exist = batch[:4]
        image, dsm, label = batch[4:7]
        remainder = list(batch[7:])
        moved = [
            move_data_to_device(anchors, device),
            move_data_to_device(anchor_exist, device),
            tails,       # deliberately CPU resident
            tail_exist,  # moved by the fusion layer only when needed
            move_data_to_device(image, device),
            move_data_to_device(dsm, device),
            move_data_to_device(label, device),
            *remainder,
        ]
        return tuple(moved) if isinstance(batch, tuple) else moved

    @staticmethod
    def _param_grad_norm(module, suffix):
        total = None
        for name, param in module.named_parameters():
            if not name.endswith(suffix) or param.grad is None:
                continue
            value = param.grad.detach().float().norm()
            total = value if total is None else total + value.pow(2)
        if total is None:
            return None
        return float(total.sqrt().cpu())

    def _param_relative_delta(self, module, suffix):
        total_delta = None
        total_base = None
        for name, param in module.named_parameters():
            if not name.endswith(suffix):
                continue
            base = self._cross_attn_initial_params.get(name)
            if base is None:
                continue
            current = param.detach().float().cpu()
            delta = (current - base).norm()
            base_norm = base.norm().clamp_min(1e-12)
            total_delta = delta.pow(2) if total_delta is None else total_delta + delta.pow(2)
            total_base = base_norm.pow(2) if total_base is None else total_base + base_norm.pow(2)
        if total_delta is None or total_base is None:
            return None
        return float((total_delta.sqrt() / total_base.sqrt()).cpu())

    def on_train_start(self):
        if not self.cross_attn_debug_enabled:
            return
        module = getattr(self.model, "cross_attn_fusion", None)
        if module is None:
            return
        self._cross_attn_initial_params = {
            name: param.detach().float().cpu().clone()
            for name, param in module.named_parameters()
            if any(
                name.endswith(suffix)
                for suffix in (
                    "q_proj.weight",
                    "k_proj.weight",
                    "sampling_offsets.weight",
                    "sampling_offsets.bias",
                )
            )
        }

    def on_before_optimizer_step(self, optimizer):
        if not self.cross_attn_debug_enabled:
            return
        every = max(1, self.cross_attn_debug_every_n_steps)
        if int(self.global_step) % every != 0:
            return
        module = getattr(self.model, "cross_attn_fusion", None)
        if module is None:
            return

        q_grad = self._param_grad_norm(module, "q_proj.weight")
        k_grad = self._param_grad_norm(module, "k_proj.weight")
        off_w_grad = self._param_grad_norm(module, "sampling_offsets.weight")
        off_b_grad = self._param_grad_norm(module, "sampling_offsets.bias")
        q_delta = self._param_relative_delta(module, "q_proj.weight")
        k_delta = self._param_relative_delta(module, "k_proj.weight")
        off_w_delta = self._param_relative_delta(
            module,
            "sampling_offsets.weight",
        )
        off_b_delta = self._param_relative_delta(
            module,
            "sampling_offsets.bias",
        )

        stats = []
        for block in getattr(module, "blocks", []):
            block_stats = getattr(block, "last_debug_stats", None)
            if block_stats:
                stats.append(block_stats)
        if stats:
            entropy = sum(s["entropy"] for s in stats) / len(stats)
            max_attn = sum(s["max_attn"] for s in stats) / len(stats)
            offset_mean = sum(s["offset_mean"] for s in stats) / len(stats)
            valid_ratio = sum(s["valid_ratio"] for s in stats) / len(stats)
        else:
            entropy = max_attn = offset_mean = valid_ratio = 0.0

        def fmt(value):
            return "None" if value is None else f"{value:.3e}"

        print(
            "[CrossAttn-Debug] "
            f"step={int(self.global_step)} | "
            f"Grad(q/k/offW/offB)="
            f"{fmt(q_grad)}/{fmt(k_grad)}/{fmt(off_w_grad)}/{fmt(off_b_grad)} | "
            f"Delta(q/k/offW/offB)="
            f"{fmt(q_delta)}/{fmt(k_delta)}/{fmt(off_w_delta)}/{fmt(off_b_delta)} | "
            f"Entropy={entropy:.3f} | MaxAttn={max_attn:.3f} | "
            f"Offset={offset_mean:.3f}px | ValidRatio={valid_ratio:.3f}"
        )

    def configure_metric(self):
        metrics_cfg = getattr(self.hparams, 'metric', None)
        if metrics_cfg is None:
            return
        metrics = generate_metric(metrics_cfg)
        self.train_metrics = metrics.clone(prefix='train_')
        self.val_metrics = metrics.clone(prefix='val_')
        self.img_predict_metrics = metrics.clone()
        self.xyz_predict_metrics = metrics.clone()

    @staticmethod
    def _unpack_fusion_batch(batch):
        """Accept both legacy camera batches and the new SAIL role-separated batch."""
        # All-tail SAIL prediction: anchors, anchor_mask, list[list[path]],
        # image, dsm, label, names.  Raw tail tensors are intentionally loaded
        # later in predict_step in bounded chunks.
        if (
            len(batch) == 7
            and isinstance(batch[0], list)
            and torch.is_tensor(batch[1])
            and isinstance(batch[2], list)
            and torch.is_tensor(batch[3])
        ):
            anchors, anchor_exist, all_tail_paths = batch[:3]
            image, dsm, label, names = batch[3:]
            return {
                "cam_features": anchors,
                "cam_exist": anchor_exist,
                "anchor_features": anchors,
                "anchor_exist": anchor_exist,
                "tail_features": None,
                "tail_exist": None,
                "all_tail_paths": all_tail_paths,
                "image": image,
                "dsm": dsm,
                "label": label,
                "names": names,
            }
        # SAIL: anchors, anchor_mask, tails, tail_mask, image, dsm, label, names
        if (
            len(batch) >= 8
            and isinstance(batch[0], list)
            and torch.is_tensor(batch[1])
            and isinstance(batch[2], list)
            and torch.is_tensor(batch[3])
        ):
            anchors, anchor_exist, tails, tail_exist = batch[:4]
            image, dsm, label = batch[4:7]
            names = batch[7]
            return {
                "cam_features": anchors,
                "cam_exist": anchor_exist,
                "anchor_features": anchors,
                "anchor_exist": anchor_exist,
                "tail_features": tails,
                "tail_exist": tail_exist,
                "all_tail_paths": None,
                "image": image,
                "dsm": dsm,
                "label": label,
                "names": names,
            }

        # Legacy: camera features, camera mask, image, dsm, label[, names]
        cam_features, cam_exist, image, dsm, label = batch[:5]
        return {
            "cam_features": cam_features,
            "cam_exist": cam_exist,
            "anchor_features": None,
            "anchor_exist": None,
            "tail_features": None,
            "tail_exist": None,
            "all_tail_paths": None,
            "image": image,
            "dsm": dsm,
            "label": label,
            "names": batch[5] if len(batch) > 5 else None,
        }

    @staticmethod
    def _model_fusion_kwargs(data):
        return {
            "cam_features": data["cam_features"],
            "cam_exist": data["cam_exist"],
            "anchor_features": data["anchor_features"],
            "anchor_exist": data["anchor_exist"],
            "tail_features": data["tail_features"],
            "tail_exist": data["tail_exist"],
        }

    @staticmethod
    def _load_tail_feature(path):
        try:
            data = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            data = torch.load(path, map_location="cpu")
        scales = ("res2", "res3", "res4", "res5")
        if not isinstance(data, dict) or any(scale not in data for scale in scales):
            raise ValueError(f"Invalid all-tail feature file: {path}")
        return {scale: data[scale] for scale in scales}

    def _iter_all_tail_feature_chunks(self, all_tail_paths):
        if len(all_tail_paths) != 1:
            raise ValueError("All-tail SAIL inference currently requires batch_size=1")
        paths = all_tail_paths[0]
        chunk_size = int(getattr(self.hparams, "sail_all_tail_chunk_size", 32))
        if chunk_size < 1:
            raise ValueError("sail_all_tail_chunk_size must be >= 1")
        for start in range(0, len(paths), chunk_size):
            chunk_paths = paths[start:start + chunk_size]
            features = [self._load_tail_feature(path) for path in chunk_paths]
            exists = torch.ones((1, len(features)), dtype=torch.bool)
            yield features, exists

    def training_step(self, batch, batch_idx):
        data = self._unpack_fusion_batch(batch)
        image, dsm, label = data["image"], data["dsm"], data["label"]
        fusion_kwargs = self._model_fusion_kwargs(data)

        need_fusion_aux = self.use_view_supervision or self.log_swin_r_stats

        tail_off_pred = None
        if self.use_tail_off_supervision:
            outputs = self.model(
                x=(image, dsm), return_tail_off_pred=True,
                return_fusion_aux=need_fusion_aux, **fusion_kwargs,
            )
            pred, tail_off_pred = outputs[:2]
            fusion_aux = outputs[2] if need_fusion_aux else None
            tdom_pred = None
        elif self.use_background_consistency and need_fusion_aux:
            pred, tdom_pred, fusion_aux = self.model(
                x=(image, dsm),
                return_tdom_pred=True,
                return_fusion_aux=True,
                **fusion_kwargs,
            )
        elif self.use_background_consistency:
            pred, tdom_pred = self.model(
                x=(image, dsm),
                return_tdom_pred=True,
                **fusion_kwargs,
            )
            fusion_aux = None
        elif need_fusion_aux:
            pred, fusion_aux = self.model(
                x=(image, dsm),
                return_fusion_aux=True,
                **fusion_kwargs,
            )
            tdom_pred = None
        else:
            pred = self.model(
                x=(image, dsm),
                **fusion_kwargs,
            )
            tdom_pred = None
            fusion_aux = None

        segmentation_loss, _ = self.loss(pred, label)
        loss = segmentation_loss
        if tail_off_pred is not None:
            tail_off_loss, _ = self.loss(tail_off_pred, label)
            loss = self._combine_tail_on_off_loss(segmentation_loss, tail_off_loss)
            self.log("train_tail_on_loss", segmentation_loss, on_step=True, on_epoch=True)
            self.log("train_tail_off_loss", tail_off_loss, on_step=True, on_epoch=True)
            self.tail_off_train_metrics.update(self._probabilities(tail_off_pred), label)
            self.log_dict(self.tail_off_train_metrics, on_step=True, on_epoch=True, prog_bar=False)

        if fusion_aux is not None and self.use_view_supervision:
            evidence_losses = []
            gate_means = []
            background_probs = []
            foreground_probs = []

            target = label
            if target.ndim == pred.ndim - 1:
                target = target.unsqueeze(1)
            target = target.float()

            for scale_aux in fusion_aux.values():
                # SAIL memory aux is a mapping of per-scale tensors/lists, not
                # candidate-gate evidence dictionaries.
                if not isinstance(scale_aux, dict):
                    continue
                evidence_logits = scale_aux.get("evidence_logits")
                evidence_valid = scale_aux.get("evidence_valid")
                evidence_gate = scale_aux.get("evidence_gate")
                if evidence_logits is None or evidence_valid is None:
                    continue

                scale_target = F.interpolate(
                    target,
                    size=evidence_logits.shape[-2:],
                    mode="nearest",
                )
                scale_target = scale_target.expand(
                    -1, evidence_logits.shape[1], -1, -1
                )
                background_mask = (
                    (scale_target < 0.5) & evidence_valid
                )
                foreground_mask = (
                    (scale_target >= 0.5) & evidence_valid
                )

                if background_mask.any():
                    evidence_losses.append(
                        F.softplus(evidence_logits[background_mask]).mean()
                    )
                    background_probs.append(
                        torch.sigmoid(
                            evidence_logits[background_mask].detach()
                        ).mean()
                    )
                if foreground_mask.any():
                    foreground_probs.append(
                        torch.sigmoid(
                            evidence_logits[foreground_mask].detach()
                        ).mean()
                    )
                if evidence_gate is not None and evidence_valid.any():
                    gate_means.append(
                        evidence_gate[evidence_valid].detach().mean()
                    )

            if evidence_losses:
                view_evidence_loss = torch.stack(evidence_losses).mean()
            else:
                view_evidence_loss = pred.sum() * 0.0
            weighted_view_loss = (
                self.view_supervision_weight * view_evidence_loss
            )
            loss = loss + weighted_view_loss

            self.log(
                "train_view_evidence_loss",
                view_evidence_loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
            self.log(
                "train_view_evidence_weighted_loss",
                weighted_view_loss,
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                logger=True,
            )
            if gate_means:
                self.log(
                    "train_evidence_gate_mean",
                    torch.stack(gate_means).mean(),
                    on_step=True,
                    on_epoch=True,
                    prog_bar=False,
                    logger=True,
                )
            if background_probs:
                self.log(
                    "train_bg_evidence_prob",
                    torch.stack(background_probs).mean(),
                    on_step=True,
                    on_epoch=True,
                    prog_bar=False,
                    logger=True,
                )
            if foreground_probs:
                self.log(
                    "train_fg_evidence_prob",
                    torch.stack(foreground_probs).mean(),
                    on_step=True,
                    on_epoch=True,
                    prog_bar=False,
                    logger=True,
                )

        if fusion_aux is not None:
            self._log_swin_r_aux(fusion_aux, "train")

        if tdom_pred is not None:
            target = label
            if target.ndim == pred.ndim - 1:
                target = target.unsqueeze(1)
            target = target.to(device=pred.device)

            with torch.no_grad():
                tdom_prob = torch.sigmoid(tdom_pred)
                stable_background = (
                    (target < 0.5)
                    & (tdom_prob < self.tdom_background_threshold)
                )

            fused_prob = torch.sigmoid(pred)
            probability_increase = F.relu(fused_prob - tdom_prob)
            stable_background_float = stable_background.to(pred.dtype)
            stable_background_count = stable_background_float.sum()
            background_consistency_loss = (
                probability_increase * stable_background_float
            ).sum() / stable_background_count.clamp_min(1.0)
            weighted_background_loss = (
                self.background_consistency_weight
                * background_consistency_loss
            )

            loss = loss + weighted_background_loss
            self.log(
                "train_bg_consistency_loss",
                background_consistency_loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
            self.log(
                "train_bg_weighted_loss",
                weighted_background_loss,
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                logger=True,
            )
            self.log(
                "train_stable_bg_ratio",
                stable_background_float.mean(),
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                logger=True,
            )
            with torch.no_grad():
                violation_ratio = (
                    (fused_prob > tdom_prob).to(pred.dtype)
                    * stable_background_float
                ).sum() / stable_background_count.clamp_min(1.0)
            self.log(
                "train_bg_violation_ratio",
                violation_ratio,
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                logger=True,
            )

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        out_class = pred.shape[1]
        if out_class > 1:
            pred_prob = torch.softmax(pred, dim=1)
        else:
            pred_prob = torch.sigmoid(pred)

        metrics = self.train_metrics(pred_prob, label)
        self.log_dict(
            metrics,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True
        )

        return loss

    @staticmethod
    def _probabilities(pred):
        return torch.softmax(pred, dim=1) if pred.shape[1] > 1 else torch.sigmoid(pred)

    def _combine_tail_on_off_loss(self, on_loss, off_loss):
        combined = on_loss + self.tail_off_loss_weight * off_loss
        if self.normalize_tail_off_loss:
            combined = combined / (1.0 + self.tail_off_loss_weight)
        return combined

    def _log_swin_r_aux(self, fusion_aux, prefix):
        """Log the scalar Swin-R controls without changing the loss."""
        if not isinstance(fusion_aux, dict):
            return
        for key, name in (
            ("swin_r_alpha", "swin_r_alpha"),
            ("swin_r_delta_ratio", "swin_r_delta_ratio"),
            ("latent_pairwise_cos", "latent_pairwise_cos"),
            ("latent_weight_entropy", "latent_weight_entropy"),
            ("latent_view_weight_mean", "latent_view_weight_mean"),
            ("tail_local_context_alpha", "tail_local_context_alpha"),
            ("tail_local_context_delta_ratio", "tail_local_context_delta_ratio"),
            ("tail_dual_memory_context_alpha", "tail_dual_memory_context_alpha"),
            ("tail_dual_memory_context_delta_ratio", "tail_dual_memory_context_delta_ratio"),
            ("tail_logit_gamma", "tail_logit_gamma"),
            ("tail_logit_abs_mean", "tail_logit_abs_mean"),
            ("tail_logit_eligible_fraction", "tail_logit_eligible_fraction"),
        ):
            values = fusion_aux.get(key)
            if not values:
                continue
            if key == "latent_view_weight_mean":
                tensors = [v.float() for v in values if torch.is_tensor(v) and v.ndim == 2]
                if tensors:
                    weights = torch.stack(tensors).mean(dim=0)
                    for latent_id in range(weights.shape[0]):
                        for view_id in range(weights.shape[1]):
                            self.log(
                                f"{prefix}_latent_w_l{latent_id}_v{view_id}",
                                weights[latent_id, view_id],
                                on_step=False,
                                on_epoch=True,
                                prog_bar=False,
                                logger=True,
                            )
                continue
            tensors = [v.float().reshape(()) for v in values if torch.is_tensor(v)]
            if tensors:
                self.log(
                    f"{prefix}_{name}",
                    torch.stack(tensors).mean(),
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    logger=True,
                )

    def validation_step(self, batch, batch_idx):
        data = self._unpack_fusion_batch(batch)
        image, dsm, label = data["image"], data["dsm"], data["label"]

        tail_off_pred = None
        if self.use_tail_off_supervision:
            outputs = self.model(
                x=(image, dsm), return_tail_off_pred=True,
                return_fusion_aux=self.log_swin_r_stats,
                **self._model_fusion_kwargs(data),
            )
            pred, tail_off_pred = outputs[:2]
            if self.log_swin_r_stats:
                self._log_swin_r_aux(outputs[2], "val")
        elif self.log_swin_r_stats:
            pred, fusion_aux = self.model(
                x=(image, dsm),
                return_fusion_aux=True,
                **self._model_fusion_kwargs(data),
            )
            self._log_swin_r_aux(fusion_aux, "val")
        else:
            pred = self.model(
                x=(image, dsm),
                **self._model_fusion_kwargs(data),
            )

        loss, _ = self.loss(pred, label)
        if tail_off_pred is not None:
            off_loss, _ = self.loss(tail_off_pred, label)
            self.log("val_tail_off_loss", off_loss, on_step=False, on_epoch=True)
            self.log("val_dual_loss", self._combine_tail_on_off_loss(loss, off_loss), on_step=False, on_epoch=True)
            self.tail_off_val_metrics.update(self._probabilities(tail_off_pred), label)
            self.log_dict(self.tail_off_val_metrics, on_step=False, on_epoch=True, prog_bar=False)

        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True
        )

        out_class = pred.shape[1]
        if out_class > 1:
            pred_prob = torch.softmax(pred, dim=1)
        else:
            pred_prob = torch.sigmoid(pred)

        metrics = self.val_metrics(pred_prob, label)
        self.log_dict(
            metrics,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True
        )

        return metrics

    def predict_step(self, batch, batch_idx):
        data = self._unpack_fusion_batch(batch)
        image, dsm, label, xyz_name = (
            data["image"], data["dsm"], data["label"], data["names"]
        )
        if xyz_name is None:
            raise ValueError("predict_step requires sample names from the inference collate function")
        fusion_kwargs = self._model_fusion_kwargs(data)
        all_tail_paths = data.get("all_tail_paths")
        
        feature_save_dir = getattr(self.hparams, 'feature_save_dir', None)
        predict_tdom_only = bool(
            getattr(self.hparams, 'predict_tdom_only', False)
        )

        if predict_tdom_only:
            pred = self.model(
                x=(image, dsm),
                cam_features=None,
                cam_exist=None,
                output_attn=False,
            )
            attns = None
        elif all_tail_paths is not None:
            pred, attns = self.model(
                x=(image, dsm),
                output_attn=True,
                names=xyz_name,
                save_dir=feature_save_dir,
                tail_feature_chunks=self._iter_all_tail_feature_chunks(all_tail_paths),
                **fusion_kwargs,
            )
        else:
            pred, attns = self.model(
                x=(image, dsm),
                output_attn=True,
                names=xyz_name,
                save_dir=feature_save_dir,
                **fusion_kwargs,
            )

        if feature_save_dir is not None and attns is not None:
            self.visualize_attention(attns, xyz_name, feature_save_dir, image.shape[-2:])
        
        out_class = pred.shape[1]
        if out_class > 1:
            pred = torch.softmax(pred, dim = 1)
        else:
            pred = torch.sigmoid(pred)

        _, xoff, yoff = xyz_name[0].split('_')
        xoff, yoff = int(xoff), int(yoff)
        H, W = image.shape[-2], image.shape[-1]
        
        return {
            "pred": pred.detach().cpu(),
            "offset": (xoff, yoff),
            "patch_size": (H, W),
            "xyz_name": xyz_name[0]
        }

    def visualize_attention(self, attns, names, save_dir, target_size):
        """
        Save camera attention and cosine maps as float32 tif files.
        """
        if not attns:
            return

        num_cameras = getattr(self.model.cross_attn_fusion, 'num_cameras', 5)

        attn_save_path = os.path.join(save_dir, 'attention_tif')
        cosine_save_path = os.path.join(save_dir, 'cosine_tif')
        score_save_path = os.path.join(save_dir, 'score_tif')
        os.makedirs(attn_save_path, exist_ok=True)
        os.makedirs(cosine_save_path, exist_ok=True)
        os.makedirs(score_save_path, exist_ok=True)

        for scale_key in ["res2", "res3", "res4", "res5"]:
            if scale_key not in attns:
                continue

            scale_data = attns[scale_key]
            attn = scale_data['attn'] if isinstance(scale_data, dict) else scale_data
            cosine = (
                scale_data.get('cosine')
                if isinstance(scale_data, dict)
                else None
            )
            score = (
                scale_data.get('score')
                if isinstance(scale_data, dict)
                else None
            )

            B, Nq, heads, Nkv = attn.shape
            H = W = int(np.sqrt(Nq))

            for b in range(B):
                name = names[b]

                avg_attn = attn[b].mean(dim=1)
                if avg_attn.shape[-1] != num_cameras:
                    avg_attn = avg_attn.view(Nq, num_cameras, -1).sum(dim=-1)

                cam_attn = avg_attn.reshape(H, W, num_cameras).cpu().numpy()
                if target_size is not None and (H != target_size[0] or W != target_size[1]):
                    cam_attn = np.stack(
                        [
                            cv2.resize(
                                cam_attn[:, :, c],
                                (target_size[1], target_size[0]),
                                interpolation=cv2.INTER_LINEAR,
                            )
                            for c in range(num_cameras)
                        ],
                        axis=-1,
                    )

                for c in range(num_cameras):
                    save_fn = f"{name}_{scale_key}_cam{c}.tif"
                    tifffile.imwrite(
                        os.path.join(attn_save_path, save_fn),
                        cam_attn[:, :, c].astype(np.float32),
                    )

                if cosine is not None:
                    # [B, Nq, heads, cameras] -> [H, W, cameras]
                    cam_cosine = cosine[b].mean(dim=1).reshape(
                        H, W, num_cameras
                    ).cpu().numpy()
                    if target_size is not None and (
                        H != target_size[0] or W != target_size[1]
                    ):
                        cam_cosine = np.stack(
                            [
                                cv2.resize(
                                    cam_cosine[:, :, c],
                                    (target_size[1], target_size[0]),
                                    interpolation=cv2.INTER_LINEAR,
                                )
                                for c in range(num_cameras)
                            ],
                            axis=-1,
                        )

                    for c in range(num_cameras):
                        save_fn = f"{name}_{scale_key}_cam{c}.tif"
                        tifffile.imwrite(
                            os.path.join(cosine_save_path, save_fn),
                            cam_cosine[:, :, c].astype(np.float32),
                        )

                if score is not None:
                    cam_score = score[b].mean(dim=1).reshape(
                        H, W, num_cameras
                    ).cpu().numpy()
                    if target_size is not None and (
                        H != target_size[0] or W != target_size[1]
                    ):
                        cam_score = np.stack(
                            [
                                cv2.resize(
                                    cam_score[:, :, c],
                                    (target_size[1], target_size[0]),
                                    interpolation=cv2.INTER_LINEAR,
                                )
                                for c in range(num_cameras)
                            ],
                            axis=-1,
                        )

                    for c in range(num_cameras):
                        save_fn = f"{name}_{scale_key}_cam{c}.tif"
                        tifffile.imwrite(
                            os.path.join(score_save_path, save_fn),
                            cam_score[:, :, c].astype(np.float32),
                        )
