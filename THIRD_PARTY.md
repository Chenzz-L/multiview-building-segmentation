# Third-party attribution

Built with DINOv3.

- Meta DINOv3: `models/backbones/dinov3/` and `vision_transformer_dinov3.py`. Original copyright headers retained. See [DINOv3 license](third_party/DINOv3-LICENSE.txt) and https://github.com/facebookresearch/dinov3.
- Mask2Former / Detectron2: decoder and pixel-decoder adaptations under `models/heads/`; retain their source notices. See [Mask2Former](third_party/Mask2Former-LICENSE.txt), [Detectron2](third_party/Detectron2-LICENSE.txt).
- OpenMMLab components: backbone/head utilities retain original source notices. See [MMSegmentation license](third_party/MMsegmentation-LICENSE.txt).

Bundled upstream files are research-tree snapshots, not claimed as original author contributions. Source hashes appear in `docs/source-manifest.json`. Path normalization and release tooling were added when preparing this repository. No blanket license is asserted over all components; upstream terms continue to apply to their respective code and weights.
