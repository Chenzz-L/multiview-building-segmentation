import os
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def vis_pca_feature(
    feat: torch.Tensor,
    title: str,
    save_dir: str = "./debug_pca",
    fname: str = None,
    max_hw: int = 128
):
    """
    feat: Tensor [B,C,H,W] or [C,H,W]
    """
    os.makedirs(save_dir, exist_ok=True)

    if feat.dim() == 4:
        feat = feat[0]  # 只看 batch=0

    C, H, W = feat.shape

    # 下采样，防止 H*W 太大 PCA 太慢
    if max(H, W) > max_hw:
        scale = max_hw / max(H, W)
        feat = torch.nn.functional.interpolate(
            feat.unsqueeze(0),
            scale_factor=scale,
            mode="bilinear",
            align_corners=False
        ).squeeze(0)
        C, H, W = feat.shape

    x = feat.reshape(C, -1).permute(1, 0).cpu().numpy()

    pca = PCA(n_components=3)
    x_pca = pca.fit_transform(x)

    # normalize to [0,1]
    x_pca = (x_pca - x_pca.min(0)) / (x_pca.max(0) - x_pca.min(0) + 1e-6)
    img = x_pca.reshape(H, W, 3)

    plt.figure(figsize=(5, 5))
    plt.imshow(img)
    plt.title(title)
    plt.axis("off")

    if fname is None:
        fname = title.replace(" ", "_").replace("/", "_") + ".png"

    save_path = os.path.join(save_dir, fname)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    return save_path
