"""Dataset for SAIL-style anchor-memory multi-view fusion.

An item keeps the two view roles separate:

* anchors come from an offline K-medoid file (``clusters``), or, in opt-in
  name-index mode, are loaded from raw files using fixed anchor identities;
* tails are raw projected image features belonging to the same ground tile,
  excluding the exact paths recorded in ``selected_anchor_names``.

The raw directory is usually very large.  Build and reuse ``view_index_path``
with ``utils/build_sail_view_index.py`` before DDP training, otherwise every
DDP rank may need to scan the raw directory once.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch

from data.datasets.dataste_support import regist_dataset
from data.datasets.dataset_Base import BaseDataset
from data.transforms.augmentation import seg_augmentation


SCALES = ("res2", "res3", "res4", "res5")
FEATURE_SHAPES = {
    "res2": (256, 128, 128),
    "res3": (256, 64, 64),
    "res4": (256, 32, 32),
    "res5": (256, 16, 16),
}
_GEOMETRIC_AUGMENTATIONS = {
    "Flip", "HorizontalFlip", "VerticalFlip", "Transpose", "Rotate",
    "RandomRotate90", "RandomResizedCrop", "Resize", "Crop",
    "CenterCrop", "RandomCrop", "ShiftScaleRotate", "Affine",
    "Perspective", "ElasticTransform", "GridDistortion",
}


def sail_tile_key(value: str, xyz_num_length: int = 3) -> str:
    """Normalise raw and TDOM names to e.g. ``3_0360_6840``."""
    parts = Path(value).stem.split("_")[-xyz_num_length:]
    if len(parts) != xyz_num_length:
        raise ValueError(f"Cannot extract {xyz_num_length}-part tile key from {value!r}")
    if "-" in parts[0]:
        parts[0] = parts[0].rsplit("-", 1)[-1]
    return "_".join(parts)


def _load_torch(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # torch < 2.0
        return torch.load(path, map_location="cpu")


def _contains_geometric_augmentation(data_pipeline) -> bool:
    if not data_pipeline:
        return False
    for item in data_pipeline:
        if isinstance(item, dict) and item.get("type") in _GEOMETRIC_AUGMENTATIONS:
            return True
    return False


def build_sail_view_index(
    raw_feature_dir: str,
    tile_keys: Optional[Iterable[str]] = None,
    xyz_num_length: int = 3,
) -> Dict[str, List[str]]:
    """Return tile -> sorted raw paths relative to ``raw_feature_dir``."""
    root = Path(raw_feature_dir)
    wanted = set(tile_keys) if tile_keys is not None else None
    index: Dict[str, List[str]] = {}
    for path in root.rglob("*.pt"):
        key = sail_tile_key(path.name, xyz_num_length)
        if wanted is not None and key not in wanted:
            continue
        index.setdefault(key, []).append(path.relative_to(root).as_posix())
    for paths in index.values():
        paths.sort()
    return index


@regist_dataset()
class Dataset_DSM_Fusion_RGB_DSM_SAILMemory(BaseDataset):
    """Load medoid anchors plus randomly sampled non-anchor raw tails.

    ``tail_count`` is deliberately bounded for training/validation.  It is not
    the total number of available tails.  Exact all-tail inference will use the
    same index but pass tail files to the fusion layer in chunks.
    """

    def __init__(
        self,
        augmentation: bool,
        anchor_data_dir: Optional[str],
        raw_feat_data_dir: str,
        xyz_data_dir: str,
        indices_name: str = "indices",
        standardization: bool = False,
        xyz_dsm_none_value: float = -99999,
        clip=None,
        data_pipeline=None,
        num_anchors: int = 3,
        tail_count: int = 8,
        tail_count_choices: Optional[Sequence[int]] = None,
        tail_sampling: str = "random",
        tail_mean_data_dir: Optional[str] = None,
        view_index_path: Optional[str] = None,
        auto_build_index: bool = True,
        xyz_num_length: int = 3,
        strict: bool = True,
        missing_anchor_policy: str = "error",
        return_all_tail_paths: bool = False,
        disable_tails: bool = False,
        synchronized_d4_augmentation: bool = False,
        seed: int = 42,
        anchor_prefix: bool = False,
        anchor_name_index_path: Optional[str] = None,
        raw_feat_data_roots: Optional[Sequence[str]] = None,
    ):
        if _contains_geometric_augmentation(data_pipeline):
            raise ValueError(
                "SAIL projected features are pixel-aligned with TDOM. "
                "Do not use geometric image augmentation unless the same "
                "transform is applied to every FPN feature scale."
            )
        # BaseDataset may prepend ToFloat, so pass a private copy.
        super().__init__(
            xyz_data_dir, augmentation, standardization, clip,
            copy.deepcopy(data_pipeline), indices_name,
        )
        if num_anchors < 1 or tail_count < 1:
            raise ValueError("num_anchors and tail_count must both be >= 1")
        if tail_sampling not in {"random", "fixed", "first"}:
            raise ValueError("tail_sampling must be one of: random, fixed, first")
        if tail_count_choices is not None:
            if tail_sampling != "random":
                raise ValueError(
                    "tail_count_choices requires tail_sampling='random' so both "
                    "the tail count and tail identities vary during training"
                )
            choices = tuple(sorted({int(value) for value in tail_count_choices}))
            if not choices or choices[0] < 1:
                raise ValueError("tail_count_choices must contain positive integers")
        else:
            choices = None
        if missing_anchor_policy not in {"error", "drop", "tdom_only"}:
            raise ValueError(
                "missing_anchor_policy must be one of: error, drop, tdom_only"
            )

        self.anchor_name_index_path = Path(anchor_name_index_path) if anchor_name_index_path else None
        self.anchor_data_dir = Path(anchor_data_dir) if anchor_data_dir else None
        self.raw_feat_data_dir = Path(raw_feat_data_dir)
        if isinstance(raw_feat_data_roots, (str, bytes)):
            raise TypeError("raw_feat_data_roots must be a list of directories, not a string")
        self.raw_feat_data_roots = list(dict.fromkeys(
            [self.raw_feat_data_dir] + [Path(p) for p in (raw_feat_data_roots or [])]
        ))
        self.tail_mean_data_dir = (
            Path(tail_mean_data_dir) if tail_mean_data_dir else None
        )
        self.xyz_data_dir = Path(xyz_data_dir)
        self.num_anchors = int(num_anchors)
        # Opt-in K sensitivity: read only the first K anchors from a larger
        # greedy anchor library, and exclude only those K identities from tails.
        # The historical strict K-directory contract remains the default.
        self.anchor_prefix = bool(anchor_prefix)
        # ``tail_count`` remains the fixed padded width consumed by the current
        # collate/model path.  With variable-Q sampling, pad to the largest Q so
        # samples with different choices can still be stacked in one batch.
        self.tail_count_choices = choices
        self.tail_count = max(int(tail_count), max(choices) if choices else 0)
        self.tail_sampling = tail_sampling
        self.xyz_num_length = int(xyz_num_length)
        self.strict = bool(strict)
        self.missing_anchor_policy = missing_anchor_policy
        self.return_all_tail_paths = bool(return_all_tail_paths)
        self.synchronized_d4_augmentation = bool(
            synchronized_d4_augmentation
        )
        if self.synchronized_d4_augmentation and self.return_all_tail_paths:
            raise ValueError(
                "synchronized_d4_augmentation requires loaded tail tensors; "
                "it cannot be combined with return_all_tail_paths=True"
            )
        # Anchor-only ablation: keep a padded, invalid tail slot for the
        # existing collate/model interface, but never index or load raw tails.
        # This makes the downstream tail update exactly zero.
        self.disable_tails = bool(disable_tails)
        self.seed = int(seed)
        self.xyz_dsm_none_value = xyz_dsm_none_value
        self._image_dir_name = "image"
        self._label_dir_name = "label"
        self._dsm_dir_name = "dsm"

        self.anchor_name_index = None
        if self.anchor_name_index_path is not None:
            if self.anchor_data_dir is not None:
                raise ValueError("Set anchor_data_dir: null when using anchor_name_index_path")
            self.anchor_name_index = self._load_anchor_name_index()
            if not any(p.is_dir() for p in self.raw_feat_data_roots):
                raise FileNotFoundError(f"No raw feature root exists: {self.raw_feat_data_roots}")
        elif self.anchor_data_dir is None or not self.anchor_data_dir.is_dir():
            raise FileNotFoundError(f"anchor_data_dir does not exist: {self.anchor_data_dir}")
        if self.tail_mean_data_dir is not None:
            if not self.tail_mean_data_dir.is_dir():
                raise FileNotFoundError(
                    f"tail_mean_data_dir does not exist: {self.tail_mean_data_dir}"
                )
            if self.return_all_tail_paths:
                raise ValueError(
                    "tail_mean_data_dir already represents all tails and cannot "
                    "be combined with return_all_tail_paths=True"
                )
            if self.tail_count != 1 or self.tail_count_choices is not None:
                raise ValueError(
                    "tail_mean_data_dir requires tail_count=1 and no "
                    "tail_count_choices"
                )
        elif self.anchor_name_index is None and not self.disable_tails and not self.raw_feat_data_dir.is_dir():
            raise FileNotFoundError(f"raw_feat_data_dir does not exist: {self.raw_feat_data_dir}")

        self.tile_names = [os.path.splitext(os.path.basename(name))[0] for name in self.ids]
        self.tile_keys = [sail_tile_key(name, self.xyz_num_length) for name in self.tile_names]
        if self.anchor_name_index is not None:
            unknown = [key for key in self.tile_keys if key not in self.anchor_name_index]
            if unknown:
                raise ValueError(
                    f"Anchor name index is incomplete: {len(unknown)} unknown requested tiles; "
                    f"examples: {unknown[:3]}. Record [] only for confirmed no-anchor tiles; "
                    "missing_anchor_policy must not silently drop unresolved identities."
                )
        self.raw_index = (
            {}
            if (self.disable_tails or self.tail_mean_data_dir is not None) and self.anchor_name_index is None
            else self._load_or_build_index(
                view_index_path=view_index_path,
                auto_build_index=auto_build_index,
            )
        )

        missing_anchors = [name for name in self.tile_names if not (
            self.anchor_name_index.get(sail_tile_key(name, self.xyz_num_length))
            if self.anchor_name_index is not None else self._anchor_path(name).is_file()
        )]
        if missing_anchors:
            message = (
                f"{len(missing_anchors)} requested tiles have no anchors in "
                f"{self.anchor_name_index_path or self.anchor_data_dir}; examples: {missing_anchors[:3]}"
            )
            if self.missing_anchor_policy == "error":
                raise FileNotFoundError(message)
            if self.missing_anchor_policy == "drop":
                keep = [i for i, name in enumerate(self.tile_names) if name not in set(missing_anchors)]
                self.ids = [self.ids[i] for i in keep]
                self.tile_names = [self.tile_names[i] for i in keep]
                self.tile_keys = [self.tile_keys[i] for i in keep]
                print(f"[SAILDataset] dropping tiles without anchors: {message}")
            else:
                print(f"[SAILDataset] tdom_only fallback for tiles without anchors: {message}")

    def _anchor_path(self, name: str) -> Path:
        if self.anchor_data_dir is None:
            raise RuntimeError("Name-index mode does not use an anchor .pt directory")
        return self.anchor_data_dir / f"{name}.pt"

    def _load_anchor_name_index(self) -> Dict[str, List[str]]:
        with self.anchor_name_index_path.open('r', encoding='utf-8-sig') as f:
            payload = json.load(f)
        if not isinstance(payload, dict) or not isinstance(payload.get('anchors'), dict):
            raise ValueError("Anchor name index must contain an 'anchors' mapping: tile -> ordered name list")
        result = {}
        for key, names in payload['anchors'].items():
            tile = sail_tile_key(str(key), self.xyz_num_length)
            if not isinstance(names, list) or any(not isinstance(n, str) or not n.strip() for n in names):
                raise ValueError(f"Invalid anchor name list for {key!r}")
            canonical = []
            for name in names:
                basename = name.replace('\\', '/').rsplit('/', 1)[-1]
                if not basename.endswith('.pt') or sail_tile_key(basename, self.xyz_num_length) != tile:
                    raise ValueError(f"Anchor {name!r} does not belong to tile {tile}")
                canonical.append(f"{tile.split('_')[0]}/{basename}")
            if len(set(canonical)) != len(canonical):
                raise ValueError(f"Duplicate anchor identities for {tile}")
            if len(canonical) > self.num_anchors and not self.anchor_prefix:
                raise ValueError(f"{tile}: {len(canonical)} names exceed num_anchors={self.num_anchors}; no implicit prefix")
            if tile in result:
                raise ValueError(f"Duplicate normalised tile key: {tile}")
            result[tile] = canonical[:self.num_anchors]
        return result

    def _resolve_raw_path(self, value: str) -> Path:
        """Resolve paths only; never select a replacement feature identity.

        Raw-index entries are authoritative when present. Explicit absolute
        entries are NOT silently redirected to another root if missing.
        Multiple roots may be used for relative region/basename entries; the
        first existing path wins, without scanning a large feature library.
        """
        path = Path(value)
        if path.is_absolute():
            return path
        for root in self.raw_feat_data_roots:
            candidate = root / path
            if candidate.is_file():
                return candidate
        return self.raw_feat_data_dir / path

    def _load_anchors_from_names(self, name: str):
        tile = sail_tile_key(name, self.xyz_num_length)
        names = self.anchor_name_index[tile]
        if not names:
            if self.missing_anchor_policy == 'error':
                raise FileNotFoundError(f"No anchor names for {tile} in {self.anchor_name_index_path}")
            return [self._zeros(torch.float16) for _ in range(self.num_anchors)], [False] * self.num_anchors, []
        # An index can route a moved view to an absolute cache path. Retain the
        # identity and refuse ambiguous routes, rather than reading stale data.
        indexed = {}
        for value in self.raw_index.get(tile, []):
            basename = str(value).replace('\\', '/').rsplit('/', 1)[-1]
            if basename in indexed and indexed[basename] != value:
                raise ValueError(f"Ambiguous raw-index paths for {tile}/{basename}")
            indexed[basename] = value
        output, dtype = [], torch.float16
        for selected in names:
            basename = selected.rsplit('/', 1)[-1]
            path = self._resolve_raw_path(indexed.get(basename, selected))
            try:
                data = _load_torch(path)
                if not isinstance(data, dict) or any(s not in data for s in SCALES):
                    raise ValueError('missing raw FPN scale')
                view = {s: data[s] for s in SCALES}
                if any(not isinstance(v, torch.Tensor) or v.ndim != 3 for v in view.values()):
                    raise ValueError('raw FPN features must be tensors [C,H,W]')
                dtype = view['res2'].dtype
                output.append(view)
            except Exception as exc:
                # Missing identities are NOT missing manifest entries. Even
                # drop/tdom_only must not hide a broken recorded anchor path.
                raise RuntimeError(f"Failed to load named anchor {selected} for {tile}: {path}") from exc
        exists = [True] * len(output)
        while len(output) < self.num_anchors:
            output.append(self._zeros(dtype))
            exists.append(False)
        return output, exists, names

    def _load_or_build_index(
        self,
        view_index_path: Optional[str],
        auto_build_index: bool,
    ) -> Dict[str, List[str]]:
        wanted = set(self.tile_keys)
        if view_index_path:
            path = Path(view_index_path)
            if path.is_file():
                with path.open("r", encoding="utf-8") as f:
                    payload = json.load(f)
                stored = payload.get("views", payload)
                result = {
                    # Keep fixed sampling tied to view identities, not the
                    # storage root: moving only some files to a cache must not
                    # reorder the candidate set and change seeded Q8 choices.
                    key: sorted(paths, key=(
                        (lambda p: str(p).replace('\\', '/').rsplit('/', 1)[-1])
                        if self.anchor_name_index is not None else None
                    ))
                    for key, paths in stored.items()
                    if key in wanted
                }
                if self.anchor_name_index is not None:
                    for key, paths in result.items():
                        names = [str(p).replace('\\', '/').rsplit('/', 1)[-1] for p in paths]
                        if len(set(names)) != len(names):
                            raise ValueError(f"Duplicate raw view identities in index for {key}")
                return result
            if not auto_build_index:
                raise FileNotFoundError(
                    f"view_index_path is missing: {path}. Build it with "
                    "utils/build_sail_view_index.py first."
                )

        if not auto_build_index:
            raise ValueError("Set view_index_path or allow auto_build_index")
        print(
            f"[SAILDataset] building raw view index for {len(wanted)} tiles from "
            f"{self.raw_feat_data_dir}; prebuilding it once is recommended for DDP."
        )
        result = build_sail_view_index(
            str(self.raw_feat_data_dir), wanted, self.xyz_num_length
        )
        if view_index_path:
            path = Path(view_index_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "format": "sail_view_index_v1",
                "raw_feature_dir": str(self.raw_feat_data_dir),
                "xyz_num_length": self.xyz_num_length,
                "views": result,
            }
            # Atomic replacement makes concurrent DDP first-runs harmless.
            with tempfile.NamedTemporaryFile(
                "w", delete=False, dir=path.parent, encoding="utf-8"
            ) as f:
                json.dump(payload, f, ensure_ascii=False)
                temporary_path = f.name
            os.replace(temporary_path, path)
        return result

    @staticmethod
    def _zeros(dtype: torch.dtype) -> Dict[str, torch.Tensor]:
        return {
            scale: torch.zeros(shape, dtype=dtype)
            for scale, shape in FEATURE_SHAPES.items()
        }

    def _load_anchors(self, name: str):
        if self.anchor_name_index is not None:
            return self._load_anchors_from_names(name)
        path = self._anchor_path(name)
        if not path.is_file():
            if self.missing_anchor_policy == "error":
                raise FileNotFoundError(f"Missing anchor file: {path}")
            return [self._zeros(torch.float16) for _ in range(self.num_anchors)], [False] * self.num_anchors, []

        data = _load_torch(path)
        if not isinstance(data, dict) or "clusters" not in data:
            raise ValueError(f"Anchor file does not have a clusters dict: {path}")
        clusters = data["clusters"]
        if any(scale not in clusters for scale in SCALES):
            raise KeyError(f"Anchor file is missing one of {SCALES}: {path}")
        available = int(clusters["res2"].shape[0])
        cluster_valid = data.get("cluster_valid", torch.ones(available, dtype=torch.bool))
        selected_names = [str(value).replace("\\", "/") for value in data.get("selected_anchor_names", [])]
        if available > self.num_anchors and not self.anchor_prefix:
            raise ValueError(
                f"{path} has K={available}, but num_anchors={self.num_anchors}. "
                "Point anchor_data_dir at the intended kK directory."
            )

        output, exists = [], []
        dtype = clusters["res2"].dtype
        for anchor_id in range(self.num_anchors):
            is_valid = anchor_id < available and bool(cluster_valid[anchor_id].item())
            if is_valid:
                view = {scale: clusters[scale][anchor_id] for scale in SCALES}
            else:
                view = self._zeros(dtype)
            output.append(view)
            exists.append(is_valid)
        if self.strict and len(selected_names) < sum(exists):
            raise ValueError(
                f"{path} has {sum(exists)} valid anchors but only "
                f"{len(selected_names)} selected_anchor_names; tails could not "
                "be safely de-duplicated."
            )
        if self.anchor_prefix:
            selected_names = selected_names[:min(available, self.num_anchors)]
        return output, exists, selected_names

    def _select_tail_paths(self, tile_key: str, selected_anchor_names: Sequence[str]):
        prefix = getattr(self, "anchor_prefix", False) or getattr(self, "anchor_name_index", None) is not None
        identity = (lambda value: str(value).replace("\\", "/").rsplit("/", 1)[-1]) if prefix else (lambda value: value)
        anchor_set = {identity(name) for name in selected_anchor_names}
        raw_paths = self.raw_index.get(tile_key, [])
        if not raw_paths:
            if self.strict:
                raise RuntimeError(
                    f"No raw views indexed for {tile_key}. Verify raw_feat_data_dir "
                    "and view_index_path."
                )
            return []
        candidates = [
            relative_path for relative_path in raw_paths
            if identity(relative_path) not in anchor_set
        ]
        if not candidates:
            # Valid boundary case: e.g. a tile has exactly one usable raw view,
            # which becomes its sole medoid anchor.  It contributes anchor memory
            # only; _load_tails pads all tail slots and tail_exist is false.
            return []
        if self.return_all_tail_paths:
            # Used only by bounded-memory all-tail inference.  Feature tensors
            # stay on disk here and are loaded in small chunks by predict_step.
            return candidates
        selected_count = self.tail_count
        if self.tail_count_choices is not None:
            # Lightning seeds Python's random state independently in each
            # worker/rank.  The state advances over epochs, so both Q and the
            # selected tail identities change whenever a tile is revisited.
            selected_count = random.choice(self.tail_count_choices)
        if self.tail_sampling == "first":
            return candidates[:selected_count]
        if self.tail_sampling == "fixed":
            digest = hashlib.blake2b(
                f"{self.seed}:{tile_key}".encode("utf-8"), digest_size=8
            ).digest()
            rng = random.Random(int.from_bytes(digest, "little"))
            return rng.sample(candidates, min(selected_count, len(candidates)))
        # Lightning seeds every worker.  Random state then advances over epochs,
        # allowing one tile to expose different tails over training.
        return random.sample(candidates, min(selected_count, len(candidates)))

    def _load_tails(self, selected_paths: Sequence[str]):
        tails, exists = [], []
        dtype = torch.float16
        for relative_path in selected_paths:
            path = (self._resolve_raw_path(relative_path)
                    if self.anchor_name_index is not None else self.raw_feat_data_dir / relative_path)
            try:
                data = _load_torch(path)
                if not isinstance(data, dict) or any(scale not in data for scale in SCALES):
                    raise ValueError("missing FPN scale")
                view = {scale: data[scale] for scale in SCALES}
                dtype = view["res2"].dtype
                tails.append(view)
                exists.append(True)
            except Exception as exc:
                if self.strict:
                    raise RuntimeError(f"Failed to load tail feature: {path}") from exc
                tails.append(self._zeros(dtype))
                exists.append(False)
        while len(tails) < self.tail_count:
            tails.append(self._zeros(dtype))
            exists.append(False)
        return tails, exists

    def _load_tail_mean(self, name: str):
        """Load one pre-averaged non-anchor tail feature as a single view.

        ``_num_views`` records how many raw non-anchor views formed the mean.
        A file with zero remaining views is represented by one padded zero slot
        whose ``tail_exist`` flag is false, preserving the existing collate
        contract without inventing image evidence.
        """
        if self.tail_mean_data_dir is None:
            raise RuntimeError("tail_mean_data_dir is not configured")
        path = self.tail_mean_data_dir / f"{name}.pt"
        if not path.is_file():
            if self.strict:
                raise FileNotFoundError(f"Missing tail-mean file: {path}")
            return [self._zeros(torch.float16)], [False]

        try:
            data = _load_torch(path)
            if not isinstance(data, dict) or any(scale not in data for scale in SCALES):
                raise ValueError("missing FPN scale")
            num_views = int(data.get("_num_views", 1))
            if num_views < 0:
                raise ValueError("_num_views must be non-negative")
            if num_views == 0:
                dtype = data["res2"].dtype
                return [self._zeros(dtype)], [False]
            view = {scale: data[scale] for scale in SCALES}
            return [view], [True]
        except Exception as exc:
            if self.strict:
                raise RuntimeError(f"Failed to load tail mean: {path}") from exc
            return [self._zeros(torch.float16)], [False]

    @staticmethod
    def _d4_numpy(array: np.ndarray, rotation: int, reflect: bool) -> np.ndarray:
        """Apply one exact D4 transform to a channel-first raster."""
        output = np.rot90(array, k=rotation, axes=(-2, -1))
        if reflect:
            output = np.flip(output, axis=-1)
        # rot90/flip may return negative-stride views, which torch.from_numpy
        # cannot consume.  Contiguous storage also makes collating predictable.
        return np.ascontiguousarray(output)

    @staticmethod
    def _d4_tensor(
        tensor: torch.Tensor,
        rotation: int,
        reflect: bool,
    ) -> torch.Tensor:
        """Apply the same exact D4 transform to an FPN feature map."""
        output = torch.rot90(tensor, k=rotation, dims=(-2, -1))
        if reflect:
            output = torch.flip(output, dims=(-1,))
        return output.contiguous()

    @classmethod
    def _d4_views(
        cls,
        views: Sequence[Dict[str, torch.Tensor]],
        rotation: int,
        reflect: bool,
    ) -> List[Dict[str, torch.Tensor]]:
        return [
            {
                scale: cls._d4_tensor(feature, rotation, reflect)
                for scale, feature in view.items()
            }
            for view in views
        ]

    def __len__(self):
        return len(self.tile_names)

    def __getitem__(self, idx):
        name = self.tile_names[idx]
        tile_key = self.tile_keys[idx]
        label_path = self.xyz_data_dir / self._label_dir_name / f"{name}.tif"
        image_path = self.xyz_data_dir / self._image_dir_name / f"{name}.tif"
        dsm_path = self.xyz_data_dir / self._dsm_dir_name / f"{name}.tif"

        xyz_label = self.open(str(label_path))
        xyz_image = self.open(str(image_path), channel_sum=3)
        xyz_dsm = self.open(str(dsm_path), none_value=self.xyz_dsm_none_value)
        if self.augmentation and self.data_pipeline is not None:
            xyz_image, xyz_label, xyz_dsm = seg_augmentation(
                image=xyz_image, mask=[xyz_label, xyz_dsm], aug=self.data_pipeline
            )

        anchors, anchor_exist, selected_anchor_names = self._load_anchors(name)
        if self.disable_tails or not any(anchor_exist):
            # Without R, the SAIL layer deliberately ignores tails.  Avoid I/O
            # and make the requested tdom_only/anchor-only fallback exact.
            tail_paths = []
            if not self.return_all_tail_paths:
                tails, tail_exist = self._load_tails([])
        elif self.tail_mean_data_dir is not None:
            tail_paths = []
            tails, tail_exist = self._load_tail_mean(name)
        else:
            tail_paths = self._select_tail_paths(tile_key, selected_anchor_names)
            if not self.return_all_tail_paths:
                tails, tail_exist = self._load_tails(tail_paths)

        if self.synchronized_d4_augmentation:
            # Uniformly sample the eight elements of the square's dihedral
            # group: four rotations, with or without one reflection.  The
            # exact same transform is applied to every aligned modality and
            # every FPN scale, so no resampling or correspondence error is
            # introduced.
            transform_id = random.randrange(8)
            rotation = transform_id % 4
            reflect = transform_id >= 4
            xyz_image = self._d4_numpy(xyz_image, rotation, reflect)
            xyz_label = self._d4_numpy(xyz_label, rotation, reflect)
            xyz_dsm = self._d4_numpy(xyz_dsm, rotation, reflect)
            anchors = self._d4_views(anchors, rotation, reflect)
            tails = self._d4_views(tails, rotation, reflect)

        if self.return_all_tail_paths:
            # Preserve the list as absolute paths; the DataLoader never loads
            # all views at once and predict_step streams these paths in chunks.
            absolute_tail_paths = [str(self._resolve_raw_path(path) if self.anchor_name_index is not None
                                       else self.raw_feat_data_dir / path) for path in tail_paths]
            return (
                anchors,
                torch.tensor(anchor_exist, dtype=torch.bool),
                absolute_tail_paths,
                torch.from_numpy(xyz_image).float(),
                torch.from_numpy(xyz_dsm).float(),
                torch.from_numpy(xyz_label).long(),
                name,
            )

        return (
            anchors,
            torch.tensor(anchor_exist, dtype=torch.bool),
            tails,
            torch.tensor(tail_exist, dtype=torch.bool),
            torch.from_numpy(xyz_image).float(),
            torch.from_numpy(xyz_dsm).float(),
            torch.from_numpy(xyz_label).long(),
            name,
        )
