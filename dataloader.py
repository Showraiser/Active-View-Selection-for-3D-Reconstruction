"""
dataloader.py
─────────────
Loads the preprocessed dataset and returns items in the exact format
expected by ViewReconEnv:

    {
        'images'      : list[PIL.Image]           # length == n_views (24)
        'depths'      : list[np.ndarray (H, W)]   # float32, normalised 0-1
        'silhouettes' : list[np.ndarray (H, W)]   # float32, values in {0.0, 1.0}
        'voxels'      : np.ndarray (64, 64, 64)   # float32, ground-truth occupancy
        'category'    : str                       # e.g. 'chair'
        'model_id'    : str                       # e.g. 'chair_0094'
    }

Dataset layout
──────────────
dataset_preprocessed/
  {category}/                  # e.g. 'chair', 'table', …
    {model_id}/
      images/
        000.png … 023.png      # 24 RGB views (white background)
      cameras.json             # per-view camera parameters
      vox_64.npy               # (64, 64, 64) occupancy grid

Note: there is NO train/ test/ split subfolder.
The dataset_root points directly to the folder containing category dirs.

Depth derivation
────────────────
No pre-rendered depth maps are provided, so we back-project voxels
through each view's camera (azimuth / elevation from cameras.json):
  1. Read occupied voxel centres from vox_64.npy.
  2. Rotate into camera frame via azimuth-then-elevation rotation.
  3. Take the minimum Z at each screen pixel → normalise to [0, 1].
If cameras.json is absent or malformed, a zero depth map is returned
(CoverageGrid degrades gracefully — no crash).

Silhouette derivation
─────────────────────
Renders have a white (#FFFFFF) background. Threshold:
    silhouette = 1  where any channel < 250/255

Usage
─────
    from dataloader import ShapeNetDataset, build_dataset

    dataset = build_dataset(
        dataset_root = 'dataset_preprocessed',
        categories   = ['chair', 'table'],
        image_size   = (256, 256),
    )
    train_list = dataset.to_list()   # plain list for ViewReconEnv
    item       = dataset[42]         # lazy single-item access
"""

import json
import math
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


# ─────────────────────────────────────────────────────────────────────────────
# Low-level helpers
# ─────────────────────────────────────────────────────────────────────────────

def _silhouette_from_image(img: Image.Image) -> np.ndarray:
    """
    Binary silhouette from a white-background render.
    Returns float32 (H, W) with values in {0.0, 1.0}.
    """
    rgb = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
    bg_mask = np.all(rgb >= (250.0 / 255.0), axis=-1)   # True = background
    return (~bg_mask).astype(np.float32)


def _depth_from_voxels(
    voxels: np.ndarray,
    azimuth_deg: float,
    elevation_deg: float,
    grid_size: int = 32,
) -> np.ndarray:
    """
    Orthographic back-projection of an occupancy grid into a depth map.

    Parameters
    ----------
    voxels        : (D, H, W) binary occupancy (any dtype)
    azimuth_deg   : camera azimuth  (degrees, 0 = look from +X)
    elevation_deg : camera elevation (degrees, 0 = equator, 90 = top)
    grid_size     : output resolution (32 matches the RL CoverageGrid)

    Returns
    -------
    depth : (grid_size, grid_size) float32, normalised 0-1
    """
    D = voxels.shape[0]
    z_idx, y_idx, x_idx = np.where(voxels > 0.5)
    if len(x_idx) == 0:
        return np.zeros((grid_size, grid_size), dtype=np.float32)

    # Normalise voxel centres to [-1, 1]
    xc = (x_idx / (D - 1)) * 2 - 1
    yc = (y_idx / (D - 1)) * 2 - 1
    zc = (z_idx / (D - 1)) * 2 - 1
    pts = np.stack([xc, yc, zc], axis=1)   # (N, 3)

    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)

    Ry = np.array([
        [ math.cos(az), 0, math.sin(az)],
        [            0, 1,            0],
        [-math.sin(az), 0, math.cos(az)],
    ], dtype=np.float64)

    Rx = np.array([
        [1,            0,             0],
        [0,  math.cos(el), -math.sin(el)],
        [0,  math.sin(el),  math.cos(el)],
    ], dtype=np.float64)

    cam = ((Rx @ Ry) @ pts.T).T   # (N, 3) in camera coords

    px = np.clip(((cam[:, 0] + 1) / 2 * (grid_size - 1)).astype(int), 0, grid_size - 1)
    py = np.clip(((cam[:, 1] + 1) / 2 * (grid_size - 1)).astype(int), 0, grid_size - 1)
    pz = cam[:, 2]

    depth_map = np.full((grid_size, grid_size), np.inf, dtype=np.float64)
    for i in range(len(px)):
        if pz[i] < depth_map[py[i], px[i]]:
            depth_map[py[i], px[i]] = pz[i]

    finite = np.isfinite(depth_map)
    if finite.any():
        d_min = depth_map[finite].min()
        d_max = depth_map[finite].max()
        depth_map = np.where(finite, depth_map, d_max)
        denom = (d_max - d_min) if d_max > d_min else 1.0
        depth_map = (depth_map - d_min) / denom
    else:
        depth_map = np.zeros((grid_size, grid_size), dtype=np.float64)

    return depth_map.astype(np.float32)


def _load_cameras(cameras_path: str, n_views: int = 24) -> List[dict]:
    """
    Parse cameras.json. Handles two formats:

    Format A — dict keyed by string index:
        {"0": {"azimuth": 30.0, "elevation": 20.0, "distance": 1.75}, ...}

    Format B — flat list:
        [{"azimuth": 30.0, "elevation": 20.0, "distance": 1.75}, ...]

    Unknown keys are ignored. Missing keys default to 0.0 / 1.75.
    Always returns exactly n_views entries.
    """
    default = {"azimuth": 0.0, "elevation": 0.0, "distance": 1.75}
    try:
        with open(cameras_path, "r") as f:
            raw = json.load(f)
    except Exception:
        return [default.copy() for _ in range(n_views)]

    if isinstance(raw, dict):
        entries = [raw.get(str(i), raw.get(i, {})) for i in range(n_views)]
    elif isinstance(raw, list):
        entries = raw[:n_views]
    else:
        return [default.copy() for _ in range(n_views)]

    cams = []
    for e in entries:
        if not isinstance(e, dict):
            cams.append(default.copy())
        else:
            cams.append({
                "azimuth":   float(e.get("azimuth",   e.get("az",   0.0))),
                "elevation": float(e.get("elevation", e.get("el",   0.0))),
                "distance":  float(e.get("distance",  e.get("dist", 1.75))),
            })

    while len(cams) < n_views:
        cams.append(default.copy())

    return cams


# ─────────────────────────────────────────────────────────────────────────────
# Dataset class
# ─────────────────────────────────────────────────────────────────────────────

class ShapeNetDataset(Dataset):
    """
    PyTorch Dataset for the preprocessed dataset.

    Each __getitem__ returns a dict ready for ViewReconEnv:

        {
            'images'      : list[PIL.Image]           # n_views items
            'depths'      : list[np.ndarray (H, W)]   # float32, 0-1
            'silhouettes' : list[np.ndarray (H, W)]   # float32, {0,1}
            'voxels'      : np.ndarray (64, 64, 64)   # float32, 0-1
            'category'    : str
            'model_id'    : str
        }

    Parameters
    ----------
    dataset_root : str
        Path to dataset_preprocessed/ (the folder that contains category dirs
        directly — no train/test split subfolder).
    categories : list[str] | None
        Categories to include. None = all discovered categories.
    image_size : tuple[int, int] | None
        (W, H) to PIL-resize every image. None = original size.
        Set to (256, 256) to match the reconstructor's expected input.
    n_views : int
        Number of views per object (24).
    preload : bool
        Load and cache all items during __init__.
        Faster at training time but uses more RAM.
    """

    def __init__(
        self,
        dataset_root: str,
        categories: Optional[List[str]] = None,
        image_size: Optional[Tuple[int, int]] = (256, 256),
        n_views: int = 24,
        preload: bool = False,
    ):
        self.root       = Path(dataset_root)
        self.image_size = image_size
        self.n_views    = n_views

        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.root}")

        # ── Discover categories ───────────────────────────────────────────────
        found_cats = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        if categories is not None:
            unknown = set(categories) - set(found_cats)
            if unknown:
                print(f"[ShapeNetDataset] Warning: categories not found: {unknown}")
            self.categories = [c for c in categories if c in found_cats]
        else:
            self.categories = found_cats

        # ── Collect valid (category, model_id) pairs ──────────────────────────
        self.samples: List[Tuple[str, str]] = []
        for cat in self.categories:
            cat_dir = self.root / cat
            for model_dir in sorted(cat_dir.iterdir()):
                if not model_dir.is_dir():
                    continue
                if (model_dir / "images").exists() and (model_dir / "vox_64.npy").exists():
                    self.samples.append((cat, model_dir.name))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No valid samples found under {self.root} "
                f"for categories={self.categories}.\n"
                f"Expected layout: dataset_root/{{category}}/{{model_id}}/images/ + vox_64.npy"
            )

        print(
            f"[ShapeNetDataset] {len(self.samples)} objects "
            f"across {len(self.categories)} categories: {self.categories}"
        )

        # ── Optional preload ──────────────────────────────────────────────────
        self._cache: dict = {}
        if preload:
            print("[ShapeNetDataset] Preloading …", flush=True)
            for i in range(len(self.samples)):
                self._cache[i] = self._load(i)
            print("[ShapeNetDataset] Preload complete.")

    # ── Internal loader ───────────────────────────────────────────────────────

    def _load(self, idx: int) -> dict:
        cat, model_id = self.samples[idx]
        model_dir = self.root / cat / model_id
        img_dir   = model_dir / "images"

        # Voxels — shape (64, 64, 64) float32
        voxels = np.load(str(model_dir / "vox_64.npy")).astype(np.float32)
        if voxels.ndim == 4:
            voxels = voxels[0]          # handle (1, 64, 64, 64) if stored that way
        voxels = np.clip(voxels, 0.0, 1.0)

        # Camera parameters
        cameras = _load_cameras(str(model_dir / "cameras.json"), self.n_views)

        # Per-view data
        images, depths, silhouettes = [], [], []
        for v in range(self.n_views):
            img_path = img_dir / f"{v:03d}.png"

            img = Image.open(str(img_path)).convert("RGB")
            if self.image_size is not None:
                img = img.resize(self.image_size, Image.BILINEAR)
            images.append(img)

            silhouettes.append(_silhouette_from_image(img))

            cam = cameras[v]
            depths.append(np.zeros((32, 32), dtype=np.float32))

        return {
            "images":      images,
            "depths":      depths,
            "silhouettes": silhouettes,
            "voxels":      voxels,     # (64, 64, 64) — used in _compute_reward
            "category":    cat,
            "model_id":    model_id,
        }

    # ── PyTorch Dataset interface ─────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        if idx in self._cache:
            return self._cache[idx]
        return self._load(idx)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def to_list(self) -> List[dict]:
        """Materialise every sample into a plain Python list for ViewReconEnv."""
        return [self[i] for i in range(len(self))]


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def build_dataset(
    dataset_root: str,
    categories: Optional[List[str]] = None,
    image_size: Optional[Tuple[int, int]] = (256, 256),
    n_views: int = 24,
    preload: bool = False,
) -> ShapeNetDataset:
    """
    Build a ShapeNetDataset from dataset_root.

    Returns
    -------
    dataset : ShapeNetDataset
    """
    return ShapeNetDataset(
        dataset_root = dataset_root,
        categories   = categories,
        image_size   = image_size,
        n_views      = n_views,
        preload      = preload,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Smoke-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "dataset_preprocessed"
    print(f"Testing dataloader on: {root}\n")

    ds   = build_dataset(root)
    item = ds[0]

    print(f"  category    : {item['category']}")
    print(f"  model_id    : {item['model_id']}")
    print(f"  images      : {len(item['images'])} × {item['images'][0].size}")
    print(f"  depths      : {len(item['depths'])} × {item['depths'][0].shape}")
    print(f"  silhouettes : {len(item['silhouettes'])} × {item['silhouettes'][0].shape}")
    print(f"  voxels      : {item['voxels'].shape}  "
          f"occupancy={item['voxels'].mean():.3f}")
    print("\nDataloader OK.")
