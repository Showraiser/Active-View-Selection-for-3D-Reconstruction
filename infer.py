"""
infer.py — Final inference for RL View Selection + 3D Reconstruction
─────────────────────────────────────────────────────────────────────
Uses a trained ViewPolicy checkpoint (ckpt_final.pt) to:
  1. Load a single 3D object (a folder of 24 PNG views + vox_64.npy)
  2. Run the RL policy to select the best N views within a given budget
  3. Feed the chosen views through the frozen EncoderDecoder reconstructor
  4. Report the selected views, reconstruction IoU, and optionally save
     the predicted voxel grid.

Usage
─────
  python infer.py --object-dir path/to/chair_0001 --budget 5
  python infer.py --object-dir path/to/chair_0001 --budget 3 --save-voxels out.npy
  python infer.py --object-dir path/to/chair_0001 --budget 8 --device cuda

Directory layout expected
─────────────────────────
  object-dir/
    images/
      000.png … 023.png   (24 RGB renders, white background)
    vox_64.npy             (64³ ground-truth occupancy — optional, used for IoU)
    cameras.json           (optional; used by depth/silhouette helpers)
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tv_models
import torchvision.transforms as T
from PIL import Image


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="RL View Selection Inference")
    p.add_argument("--object-dir",       required=True,
                   help="Path to the object folder (must contain images/ subdir)")
    p.add_argument("--budget",           type=int, default=5, choices=[3, 5, 8],
                   help="View budget: how many views the policy may select (3, 5, or 8)")
    p.add_argument("--policy-ckpt",      default="checkpoints/ckpt_final.pt",
                   help="Path to the RL policy checkpoint (ckpt_final.pt)")
    p.add_argument("--recon-ckpt",       default=None,
                   help="Path to the EncoderDecoder reconstructor checkpoint (.pth). "
                        "If omitted, no reconstruction is run.")
    p.add_argument("--save-voxels",      default=None,
                   help="If given, save the predicted voxel grid to this .npy path")
    p.add_argument("--iou-threshold",    type=float, default=0.4,
                   help="Threshold for binarising predicted voxels (default 0.4)")
    p.add_argument("--gt-threshold",     type=float, default=0.5,
                   help="Threshold for binarising ground-truth voxels (default 0.5)")
    p.add_argument("--device",           default="cpu",
                   help="Device for the RL policy (cpu or cuda). "
                        "Reconstructor always runs on CPU.")
    p.add_argument("--n-views",          type=int, default=24,
                   help="Number of candidate views per object (default 24)")
    p.add_argument("--recon-img-size",   type=int, default=256,
                   help="Image resolution expected by the reconstructor (default 256)")
    p.add_argument("--feature-dim",      type=int, default=512)
    p.add_argument("--voxel-size",       type=int, default=64)
    p.add_argument("--greedy",           action="store_true",
                   help="Use greedy (argmax) action selection instead of sampling")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Inline copies of model classes (so infer.py works standalone)
# ─────────────────────────────────────────────────────────────────────────────

# ── State builders ────────────────────────────────────────────────────────────

class CoverageGrid:
    def __init__(self):
        self.grid = np.zeros((32, 32, 32), dtype=np.float32)

    def reset(self):
        self.grid = np.zeros((32, 32, 32), dtype=np.float32)

    def update(self, depth_map: np.ndarray, silhouette: np.ndarray):
        depth_resized = np.array(
            Image.fromarray(depth_map).resize((32, 32), Image.BILINEAR)
        )
        sil_resized = np.array(
            Image.fromarray(silhouette.astype(np.float32)).resize((32, 32), Image.BILINEAR)
        )
        z_indices = np.clip((depth_resized * 31).astype(int), 0, 31)
        mask = sil_resized > 0.5
        xs, ys = np.where(mask)
        for x, y in zip(xs, ys):
            self.grid[x, y, z_indices[x, y]] = 1.0

    def get(self) -> np.ndarray:
        return self.grid.copy()


class ImageFeatureExtractor:
    def __init__(self):
        resnet = tv_models.resnet50(weights=tv_models.ResNet50_Weights.DEFAULT)
        self.model = torch.nn.Sequential(*list(resnet.children())[:-1])
        self.model.eval()
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])
        self.features = []

    def reset(self):
        self.features = []

    def add_view(self, image: Image.Image):
        tensor = self.transform(image).unsqueeze(0)
        with torch.no_grad():
            feat = self.model(tensor).squeeze().numpy()   # (2048,)
        self.features.append(feat.reshape(4, 512).mean(axis=0))

    def get(self) -> np.ndarray:
        if not self.features:
            return np.zeros(512, dtype=np.float32)
        return np.mean(self.features, axis=0).astype(np.float32)


class ViewHistoryMask:
    def __init__(self, n_views: int = 24):
        self.n_views = n_views
        self.mask = np.zeros(n_views, dtype=np.float32)

    def reset(self):
        self.mask = np.zeros(self.n_views, dtype=np.float32)

    def mark(self, idx: int):
        self.mask[idx] = 1.0

    def get(self) -> np.ndarray:
        return self.mask.copy()

    def visited(self) -> set:
        return set(np.where(self.mask == 1.0)[0].tolist())


# ── RL Policy ─────────────────────────────────────────────────────────────────

class CoverageGridEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=4, stride=2, padding=1), nn.ReLU(True),
            nn.Conv3d(16, 32, kernel_size=4, stride=2, padding=1), nn.ReLU(True),
            nn.Conv3d(32, 64, kernel_size=4, stride=2, padding=1), nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4 * 4, 256), nn.ReLU(True),
        )

    def forward(self, x):
        return self.net(x.unsqueeze(1))   # (B, 32, 32, 32) → (B, 256)


class ViewPolicy(nn.Module):
    def __init__(self, n_views: int = 24):
        super().__init__()
        self.n_views = n_views
        self.coverage_encoder = CoverageGridEncoder()
        self.feature_encoder  = nn.Sequential(nn.Linear(512, 256), nn.ReLU(True))
        self.history_encoder  = nn.Sequential(nn.Linear(n_views, 64), nn.ReLU(True))
        self.shared    = nn.Sequential(nn.Linear(576, 512), nn.ReLU(True))
        self.view_head = nn.Sequential(nn.Linear(512, 256), nn.ReLU(True), nn.Linear(256, n_views))
        self.value_head= nn.Sequential(nn.Linear(512, 128), nn.ReLU(True), nn.Linear(128, 1))

    def _encode(self, obs: dict) -> torch.Tensor:
        cov  = self.coverage_encoder(obs["coverage_grid"])
        feat = self.feature_encoder(obs["image_features"])
        hist = self.history_encoder(obs["view_mask"])
        return self.shared(torch.cat([cov, feat, hist], dim=-1))

    def forward(self, obs, action_mask=None):
        latent = self._encode(obs)
        logits = self.view_head(latent)
        values = self.value_head(latent).squeeze(-1)
        if action_mask is not None:
            logits = logits.masked_fill(action_mask == 0, float("-inf"))
        return logits, values

    @torch.no_grad()
    def select_view(self, obs: dict, action_mask: torch.Tensor,
                    greedy: bool = False) -> int:
        logits, _ = self.forward(obs, action_mask)
        if greedy:
            return int(logits.argmax(dim=-1).item())
        dist = torch.distributions.Categorical(logits=logits)
        return int(dist.sample().item())


# ── Reconstructor ─────────────────────────────────────────────────────────────

import torch.nn.functional as F
import torchvision.models as models


class ResBlock3D(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(ch, ch, 3, padding=1), nn.InstanceNorm3d(ch), nn.ReLU(True),
            nn.Conv3d(ch, ch, 3, padding=1), nn.InstanceNorm3d(ch),
        )

    def forward(self, x):
        return F.relu(x + self.net(x))


class ViewFusion(nn.Module):
    def __init__(self, n_views, feat_dim):
        super().__init__()
        self.attn = nn.Linear(feat_dim, 1)

    def forward(self, feats):
        w = torch.softmax(self.attn(feats).squeeze(-1), dim=1).unsqueeze(-1)
        return (feats * w).sum(dim=1)


class Encoder(nn.Module):
    def __init__(self, feature_dim=512, img_size=256, n_views=24):
        super().__init__()
        effnet = models.efficientnet_v2_s(
            weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
        )
        self.backbone = effnet.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        with torch.no_grad():
            dummy = torch.zeros(1, 3, img_size, img_size)
            c = self.backbone(dummy).shape[1]
        self.fc = nn.Linear(c, feature_dim)

    def forward_single(self, x):
        return self.fc(torch.flatten(self.pool(self.backbone(x)), 1))

    def forward(self, views):
        return self.forward_single(views)


class Decoder(nn.Module):
    def __init__(self, feature_dim=512, voxel_size=64, start_channels=256):
        super().__init__()
        self.start_size = 4
        self.start_ch   = start_channels
        self.fc = nn.Linear(feature_dim, start_channels * 64)
        self.up_blocks = nn.ModuleList([
            nn.Sequential(nn.ConvTranspose3d(256, 128, 4, 2, 1), nn.InstanceNorm3d(128), nn.ReLU(True)),
            nn.Sequential(nn.ConvTranspose3d(128,  64, 4, 2, 1), nn.InstanceNorm3d( 64), nn.ReLU(True)),
            nn.Sequential(nn.ConvTranspose3d( 64,  32, 4, 2, 1), nn.InstanceNorm3d( 32), nn.ReLU(True)),
            nn.Sequential(nn.ConvTranspose3d( 32,  16, 4, 2, 1), nn.InstanceNorm3d( 16), nn.ReLU(True)),
        ])
        self.res_blocks = nn.ModuleList([ResBlock3D(c) for c in [128, 64, 32, 16]])
        self.final_conv = nn.Sequential(
            nn.Conv3d(16, 16, 3, padding=1), nn.InstanceNorm3d(16), nn.ReLU(True),
            nn.Conv3d(16, 1, 1),
        )

    def forward(self, x):
        B = x.shape[0]
        x = self.fc(x).view(B, self.start_ch, 4, 4, 4)
        for up, res in zip(self.up_blocks, self.res_blocks):
            x = res(up(x))
        return self.final_conv(x)


class EncoderDecoder(nn.Module):
    def __init__(self, feature_dim=512, voxel_size=64, img_size=256, n_views=24):
        super().__init__()
        self.encoder    = Encoder(feature_dim, img_size, n_views)
        self.view_fuser = ViewFusion(n_views, feature_dim)
        self.decoder    = Decoder(feature_dim, voxel_size)

    def forward(self, views):
        B, V, C, H, W = views.shape
        feats = self.encoder(views.view(B * V, C, H, W)).view(B, V, -1)
        return self.decoder(self.view_fuser(feats))


# ─────────────────────────────────────────────────────────────────────────────
# Data loading helpers
# ─────────────────────────────────────────────────────────────────────────────

def _silhouette(img: Image.Image) -> np.ndarray:
    rgb = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
    return (~np.all(rgb >= (250 / 255.0), axis=-1)).astype(np.float32)


def _load_object(object_dir: str, n_views: int = 24,
                 img_size: int = 256) -> dict:
    """Load all views, silhouettes, and voxels from an object folder."""
    p = Path(object_dir)
    img_dir = p / "images"
    if not img_dir.exists():
        raise FileNotFoundError(f"No images/ subfolder found in {p}")

    images, silhouettes, depths = [], [], []
    for v in range(n_views):
        fpath = img_dir / f"{v:03d}.png"
        if not fpath.exists():
            raise FileNotFoundError(f"Missing view image: {fpath}")
        img = Image.open(str(fpath)).convert("RGB")
        if img_size:
            img = img.resize((img_size, img_size), Image.BILINEAR)
        images.append(img)
        silhouettes.append(_silhouette(img))
        depths.append(np.zeros((32, 32), dtype=np.float32))

    voxels = None
    vox_path = p / "vox_64.npy"
    if vox_path.exists():
        voxels = np.load(str(vox_path)).astype(np.float32)
        if voxels.ndim == 4:
            voxels = voxels[0]
        voxels = np.clip(voxels, 0.0, 1.0)

    return {
        "images":      images,
        "silhouettes": silhouettes,
        "depths":      depths,
        "voxels":      voxels,
        "model_id":    p.name,
        "category":    p.parent.name,
    }


def _obs_to_tensor(obs: dict, device: str) -> dict:
    return {k: torch.FloatTensor(v).unsqueeze(0).to(device)
            for k, v in obs.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Main inference routine
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(args):
    print("\n" + "═" * 60)
    print("  RL View Selection — Inference")
    print("═" * 60)

    # ── 1. Load object ────────────────────────────────────────────────────────
    print(f"\n[1/4] Loading object from: {args.object_dir}")
    obj = _load_object(args.object_dir, n_views=args.n_views,
                       img_size=args.recon_img_size)
    print(f"      Category : {obj['category']}  |  Model : {obj['model_id']}")
    print(f"      GT voxels: {'✓ found' if obj['voxels'] is not None else '✗ not found (IoU skipped)'}")

    # ── 2. Load RL policy ─────────────────────────────────────────────────────
    print(f"\n[2/4] Loading RL policy from: {args.policy_ckpt}")
    if not os.path.isfile(args.policy_ckpt):
        raise FileNotFoundError(
            f"Policy checkpoint not found: {args.policy_ckpt}\n"
            "Pass --policy-ckpt path/to/ckpt_final.pt"
        )
    policy = ViewPolicy(n_views=args.n_views).to(args.device)
    state  = torch.load(args.policy_ckpt, map_location="cpu")
    policy.load_state_dict(state.get("policy", state))
    policy.eval()
    total_eps = state.get("total_episodes", "?")
    print(f"      Loaded checkpoint from episode {total_eps}")

    # ── 3. Run RL episode — view selection ────────────────────────────────────
    print(f"\n[3/4] Running policy (budget={args.budget}, "
          f"{'greedy' if args.greedy else 'sampled'})...")

    cov_grid  = CoverageGrid()
    feat_ext  = ImageFeatureExtractor()
    view_mask = ViewHistoryMask(args.n_views)

    selected = []
    selection_log = []

    for step in range(args.budget):
        obs = {
            "coverage_grid":  cov_grid.get(),
            "image_features": feat_ext.get(),
            "view_mask":      view_mask.get(),
        }
        obs_t = _obs_to_tensor(obs, args.device)

        # Action mask: 1 = available, 0 = already visited
        avail = np.ones(args.n_views, dtype=np.float32)
        for v in view_mask.visited():
            avail[v] = 0.0
        mask_t = torch.FloatTensor(avail).unsqueeze(0).to(args.device)

        # Policy selects next view
        chosen = policy.select_view(obs_t, mask_t, greedy=args.greedy)
        selected.append(chosen)

        # Get softmax probabilities for logging
        with torch.no_grad():
            logits, value = policy(obs_t, mask_t)
            probs = torch.softmax(logits, dim=-1).squeeze()
            conf  = float(probs[chosen].item())
            val   = float(value.item())

        selection_log.append({
            "step": step + 1, "view": chosen,
            "confidence": conf, "value_est": val,
        })

        # Update RL state
        cov_grid.update(obj["depths"][chosen], obj["silhouettes"][chosen])
        feat_ext.add_view(obj["images"][chosen])
        view_mask.mark(chosen)

        print(f"      Step {step+1}/{args.budget}: view {chosen:2d}  "
              f"conf={conf:.3f}  V̂={val:.4f}")

    print(f"\n      Selected views: {selected}")

    # ── 4. Reconstruct ────────────────────────────────────────────────────────
    iou_score = None

    if args.recon_ckpt is not None:
        print(f"\n[4/4] Reconstructing with: {args.recon_ckpt}")
        if not os.path.isfile(args.recon_ckpt):
            print(f"      [WARNING] Reconstructor checkpoint not found — skipping reconstruction.")
        else:
            recon_transform = T.Compose([
                T.Resize((args.recon_img_size, args.recon_img_size)),
                T.ToTensor(),
            ])

            # Load reconstructor
            reconstructor = EncoderDecoder(
                feature_dim=args.feature_dim,
                voxel_size=args.voxel_size,
                img_size=args.recon_img_size,
                n_views=args.n_views,
            )
            ckpt = torch.load(args.recon_ckpt, map_location="cpu")
            reconstructor.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
            reconstructor.eval()
            for p in reconstructor.parameters():
                p.requires_grad_(False)

            # Build input tensor (1, V, 3, H, W)
            selected_imgs = [obj["images"][v] for v in selected]
            views_t = torch.stack([recon_transform(im) for im in selected_imgs]).unsqueeze(0)

            with torch.no_grad():
                raw = torch.sigmoid(reconstructor(views_t))    # (1, 1, 64, 64, 64)

            pred_vox = (raw.squeeze().numpy() > args.iou_threshold).astype(np.float32)
            print(f"      Predicted occupancy: {pred_vox.mean():.4f} "
                  f"({int(pred_vox.sum())} voxels)")

            # IoU vs ground truth
            if obj["voxels"] is not None:
                gt_vox = (obj["voxels"] > args.gt_threshold).astype(np.float32)
                inter  = float((pred_vox * gt_vox).sum())
                union  = float(np.clip(pred_vox + gt_vox, 0, 1).sum())
                iou_score = inter / union if union > 0 else 0.0
                print(f"      IoU vs ground truth : {iou_score:.4f}")

            # Save voxels
            if args.save_voxels:
                np.save(args.save_voxels, raw.squeeze().numpy())
                print(f"      Saved predicted voxels → {args.save_voxels}")
    else:
        print(f"\n[4/4] Reconstruction skipped (no --recon-ckpt provided)")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("  SUMMARY")
    print("─" * 60)
    print(f"  Object      : {obj['category']} / {obj['model_id']}")
    print(f"  Budget      : {args.budget} views")
    print(f"  Selected    : {selected}")
    for entry in selection_log:
        print(f"    step {entry['step']}: view {entry['view']:2d}  "
              f"conf={entry['confidence']:.3f}  V̂={entry['value_est']:.4f}")
    if iou_score is not None:
        rating = ("excellent" if iou_score > 0.65 else
                  "good"      if iou_score > 0.45 else
                  "fair"      if iou_score > 0.30 else "poor")
        print(f"  IoU         : {iou_score:.4f}  ({rating})")
    print("─" * 60 + "\n")

    return {
        "selected_views": selected,
        "selection_log":  selection_log,
        "iou":            iou_score,
        "object":         f"{obj['category']}/{obj['model_id']}",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Batch helper (optional programmatic use)
# ─────────────────────────────────────────────────────────────────────────────

def batch_infer(object_dirs: list, budget: int, policy_ckpt: str,
                recon_ckpt: str = None, device: str = "cpu",
                greedy: bool = False, n_views: int = 24,
                recon_img_size: int = 256) -> list:
    """
    Run inference on a list of object directories.

    Returns a list of result dicts (one per object), each containing:
      selected_views, selection_log, iou, object.
    """
    import types
    args = types.SimpleNamespace(
        budget=budget,
        policy_ckpt=policy_ckpt,
        recon_ckpt=recon_ckpt,
        device=device,
        greedy=greedy,
        n_views=n_views,
        recon_img_size=recon_img_size,
        feature_dim=512,
        voxel_size=64,
        iou_threshold=0.4,
        gt_threshold=0.5,
        save_voxels=None,
    )
    results = []
    for obj_dir in object_dirs:
        args.object_dir = obj_dir
        try:
            results.append(run_inference(args))
        except Exception as e:
            print(f"[ERROR] {obj_dir}: {e}")
            results.append({"object": obj_dir, "error": str(e)})
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    result = run_inference(args)
