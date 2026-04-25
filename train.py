"""
train.py
────────
Entry point for RL training.

Usage
─────
  python train.py                          # fresh run
  python train.py --resume                 # auto-load latest checkpoint
  python train.py --dummy-vec              # sequential envs (no multiprocessing)
  python train.py --smoke-test             # 3 episodes per phase, quick sanity check
  python train.py --n-envs 4              # override Config.n_envs
  python train.py --dataset-root /path    # override Config.dataset_root

Environment variable overrides (useful on Kaggle / Colab):
  DATASET_ROOT=/path/to/data python train.py

Important: use --dummy-vec if you have GPU memory issues with SubprocVecEnv.
The reconstructor inside each env worker always runs on CPU.
"""

import argparse
import os
import glob
import torch

# ── CLI arguments ─────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--resume",       action="store_true",
                    help="Continue from the latest RL checkpoint.")
parser.add_argument("--dummy-vec",    action="store_true",
                    help="Use DummyVecEnv (sequential) instead of SubprocVecEnv.")
parser.add_argument("--smoke-test",   action="store_true",
                    help="Run 3 episodes per phase for a quick sanity check.")
parser.add_argument("--n-envs",       type=int, default=None,
                    help="Override Config.n_envs.")
parser.add_argument("--dataset-root", type=str, default=None,
                    help="Override Config.dataset_root.")
args = parser.parse_args()

# ── Config ────────────────────────────────────────────────────────────────────

from config import Config
cfg = Config()

if args.smoke_test:
    cfg.phase1_episodes  = 3
    cfg.phase2_episodes  = 3
    cfg.n_steps_per_env  = 16
    cfg.checkpoint_every = 2
    cfg.log_every        = 1
    print("[smoke-test] Overriding episode counts to 3.")

if args.n_envs:
    cfg.n_envs = args.n_envs

if args.dataset_root:
    cfg.dataset_root = args.dataset_root

if os.environ.get("DATASET_ROOT"):
    cfg.dataset_root = os.environ["DATASET_ROOT"]

cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device            : {cfg.device}")
print(f"Dataset root      : {cfg.dataset_root}")
print(f"Reconstructor ckpt: {cfg.reconstructor_ckpt}")

# ── Validate reconstructor checkpoint exists before spawning workers ──────────

if not os.path.isfile(cfg.reconstructor_ckpt):
    raise FileNotFoundError(
        f"Reconstructor checkpoint not found: {cfg.reconstructor_ckpt}\n"
        f"Update Config.reconstructor_ckpt in config.py to point to your .pth file."
    )

# ── Imports ───────────────────────────────────────────────────────────────────

from policy.view_policy      import ViewPolicy
from training.parallel_envs  import SubprocVecEnv, DummyVecEnv
from training.ppo_trainer    import PPOTrainer
from env.view_recon_env      import ViewReconEnv
from dataloader              import build_dataset

# ── Load dataset ──────────────────────────────────────────────────────────────

print("Loading dataset …")
dataset = build_dataset(
    dataset_root = cfg.dataset_root,
    categories   = cfg.categories,
    image_size   = (cfg.recon_img_size, cfg.recon_img_size),   # 256×256
    n_views      = cfg.n_views,
    preload      = False,   # set True if you have enough RAM for a speed boost
)
print(f"Dataset size: {len(dataset)} objects")

# ── Environment factory ───────────────────────────────────────────────────────

def make_env_fn(dataset_list, reconstructor_ckpt, view_budget, seed):
    """
    Returns a no-argument callable that builds one ViewReconEnv.

    The reconstructor checkpoint path (not a loaded model) is passed so
    each subprocess can load its own instance — CUDA objects cannot be
    pickled across multiprocessing Pipes.
    """
    def _make():
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
        return ViewReconEnv(
            dataset            = dataset,
            reconstructor_ckpt = reconstructor_ckpt,
            view_budget        = view_budget,
            lambda_cost        = cfg.cost_lambda,
            recon_img_size     = cfg.recon_img_size,
            voxel_size         = cfg.voxel_size,
            feature_dim        = cfg.feature_dim,
            n_views            = cfg.n_views,
        )
    return _make

# ── Build vectorised environment ──────────────────────────────────────────────

env_fns = [
    make_env_fn(
        dataset_list      = dataset,
        reconstructor_ckpt= cfg.reconstructor_ckpt,
        view_budget       = cfg.phase1_view_budget,
        seed              = i,
    )
    for i in range(cfg.n_envs)
]

VecEnvClass = DummyVecEnv if args.dummy_vec else SubprocVecEnv
print(f"VecEnv : {VecEnvClass.__name__} × {cfg.n_envs} workers")
vec_env = VecEnvClass(env_fns)

# ── Build RL policy ───────────────────────────────────────────────────────────

policy = ViewPolicy(n_views=cfg.n_views).to(cfg.device)
print(f"Policy parameters: {sum(p.numel() for p in policy.parameters()):,}")

# ── Clear stale RL checkpoints on a fresh run ─────────────────────────────────

if not args.resume:
    stale = glob.glob(os.path.join(cfg.checkpoint_dir, "ckpt_*.pt"))
    for f in stale:
        os.remove(f)
    if stale:
        print(f"Removed {len(stale)} stale RL checkpoint(s).")
    print("Starting fresh training run.")

# ── Train ─────────────────────────────────────────────────────────────────────

trainer = PPOTrainer(policy, vec_env, cfg)
try:
    trainer.train()
finally:
    vec_env.close()
    trainer.logger.close()
    print("Done.")
