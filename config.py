class Config:
    # ── Dataset ───────────────────────────────────────────────────────────────
    # Path to the dataset_preprocessed/ folder.
    # Override via --dataset-root CLI arg or $DATASET_ROOT env var in train.py.
    dataset_root = r"C:\Users\Aalin\Desktop\ML\RL\ModelNet_out_2"
    categories   = ["bed", "chair", "desk", "sofa", "table"]

    # ── Reconstructor ─────────────────────────────────────────────────────────
    # Path to the pretrained EncoderDecoder checkpoint (.pth file).
    # This model is loaded inside each environment worker (CPU) to score
    # reconstructions and compute the IoU reward.
    reconstructor_ckpt = r"C:\Users\Aalin\Desktop\ML\3D reconstruction\ep78_encoder_decoder_64_frz(iou0.4893_loss0.2232).pth"
    feature_dim        = 512   # encoder output / decoder input dimension
    voxel_size         = 64    # reconstructor output grid: (64, 64, 64)
    recon_img_size     = 256   # input image size expected by the reconstructor

    # ── Environment ───────────────────────────────────────────────────────────
    n_views      = 24          # 3D-R2N2 renders 24 views per object
    cost_lambda  = 0.05        # reward = iou - cost_lambda * steps_used

    # ── Parallelism ───────────────────────────────────────────────────────────
    n_envs = 8                 # number of parallel environment workers
                               # drop to 4 if OOM, or use --dummy-vec to go sequential

    # ── Rollout buffer ────────────────────────────────────────────────────────
    n_steps_per_env = 256      # total buffer size = n_envs × n_steps_per_env = 2 048

    # ── PPO ───────────────────────────────────────────────────────────────────
    learning_rate   = 3e-4
    gamma           = 0.99
    gae_lambda      = 0.95
    clip_ratio      = 0.2
    n_epochs        = 4
    minibatch_size  = 64
    value_loss_coef = 0.5
    max_grad_norm   = 0.5

    # ── Entropy ───────────────────────────────────────────────────────────────
    entropy_coef_view = 0.01   # encourages exploration of unseen viewpoints

    # ── Training phases ───────────────────────────────────────────────────────
    phase1_episodes    = 10_000
    phase1_view_budget = 5

    phase2_episodes     = 20_000
    phase2_view_budgets = [3, 5, 8]   # sampled uniformly at each reset

    # ── Checkpointing ─────────────────────────────────────────────────────────
    checkpoint_dir   = "checkpoints"
    checkpoint_every = 500     # save every N episodes

    # ── Logging ───────────────────────────────────────────────────────────────
    log_dir   = "logs"
    log_every = 100            # print + CSV flush every N episodes

    # ── Device (RL policy only) ───────────────────────────────────────────────
    # The reconstructor always runs on CPU inside environment workers.
    device = "cuda"
