"""
policy/view_policy.py
─────────────────────
Single-head actor-critic policy for view selection.

Architecture
────────────
  Encoder branches
    coverage_encoder : CoverageGridEncoder  (32³ voxel grid)  → 256
    feature_encoder  : MLP                  (512 image feats)  → 256
    history_encoder  : MLP                  (24 view mask)     →  64
                                             ─────────────────────
    concat + shared trunk                   576  →  512

  Heads
    view_head  : Linear  512 → 256 → 24   (actor — picks next view)
    value_head : Linear  512 → 128 →  1   (critic — estimates V(s))

The old model-selection head has been fully removed. There is exactly
one reconstructor; the environment calls it directly at episode end.
"""

import torch
import torch.nn as nn


# ── Sub-module: 3-D CNN for coverage grid ────────────────────────────────────

class CoverageGridEncoder(nn.Module):
    """
    Input : (B, 32, 32, 32) voxel occupancy grid
    Output: (B, 256)

    Stride-2 convolutions halve every spatial dimension:
      32³ → 16³ → 8³ → 4³  → Flatten(64×4³=4096) → Linear → 256
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4 * 4, 256),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 32, 32, 32)  →  add channel dim  →  (B, 1, 32, 32, 32)
        return self.net(x.unsqueeze(1))


# ── Main policy ───────────────────────────────────────────────────────────────

class ViewPolicy(nn.Module):
    """
    Actor-critic policy that selects the next viewpoint.

    Parameters
    ----------
    n_views : int
        Number of candidate views (default 24, matching 3D-R2N2 rendering).
    """

    def __init__(self, n_views: int = 24):
        super().__init__()
        self.n_views = n_views

        # ── Encoder branches ─────────────────────────────────────────────────
        self.coverage_encoder = CoverageGridEncoder()           # → 256
        self.feature_encoder  = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(inplace=True),
        )                                                        # → 256
        self.history_encoder  = nn.Sequential(
            nn.Linear(n_views, 64), nn.ReLU(inplace=True),
        )                                                        # →  64

        # ── Shared trunk  (256 + 256 + 64 = 576 → 512) ───────────────────────
        self.shared = nn.Sequential(
            nn.Linear(576, 512), nn.ReLU(inplace=True),
        )

        # ── Actor head: picks next view ───────────────────────────────────────
        self.view_head = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(inplace=True),
            nn.Linear(256, n_views),
        )

        # ── Critic head: estimates state value ───────────────────────────────
        self.value_head = nn.Sequential(
            nn.Linear(512, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

        self._init_weights()

    # ── Weight initialisation ─────────────────────────────────────────────────

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv3d)):
                nn.init.orthogonal_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ── Shared encoding step ──────────────────────────────────────────────────

    def _encode(self, obs: dict) -> torch.Tensor:
        """
        Parameters
        ----------
        obs : dict with keys
            'coverage_grid'  : (B, 32, 32, 32) float32
            'image_features' : (B, 512)         float32
            'view_mask'      : (B, n_views)      float32

        Returns
        -------
        latent : (B, 512)
        """
        cov  = self.coverage_encoder(obs['coverage_grid'])   # (B, 256)
        feat = self.feature_encoder(obs['image_features'])   # (B, 256)
        hist = self.history_encoder(obs['view_mask'])        # (B,  64)
        return self.shared(torch.cat([cov, feat, hist], dim=-1))  # (B, 512)

    # ── Forward pass ──────────────────────────────────────────────────────────

    def forward(self,
                obs: dict,
                action_mask: torch.Tensor = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        obs         : dict of (B, *) tensors
        action_mask : (B, n_views) float32 — 1 = available, 0 = already visited

        Returns
        -------
        view_logits : (B, n_views)
        values      : (B,)
        """
        latent      = self._encode(obs)
        view_logits = self.view_head(latent)
        values      = self.value_head(latent).squeeze(-1)

        if action_mask is not None:
            # Mask out already-visited views so they are never sampled
            view_logits = view_logits.masked_fill(action_mask == 0, float('-inf'))

        return view_logits, values

    # ── Inference (no gradient) ───────────────────────────────────────────────

    @torch.no_grad()
    def act(self,
            obs: dict,
            action_mask: torch.Tensor
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample an action from the current policy.

        Parameters
        ----------
        obs         : dict of (n_envs, *) tensors
        action_mask : (n_envs, n_views) float32

        Returns
        -------
        actions   : (n_envs,) int64
        log_probs : (n_envs,) float32
        values    : (n_envs,) float32
        """
        view_logits, values = self.forward(obs, action_mask)
        dist      = torch.distributions.Categorical(logits=view_logits)
        actions   = dist.sample()
        log_probs = dist.log_prob(actions)
        return actions, log_probs, values

    # ── Re-evaluation with gradient (used inside PPO update) ─────────────────

    def evaluate_actions(self,
                         obs: dict,
                         actions: torch.Tensor,
                         action_masks: torch.Tensor
                         ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Re-compute log-probs, values, and entropy for a batch of stored actions.

        Returns
        -------
        log_probs : (B,)
        values    : (B,)
        entropy   : (B,)
        """
        view_logits, values = self.forward(obs, action_masks)
        dist      = torch.distributions.Categorical(logits=view_logits)
        log_probs = dist.log_prob(actions)
        entropy   = dist.entropy()
        return log_probs, values, entropy
