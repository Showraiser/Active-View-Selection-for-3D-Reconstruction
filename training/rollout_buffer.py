import numpy as np
import torch


class RolloutBuffer:
    """
    Stores n_steps × n_envs transitions, then computes GAE.
    No model-step tracking — pure view selection buffer.
    """

    def __init__(self, n_steps: int, n_envs: int, n_views: int, device: str):
        self.n_steps  = n_steps
        self.n_envs   = n_envs
        self.n_views  = n_views   # 24
        self.device   = device
        self._alloc()

    def _alloc(self):
        N, E, V = self.n_steps, self.n_envs, self.n_views

        # Observations (numpy until minibatch is sent to GPU)
        self.cov_grids  = np.zeros((N, E, 32, 32, 32), dtype=np.float32)
        self.img_feats  = np.zeros((N, E, 512),         dtype=np.float32)
        self.hist_masks = np.zeros((N, E, V),            dtype=np.float32)

        # Actions and masks
        self.actions      = np.zeros((N, E),    dtype=np.int64)
        self.action_masks = np.zeros((N, E, V), dtype=np.float32)

        # PPO scalars
        self.log_probs = np.zeros((N, E), dtype=np.float32)
        self.values    = np.zeros((N, E), dtype=np.float32)
        self.rewards   = np.zeros((N, E), dtype=np.float32)
        self.dones     = np.zeros((N, E), dtype=np.float32)

        # Filled by compute_gae()
        self.advantages = np.zeros((N, E), dtype=np.float32)
        self.returns    = np.zeros((N, E), dtype=np.float32)

        self.ptr  = 0
        self.full = False

    def reset(self):
        self._alloc()

    def add(self, obs, actions, log_probs, values,
            rewards, dones, action_masks):
        t = self.ptr
        self.cov_grids[t]   = obs['coverage_grid']
        self.img_feats[t]   = obs['image_features']
        self.hist_masks[t]  = obs['view_mask']         # ← key matches Person 1
        self.actions[t]     = actions
        self.log_probs[t]   = log_probs
        self.values[t]      = values
        self.rewards[t]     = rewards
        self.dones[t]       = dones.astype(np.float32)
        self.action_masks[t] = action_masks

        self.ptr += 1
        if self.ptr == self.n_steps:
            self.full = True

    def compute_gae(self, last_values: np.ndarray, last_dones: np.ndarray,
                    gamma: float, gae_lambda: float):
        gae = np.zeros(self.n_envs, dtype=np.float32)

        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                next_non_terminal = 1.0 - last_dones.astype(np.float32)
                next_values       = last_values
            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                next_values       = self.values[t + 1]

            delta = (self.rewards[t]
                     + gamma * next_values * next_non_terminal
                     - self.values[t])
            gae = delta + gamma * gae_lambda * next_non_terminal * gae
            self.advantages[t] = gae

        self.returns = self.advantages + self.values

    def get_batches(self, minibatch_size: int):
        assert self.full, "Buffer not full — call add() n_steps times first."
        N  = self.n_steps * self.n_envs
        V  = self.n_views

        # Flatten
        cov   = self.cov_grids.reshape(N, 32, 32, 32)
        feat  = self.img_feats.reshape(N, 512)
        hist  = self.hist_masks.reshape(N, V)
        acts  = self.actions.reshape(N)
        lps   = self.log_probs.reshape(N)
        advs  = self.advantages.reshape(N)
        rets  = self.returns.reshape(N)
        amask = self.action_masks.reshape(N, V)

        # Normalise advantages
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        indices = np.random.permutation(N)
        for start in range(0, N, minibatch_size):
            idx = indices[start: start + minibatch_size]
            obs_batch = {
                'coverage_grid':  torch.FloatTensor(cov[idx]).to(self.device),
                'image_features': torch.FloatTensor(feat[idx]).to(self.device),
                'view_mask':      torch.FloatTensor(hist[idx]).to(self.device),
            }
            yield (
                obs_batch,
                torch.LongTensor(acts[idx]).to(self.device),
                torch.FloatTensor(lps[idx]).to(self.device),
                torch.FloatTensor(advs[idx]).to(self.device),
                torch.FloatTensor(rets[idx]).to(self.device),
                torch.FloatTensor(amask[idx]).to(self.device),
            )