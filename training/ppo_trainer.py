import os
import numpy as np
import torch
import torch.nn.functional as F

from training.rollout_buffer import RolloutBuffer
from utils.checkpoint import CheckpointManager
from utils.logger import Logger


def _obs_to_tensor(obs: dict, device: str) -> dict:
    return {k: torch.FloatTensor(v).to(device) for k, v in obs.items()}


class PPOTrainer:
    def __init__(self, policy, vec_env, config):
        self.policy  = policy
        self.vec_env = vec_env
        self.cfg     = config
        self.device  = config.device

        self.optimizer = torch.optim.Adam(
            policy.parameters(), lr=config.learning_rate
        )
        self.buffer = RolloutBuffer(
            n_steps  = config.n_steps_per_env,
            n_envs   = config.n_envs,
            n_views  = config.n_views,
            device   = config.device,
        )
        self.ckpt   = CheckpointManager(config.checkpoint_dir)
        self.logger = Logger(config.log_dir, config.categories)

        self.total_episodes = 0
        self.total_steps    = 0
        self.phase          = 1
        self.current_obs    = None
        self.current_masks  = None   # (n_envs, 24) action masks between rollouts

    # ── Entry point ───────────────────────────────────────────────────────

    def train(self):
        self._maybe_resume()

        print("=== Phase 1: fixed view budget B=5 =====================")
        self.vec_env.set_phase(1)
        self.vec_env.set_budgets([self.cfg.phase1_view_budget] * self.cfg.n_envs)
        self.current_obs, self.current_masks = self.vec_env.reset()
        self._run_phase(target_episodes=self.cfg.phase1_episodes)

        print("=== Phase 2: curriculum {3, 5, 8} ======================")
        self.phase = 2
        self.vec_env.set_phase(2)
        self.current_obs, self.current_masks = self.vec_env.reset()
        self._run_phase(target_episodes=self.cfg.phase2_episodes)

        print("=== Training complete. ==================================")
        self.ckpt.save(self._state_dict(), tag="final")

    # ── Phase loop ────────────────────────────────────────────────────────

    def _run_phase(self, target_episodes: int):
        episodes_at_start = self.total_episodes

        while (self.total_episodes - episodes_at_start) < target_episodes:
            ep_rewards, ep_infos = self._collect_rollout()
            metrics = self._update()

            # ── Per-episode terminal output ───────────────────────────────
            for idx, (r, info) in enumerate(zip(ep_rewards, ep_infos)):
                ep_num = self.total_episodes + idx + 1
                iou    = info.get('iou',      float('nan'))
                views  = info.get('n_views',  '?')
                cat    = info.get('category', '?')
                print(
                    f"  ep {ep_num:>6} | "
                    f"reward {r:+.4f} | "
                    f"IoU {iou:.4f} | "
                    f"views {views} | "
                    f"cat {cat} | "
                    f"π {metrics['policy_loss']:.4f}  "
                    f"V {metrics['value_loss']:.4f}  "
                    f"H {metrics['entropy']:.3f}"
                )

            self.total_episodes += len(ep_rewards)
            self.logger.record(ep_rewards, ep_infos, metrics, self.total_episodes)

            if self.total_episodes % self.cfg.checkpoint_every < self.cfg.n_envs:
                self.ckpt.save(self._state_dict(), tag=f"ep{self.total_episodes}")
                print(f"  [checkpoint] episode {self.total_episodes}")

    # ── Rollout collection ────────────────────────────────────────────────

    def _collect_rollout(self):
        self.buffer.reset()
        self.policy.eval()

        completed_rewards = []
        completed_infos   = []
        obs   = self.current_obs
        masks = self.current_masks   # (n_envs, 24) float32

        print(f"  [rollout] collecting {self.cfg.n_steps_per_env} steps × {self.cfg.n_envs} envs "
              f"(total ep so far: {self.total_episodes}) ...", flush=True)

        for _ in range(self.cfg.n_steps_per_env):
            obs_t  = _obs_to_tensor(obs, self.device)
            mask_t = torch.FloatTensor(masks).to(self.device)

            with torch.no_grad():
                actions, log_probs, values = self.policy.act(obs_t, mask_t)

            next_obs, rewards, dones, next_masks, infos = self.vec_env.step(
                actions.cpu().numpy()
            )

            for i, done in enumerate(dones):
                if done:
                    completed_rewards.append(rewards[i])
                    completed_infos.append(infos[i])

            self.buffer.add(
                obs          = obs,
                actions      = actions.cpu().numpy(),
                log_probs    = log_probs.cpu().numpy(),
                values       = values.cpu().numpy(),
                rewards      = rewards,
                dones        = dones,
                action_masks = masks,
            )

            obs   = next_obs
            masks = next_masks
            self.total_steps += self.cfg.n_envs

        # Bootstrap value after the final stored step
        with torch.no_grad():
            _, last_values = self.policy.forward(
                _obs_to_tensor(obs, self.device)
            )

        self.buffer.compute_gae(
            last_values = last_values.cpu().numpy(),
            last_dones  = dones.astype(np.float32),
            gamma       = self.cfg.gamma,
            gae_lambda  = self.cfg.gae_lambda,
        )

        self.current_obs   = obs
        self.current_masks = masks
        return completed_rewards, completed_infos

    # ── PPO update ────────────────────────────────────────────────────────

    def _update(self) -> dict:
        self.policy.train()
        policy_losses, value_losses, entropies = [], [], []

        for _ in range(self.cfg.n_epochs):
            for batch in self.buffer.get_batches(self.cfg.minibatch_size):
                obs_b, actions_b, old_lp_b, adv_b, ret_b, amask_b = batch

                log_probs, values, entropy = self.policy.evaluate_actions(
                    obs_b, actions_b, amask_b
                )

                # Clipped surrogate loss
                ratio    = torch.exp(log_probs - old_lp_b)
                pg_loss1 = -adv_b * ratio
                pg_loss2 = -adv_b * torch.clamp(
                    ratio, 1 - self.cfg.clip_ratio, 1 + self.cfg.clip_ratio
                )
                policy_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                value_loss = F.mse_loss(values, ret_b)

                # Entropy bonus (maximise entropy → subtract from loss)
                entropy_loss = -self.cfg.entropy_coef_view * entropy.mean()

                loss = policy_loss + self.cfg.value_loss_coef * value_loss + entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.cfg.max_grad_norm
                )
                self.optimizer.step()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy.mean().item())

        return {
            'policy_loss': float(np.mean(policy_losses)),
            'value_loss':  float(np.mean(value_losses)),
            'entropy':     float(np.mean(entropies)),
        }

    # ── Helpers ───────────────────────────────────────────────────────────

    def _state_dict(self) -> dict:
        return {
            'policy':         self.policy.state_dict(),
            'optimizer':      self.optimizer.state_dict(),
            'total_episodes': self.total_episodes,
            'total_steps':    self.total_steps,
            'phase':          self.phase,
        }

    def _maybe_resume(self):
        state = self.ckpt.load_latest()
        if state is None:
            self.current_obs, self.current_masks = self.vec_env.reset()
            return
        self.policy.load_state_dict(state['policy'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.total_episodes = state['total_episodes']
        self.total_steps    = state['total_steps']
        self.phase          = state['phase']
        self.current_obs, self.current_masks = self.vec_env.reset()
        print(f"  Resumed from episode {self.total_episodes} (phase {self.phase})")