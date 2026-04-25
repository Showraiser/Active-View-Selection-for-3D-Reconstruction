"""
SubprocVecEnv  — true parallelism (default on Kaggle)
DummyVecEnv    — sequential fallback if multiprocessing fails in notebook kernel

Both expose the same interface:
  reset()            → stacked obs dict
  step(actions)      → stacked obs dict, rewards, dones, action_masks, infos
  set_phase(phase)
  set_budgets(list)
  close()
"""

import numpy as np
import multiprocessing as mp
from multiprocessing import Process, Pipe


# ── Subprocess worker ─────────────────────────────────────────────────────────

def _worker(conn, env_fn):
    env = env_fn()
    while True:
        cmd, payload = conn.recv()

        if cmd == 'reset':
            obs, _ = env.reset()
            mask = env.get_action_mask().astype(np.float32)
            conn.send((obs, mask))

        elif cmd == 'step':
            obs, reward, done, _, info = env.step(payload)
            if done:
                obs, _ = env.reset()
            mask = env.get_action_mask().astype(np.float32)
            conn.send((obs, float(reward), bool(done), mask, info))

        elif cmd == 'set_phase':
            env.set_phase(payload)
            conn.send('ok')

        elif cmd == 'set_budget':
            env.set_view_budget(payload)
            conn.send('ok')

        elif cmd == 'close':
            conn.close()
            break

        else:
            raise ValueError(f"Unknown command: {cmd}")


# ── SubprocVecEnv ─────────────────────────────────────────────────────────────

class SubprocVecEnv:
    def __init__(self, env_fns):
        self.n_envs = len(env_fns)
        self.parent_conns, child_conns = zip(*[Pipe() for _ in range(self.n_envs)])
        self.processes = []
        for fn, child in zip(env_fns, child_conns):
            p = Process(target=_worker, args=(child, fn), daemon=True)
            p.start()
            self.processes.append(p)

    def reset(self):
        for c in self.parent_conns:
            c.send(('reset', None))
        results = [c.recv() for c in self.parent_conns]
        obs_list, mask_list = zip(*results)
        return (
            self._stack(list(obs_list)),
            np.stack(mask_list, axis=0),      # (n_envs, 24)
        )

    def step(self, actions):
        for c, a in zip(self.parent_conns, actions):
            c.send(('step', int(a)))
        results = [c.recv() for c in self.parent_conns]
        obs_list, rewards, dones, masks, infos = zip(*results)
        return (
            self._stack(list(obs_list)),
            np.array(rewards, dtype=np.float32),
            np.array(dones,   dtype=bool),
            np.stack(masks,   axis=0),         # (n_envs, 24)
            list(infos),
        )

    def set_phase(self, phase: int):
        for c in self.parent_conns:
            c.send(('set_phase', phase))
        for c in self.parent_conns:
            c.recv()

    def set_budgets(self, budgets):
        for c, b in zip(self.parent_conns, budgets):
            c.send(('set_budget', b))
        for c in self.parent_conns:
            c.recv()

    def close(self):
        for c in self.parent_conns:
            c.send(('close', None))
        for p in self.processes:
            p.join()

    @staticmethod
    def _stack(obs_list):
        return {k: np.stack([o[k] for o in obs_list], axis=0) for k in obs_list[0]}


# ── DummyVecEnv (sequential fallback) ────────────────────────────────────────

class DummyVecEnv:
    def __init__(self, env_fns):
        self.envs   = [fn() for fn in env_fns]
        self.n_envs = len(self.envs)

    def reset(self):
        obs_list, mask_list = [], []
        for env in self.envs:
            obs, _ = env.reset()
            obs_list.append(obs)
            mask_list.append(env.get_action_mask().astype(np.float32))
        return SubprocVecEnv._stack(obs_list), np.stack(mask_list, axis=0)

    def step(self, actions):
        obs_list, rewards, dones, masks, infos = [], [], [], [], []
        for env, action in zip(self.envs, actions):
            obs, reward, done, _, info = env.step(int(action))
            if done:
                obs, _ = env.reset()
            obs_list.append(obs)
            rewards.append(float(reward))
            dones.append(bool(done))
            masks.append(env.get_action_mask().astype(np.float32))
            infos.append(info)
        return (
            SubprocVecEnv._stack(obs_list),
            np.array(rewards, dtype=np.float32),
            np.array(dones,   dtype=bool),
            np.stack(masks,   axis=0),
            infos,
        )

    def set_phase(self, phase):
        for e in self.envs:
            e.set_phase(phase)

    def set_budgets(self, budgets):
        for e, b in zip(self.envs, budgets):
            e.set_view_budget(b)

    def close(self):
        pass