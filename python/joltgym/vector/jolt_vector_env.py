"""Vectorized environment using C++ WorldPool for parallel simulation."""

import numpy as np
from gymnasium import spaces


class JoltVectorEnv:
    """N parallel HalfCheetah environments stepped in C++ threads.

    All N PhysicsSystem instances are stepped in parallel — the entire
    hot loop (action apply, physics step, obs extraction, reward compute)
    runs in C++ with the GIL released.
    """

    def __init__(self, num_envs, model_path, **kwargs):
        from joltgym import joltgym_native

        self._pool = joltgym_native.WorldPool(
            num_envs=num_envs,
            model_path=model_path,
            **kwargs,
        )
        self.num_envs = num_envs

        obs_dim = self._pool.obs_dim
        act_dim = self._pool.act_dim

        self.single_observation_space = spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32)
        self.single_action_space = spaces.Box(-1.0, 1.0, (act_dim,), np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (num_envs, obs_dim), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (num_envs, act_dim), np.float32)

    def step(self, actions):
        actions = np.asarray(actions, dtype=np.float32)
        obs, rewards, dones = self._pool.step_all(actions)
        infos = [{} for _ in range(self.num_envs)]
        truncs = np.zeros(self.num_envs, dtype=bool)
        return obs, rewards, dones, truncs, infos

    def reset(self, *, seed=None, options=None):
        obs = self._pool.reset_all(seed=seed)
        infos = [{} for _ in range(self.num_envs)]
        return obs, infos

    def close(self):
        pass
