"""Vectorized environment using C++ WorldPool for parallel simulation."""

import numpy as np
from gymnasium import spaces


class JoltVectorEnv:
    """N parallel HalfCheetah environments stepped in C++ threads.

    Wraps the C++ `WorldPool` class for maximum throughput. All N
    `PhysicsSystem` instances are stepped in parallel via native OS threads
    with the GIL released — the entire hot loop (action apply, physics step,
    observation extraction, reward computation) runs in C++.

    Achieves ~73K env-steps/sec at 256 environments on Apple Silicon.

    Attributes:
        num_envs: Number of parallel environments.
        observation_space: Batched observation space `(num_envs, obs_dim)`.
        action_space: Batched action space `(num_envs, act_dim)`.
        single_observation_space: Single-env observation space `(obs_dim,)`.
        single_action_space: Single-env action space `(act_dim,)`.
    """

    def __init__(self, num_envs, model_path, **kwargs):
        """Initialize the vectorized environment pool.

        Args:
            num_envs: Number of parallel environments to create.
            model_path: Path to the MJCF XML model file.
            **kwargs: Forwarded to `WorldPool` (e.g. `forward_reward_weight`,
                `ctrl_cost_weight`).
        """
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
        """Step all environments in parallel.

        The GIL is released for the entire duration of the C++ computation.
        Environments that reach a terminal state are auto-reset.

        Args:
            actions: Array of shape `(num_envs, act_dim)`, dtype `float32`.

        Returns:
            obs: Observations, shape `(num_envs, obs_dim)`.
            rewards: Rewards, shape `(num_envs,)`.
            dones: Terminal flags, shape `(num_envs,)`. `True` indicates
                the environment was auto-reset.
            truncs: Truncation flags, shape `(num_envs,)` (always `False`).
            infos: List of empty dicts.
        """
        actions = np.asarray(actions, dtype=np.float32)
        obs, rewards, dones = self._pool.step_all(actions)
        infos = [{} for _ in range(self.num_envs)]
        truncs = np.zeros(self.num_envs, dtype=bool)
        return obs, rewards, dones, truncs, infos

    def reset(self, *, seed=None, options=None):
        """Reset all environments in parallel.

        Args:
            seed: Optional base seed. Environment *i* receives `seed + i`.
            options: Unused, present for compatibility.

        Returns:
            obs: Initial observations, shape `(num_envs, obs_dim)`.
            infos: List of empty dicts.
        """
        obs = self._pool.reset_all(seed=seed)
        infos = [{} for _ in range(self.num_envs)]
        return obs, infos

    def close(self):
        """Clean up resources (no-op, pool is managed by C++)."""
        pass
