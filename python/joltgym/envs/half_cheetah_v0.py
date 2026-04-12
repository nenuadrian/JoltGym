"""HalfCheetah-v0 environment using JoltGym physics engine."""

import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces


def _asset_path(name: str) -> str:
    return os.path.join(os.path.dirname(__file__), "..", "assets", name)


class HalfCheetahEnv(gym.Env):
    """HalfCheetah environment powered by Jolt Physics.

    Observation Space: Box(-inf, inf, (17,)) — qpos[1:] ++ qvel
    Action Space: Box(-1, 1, (6,)) — normalized joint torques
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(self, render_mode=None, forward_reward_weight=1.0,
                 ctrl_cost_weight=0.1, reset_noise_scale=0.1):
        super().__init__()

        from joltgym import joltgym_native

        self._core = joltgym_native.HalfCheetahCore(
            model_path=_asset_path("half_cheetah.xml"),
            forward_reward_weight=forward_reward_weight,
            ctrl_cost_weight=ctrl_cost_weight,
        )
        self.render_mode = render_mode
        self._reset_noise_scale = reset_noise_scale

        obs_dim = self._core.get_obs_dim()
        act_dim = self._core.get_action_dim()

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32
        )

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        obs, reward, terminated, truncated = self._core.step(action)

        info = {
            "x_position": self._core.get_root_x(),
            "x_velocity": self._core.get_x_velocity(),
            "reward_run": self._core.get_forward_reward(),
            "reward_ctrl": -self._core.get_ctrl_cost(),
        }

        return np.asarray(obs), float(reward), bool(terminated), bool(truncated), info

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        if seed is not None:
            obs = self._core.reset(seed=seed, noise_scale=self._reset_noise_scale)
        else:
            obs = self._core.reset(noise_scale=self._reset_noise_scale)

        info = {}
        return np.asarray(obs), info

    def render(self):
        pass  # TODO: Integrate Vulkan renderer

    def close(self):
        self._core.shutdown()
