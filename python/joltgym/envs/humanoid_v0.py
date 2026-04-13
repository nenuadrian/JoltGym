"""Humanoid-v0 environment using JoltGym physics engine.

3D humanoid locomotion with 17 actuated joints and a free (6DOF) root body.
Observation is qpos[2:] ++ qvel (skip root x,y position).
"""

import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces


def _asset_path(name: str) -> str:
    return os.path.join(os.path.dirname(__file__), "..", "assets", name)


class HumanoidEnv(gym.Env):
    """Humanoid environment powered by Jolt Physics.

    Observation Space: Box(-inf, inf, (45,))
        qpos[2:]: root_z(1), quat(4), joints(17) = 22
        qvel: linear(3), angular(3), joints(17) = 23
        Total: 45

    Action Space: Box(-0.4, 0.4, (17,)) — normalized joint torques
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 67}

    def __init__(self, render_mode=None,
                 forward_reward_weight=1.25,
                 ctrl_cost_weight=0.1,
                 healthy_reward=5.0,
                 healthy_z_min=1.0,
                 healthy_z_max=2.0,
                 reset_noise_scale=0.005):
        super().__init__()

        from joltgym import joltgym_native

        self._core = joltgym_native.HumanoidCore(
            model_path=_asset_path("humanoid.xml"),
            forward_reward_weight=forward_reward_weight,
            ctrl_cost_weight=ctrl_cost_weight,
            healthy_reward=healthy_reward,
            healthy_z_min=healthy_z_min,
            healthy_z_max=healthy_z_max,
        )
        self.render_mode = render_mode
        self._reset_noise_scale = reset_noise_scale

        obs_dim = self._core.get_obs_dim()
        act_dim = self._core.get_action_dim()

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-0.4, high=0.4, shape=(act_dim,), dtype=np.float32
        )

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        obs, reward, terminated, truncated = self._core.step(action)

        info = {
            "x_position": self._core.get_root_x(),
            "z_position": self._core.get_root_z(),
            "x_velocity": self._core.get_x_velocity(),
            "reward_forward": self._core.get_forward_reward(),
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
