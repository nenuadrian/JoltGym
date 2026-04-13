"""Humanoid-v0 environment using JoltGym physics engine."""

import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces


def _asset_path(name: str) -> str:
    return os.path.join(os.path.dirname(__file__), "..", "assets", name)


class HumanoidEnv(gym.Env):
    """3D bipedal humanoid locomotion environment powered by Jolt Physics.

    A humanoid with 17 actuated joints (abdomen, hips, knees, shoulders,
    elbows) and a free 6DOF root body. The goal is to walk forward while
    staying upright.

    Observation:
        `Box(-inf, inf, (45,))` — `qpos[2:]` (skip root X, Y) concatenated
        with `qvel`.

        | Index  | Dim | Content |
        |--------|-----|---------|
        | 0      | 1   | root Z position (height) |
        | 1–4    | 4   | root quaternion (w, x, y, z) |
        | 5–21   | 17  | joint angles |
        | 22–24  | 3   | root linear velocity (x, y, z) |
        | 25–27  | 3   | root angular velocity (x, y, z) |
        | 28–44  | 17  | joint velocities |

    Action:
        `Box(-0.4, 0.4, (17,))` — normalized joint torques for 17 actuated
        joints: abdomen (3), right hip (3) + knee, left hip (3) + knee,
        right shoulder (2) + elbow, left shoulder (2) + elbow.

    Reward:
        `forward_reward_weight * x_velocity + healthy_reward * is_healthy
        - ctrl_cost_weight * sum(action²)`

    Termination:
        Episode ends when root Z position is outside
        `[healthy_z_min, healthy_z_max]`.

    Attributes:
        observation_space: Gymnasium Box space of shape `(45,)`.
        action_space: Gymnasium Box space of shape `(17,)`.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 67}

    def __init__(self, render_mode=None,
                 forward_reward_weight=1.25,
                 ctrl_cost_weight=0.1,
                 healthy_reward=5.0,
                 healthy_z_min=1.0,
                 healthy_z_max=2.0,
                 reset_noise_scale=0.005):
        """Initialize the Humanoid environment.

        Args:
            render_mode: Rendering mode — `"human"`, `"rgb_array"`, or `None`.
            forward_reward_weight: Multiplier on the forward velocity reward.
            ctrl_cost_weight: Multiplier on the control cost penalty.
            healthy_reward: Bonus reward for staying upright each step.
            healthy_z_min: Minimum root Z height to be considered healthy.
            healthy_z_max: Maximum root Z height to be considered healthy.
            reset_noise_scale: Standard deviation of noise added on reset.
        """
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
        """Run one timestep (frame_skip=5 physics steps at dt=0.003s).

        Args:
            action: Normalized joint torques, shape `(17,)`, range `[-0.4, 0.4]`.

        Returns:
            obs: Observation array of shape `(45,)`.
            reward: Scalar reward.
            terminated: `True` when root Z leaves `[healthy_z_min, healthy_z_max]`.
            truncated: Always `False`.
            info: Dict with keys `x_position`, `z_position`, `x_velocity`,
                `reward_forward`, `reward_ctrl`.
        """
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
        """Reset the environment to the initial state with optional noise.

        Args:
            seed: Random seed for reproducible resets.
            options: Unused, present for Gymnasium compatibility.

        Returns:
            obs: Initial observation array of shape `(45,)`.
            info: Empty dict.
        """
        super().reset(seed=seed)

        if seed is not None:
            obs = self._core.reset(seed=seed, noise_scale=self._reset_noise_scale)
        else:
            obs = self._core.reset(noise_scale=self._reset_noise_scale)

        info = {}
        return np.asarray(obs), info

    def render(self):
        """Render the environment (not yet implemented)."""
        pass  # TODO: Integrate Vulkan renderer

    def close(self):
        """Shut down the underlying C++ physics engine."""
        self._core.shutdown()
