"""HalfCheetah-v0 environment using JoltGym physics engine."""

import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces


def _asset_path(name: str) -> str:
    return os.path.join(os.path.dirname(__file__), "..", "assets", name)


class HalfCheetahEnv(gym.Env):
    """2D planar cheetah locomotion environment powered by Jolt Physics.

    A 4-legged cheetah with 6 actuated hinge joints (back thigh/shin/foot,
    front thigh/shin/foot) and a slide+hinge root. The goal is to run
    forward (positive X direction) as fast as possible.

    Observation:
        `Box(-inf, inf, (17,))` — `qpos[1:]` (skip root X) concatenated with `qvel`.

        | Index | Dim | Content |
        |-------|-----|---------|
        | 0     | 1   | root Z position (height) |
        | 1     | 1   | root Y rotation (torso angle) |
        | 2–7   | 6   | joint angles: bthigh, bshin, bfoot, fthigh, fshin, ffoot |
        | 8–10  | 3   | root velocities: vx, vz, angular vy |
        | 11–16 | 6   | joint velocities |

    Action:
        `Box(-1, 1, (6,))` — normalized joint torques scaled by gear ratios
        (120, 90, 60, 120, 60, 30).

    Reward:
        `forward_reward_weight * x_velocity - ctrl_cost_weight * sum(action²)`

    Attributes:
        observation_space: Gymnasium Box space of shape `(17,)`.
        action_space: Gymnasium Box space of shape `(6,)`.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(self, render_mode=None, forward_reward_weight=1.0,
                 ctrl_cost_weight=0.1, reset_noise_scale=0.1):
        """Initialize the HalfCheetah environment.

        Args:
            render_mode: Rendering mode — `"human"` for window, `"rgb_array"` for
                pixel output, or `None` to disable.
            forward_reward_weight: Multiplier on the forward velocity reward term.
            ctrl_cost_weight: Multiplier on the control cost penalty term.
            reset_noise_scale: Standard deviation of Gaussian noise added to joint
                positions and velocities on reset.
        """
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
        """Run one timestep (frame_skip=5 physics steps at dt=0.01s).

        Args:
            action: Normalized joint torques, shape `(6,)`, range `[-1, 1]`.

        Returns:
            obs: Observation array of shape `(17,)`.
            reward: Scalar reward (`forward_vel - ctrl_cost`).
            terminated: Always `False` (HalfCheetah has no terminal state).
            truncated: Always `False`.
            info: Dict with keys `x_position`, `x_velocity`, `reward_run`,
                `reward_ctrl`.
        """
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
        """Reset the environment to the initial state with optional noise.

        Args:
            seed: Random seed for reproducible resets.
            options: Unused, present for Gymnasium compatibility.

        Returns:
            obs: Initial observation array of shape `(17,)`.
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
