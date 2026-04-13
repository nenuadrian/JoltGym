"""Multi-agent CheetahRace environment using JoltGym physics engine."""

import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces


def _asset_path(name: str) -> str:
    return os.path.join(os.path.dirname(__file__), "..", "assets", name)


class CheetahRaceEnv(gym.Env):
    """Multi-agent cheetah race in a shared physics world.

    N cheetahs are placed side-by-side (Y-offset) and race forward (X-axis).
    All agents share the same `PhysicsWorld`, so they can physically collide
    and interact.

    This wraps all agents into a single Gymnasium env suitable for
    independent-learner multi-agent training with parameter sharing.
    Observations and actions are flat concatenations of per-agent vectors.

    Observation:
        `Box(-inf, inf, (N*17,))` — concatenation of each agent's
        `qpos[1:]` + `qvel`.

    Action:
        `Box(-1, 1, (N*6,))` — concatenation of each agent's normalized
        joint torques.

    Reward:
        Sum of all agents' individual rewards
        (`forward_velocity - ctrl_cost` per agent).

    Attributes:
        num_agents: Number of cheetahs in the race.
        observation_space: Gymnasium Box space of shape `(num_agents * 17,)`.
        action_space: Gymnasium Box space of shape `(num_agents * 6,)`.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(self, num_agents=2, render_mode=None,
                 agent_spacing=3.0,
                 forward_reward_weight=1.0,
                 ctrl_cost_weight=0.1,
                 reset_noise_scale=0.1):
        """Initialize the CheetahRace environment.

        Args:
            num_agents: Number of cheetahs in the race.
            render_mode: Rendering mode — `"human"`, `"rgb_array"`, or `None`.
            agent_spacing: Y-axis distance between adjacent agents.
            forward_reward_weight: Multiplier on forward velocity reward.
            ctrl_cost_weight: Multiplier on control cost penalty.
            reset_noise_scale: Standard deviation of noise added on reset.
        """
        super().__init__()

        from joltgym import joltgym_native

        self.num_agents = num_agents
        self._core = joltgym_native.MultiAgentEnv(
            num_agents=num_agents,
            model_path=_asset_path("half_cheetah.xml"),
            agent_spacing=agent_spacing,
            forward_reward_weight=forward_reward_weight,
            ctrl_cost_weight=ctrl_cost_weight,
        )
        self.render_mode = render_mode
        self._reset_noise_scale = reset_noise_scale
        self._step_count = 0

        obs_dim = self._core.obs_dim
        act_dim = self._core.act_dim

        # Flat observation/action spaces (all agents concatenated)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(num_agents * obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(num_agents * act_dim,), dtype=np.float32
        )

        self._per_agent_obs_dim = obs_dim
        self._per_agent_act_dim = act_dim

    def step(self, action):
        """Run one timestep for all agents simultaneously.

        Args:
            action: Flat array of shape `(num_agents * 6,)`.

        Returns:
            obs: Flat observation of shape `(num_agents * 17,)`.
            reward: Scalar total reward (sum of all agents).
            terminated: Always `False`.
            truncated: Always `False`.
            info: Dict with `per_agent_reward` array, and per-agent
                `agent_{i}_x` / `agent_{i}_xvel` keys.
        """
        action = np.asarray(action, dtype=np.float32).reshape(
            self.num_agents, self._per_agent_act_dim)
        obs, rewards = self._core.step(action)

        self._step_count += 1

        info = {
            "per_agent_reward": rewards.copy(),
        }
        for i in range(self.num_agents):
            info[f"agent_{i}_x"] = self._core.get_agent_x(i)
            info[f"agent_{i}_xvel"] = self._core.get_agent_x_velocity(i)

        total_reward = float(rewards.sum())

        return (obs.flatten().astype(np.float32),
                total_reward, False, False, info)

    def reset(self, *, seed=None, options=None):
        """Reset all agents to their initial positions.

        Args:
            seed: Random seed for reproducible resets.
            options: Unused, present for Gymnasium compatibility.

        Returns:
            obs: Flat initial observation of shape `(num_agents * 17,)`.
            info: Empty dict.
        """
        super().reset(seed=seed)
        self._step_count = 0

        obs = self._core.reset_all(
            seed=seed if seed is not None else None,
            noise_scale=self._reset_noise_scale,
        )

        return obs.flatten().astype(np.float32), {}

    def render(self):
        """Render the environment (not yet implemented)."""
        pass

    def close(self):
        """Clean up resources."""
        pass

    def get_per_agent_obs(self, flat_obs):
        """Split a flat observation into per-agent arrays.

        Args:
            flat_obs: Flat observation of shape `(num_agents * 17,)`.

        Returns:
            Per-agent observations of shape `(num_agents, 17)`.
        """
        return flat_obs.reshape(self.num_agents, self._per_agent_obs_dim)

    def get_per_agent_actions(self, flat_action):
        """Split a flat action into per-agent arrays.

        Args:
            flat_action: Flat action of shape `(num_agents * 6,)`.

        Returns:
            Per-agent actions of shape `(num_agents, 6)`.
        """
        return flat_action.reshape(self.num_agents, self._per_agent_act_dim)
