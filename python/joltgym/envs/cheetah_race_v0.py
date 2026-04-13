"""Multi-agent CheetahRace: N cheetahs in one shared physics world.

Each agent is an independent learner observing its own state.
All agents share the same physics world — they can collide and interact.
"""

import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces


def _asset_path(name: str) -> str:
    return os.path.join(os.path.dirname(__file__), "..", "assets", name)


class CheetahRaceEnv(gym.Env):
    """Multi-agent cheetah race in a shared physics world.

    N cheetahs are placed side-by-side (Y-offset) and race forward (X-axis).
    Each agent gets its own observation and reward. Actions are concatenated.

    This wraps all agents into a single Gymnasium env suitable for
    independent-learner multi-agent training: the action space is
    (num_agents * 6,) and observations are (num_agents * 17,).
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(self, num_agents=2, render_mode=None,
                 agent_spacing=3.0,
                 forward_reward_weight=1.0,
                 ctrl_cost_weight=0.1,
                 reset_noise_scale=0.1):
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
        super().reset(seed=seed)
        self._step_count = 0

        obs = self._core.reset_all(
            seed=seed if seed is not None else None,
            noise_scale=self._reset_noise_scale,
        )

        return obs.flatten().astype(np.float32), {}

    def render(self):
        pass

    def close(self):
        pass

    # --- Convenience for per-agent access ---

    def get_per_agent_obs(self, flat_obs):
        """Split flat obs into per-agent observations."""
        return flat_obs.reshape(self.num_agents, self._per_agent_obs_dim)

    def get_per_agent_actions(self, flat_action):
        """Split flat action into per-agent actions."""
        return flat_action.reshape(self.num_agents, self._per_agent_act_dim)
