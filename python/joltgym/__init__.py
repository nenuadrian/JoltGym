"""JoltGym — MuJoCo-compatible physics simulation for RL, built on Jolt Physics."""

from joltgym.envs.registration import register_envs
from joltgym.envs import HalfCheetahEnv

register_envs()


def make(env_id: str, **kwargs):
    """Create a JoltGym environment by ID."""
    import gymnasium as gym
    return gym.make(env_id, **kwargs)
