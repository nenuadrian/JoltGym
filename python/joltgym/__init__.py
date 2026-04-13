"""JoltGym — MuJoCo-compatible physics simulation for RL, built on Jolt Physics.

JoltGym provides high-performance Gymnasium-compatible environments powered by
the Jolt Physics engine with Vulkan rendering and zero-copy Python bindings.

Registered environments:

- `JoltGym/HalfCheetah-v0` — 2D planar cheetah (6 actuated joints)
- `JoltGym/Humanoid-v0` — 3D bipedal humanoid (17 actuated joints)
- `JoltGym/CheetahRace-v0` — N multi-agent cheetahs in a shared world
"""

from joltgym.envs.registration import register_envs
from joltgym.envs import HalfCheetahEnv

register_envs()


def make(env_id: str, **kwargs):
    """Create a JoltGym environment by ID.

    Wraps `gymnasium.make()` with JoltGym's registered environments.

    Args:
        env_id: Environment identifier, e.g. `"JoltGym/HalfCheetah-v0"`.
        **kwargs: Forwarded to the environment constructor.

    Returns:
        A Gymnasium environment instance.

    Examples:
        >>> import joltgym
        >>> env = joltgym.make("JoltGym/HalfCheetah-v0")
        >>> obs, info = env.reset(seed=42)
    """
    import gymnasium as gym
    return gym.make(env_id, **kwargs)
