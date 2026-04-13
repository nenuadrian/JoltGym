"""Register JoltGym environments with Gymnasium."""

import gymnasium as gym

_registered = False


def register_envs():
    global _registered
    if _registered:
        return
    _registered = True

    gym.register(
        id="JoltGym/HalfCheetah-v0",
        entry_point="joltgym.envs.half_cheetah_v0:HalfCheetahEnv",
        max_episode_steps=1000,
    )
    gym.register(
        id="JoltGym/CheetahRace-v0",
        entry_point="joltgym.envs.cheetah_race_v0:CheetahRaceEnv",
        max_episode_steps=1000,
    )
