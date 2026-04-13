# Environments

JoltGym provides three Gymnasium-compatible environments, all registered under the `JoltGym/` namespace.

## HalfCheetah-v0

::: joltgym.envs.half_cheetah_v0.HalfCheetahEnv
    options:
      show_source: false
      members: [__init__, step, reset, close]

---

## Humanoid-v0

::: joltgym.envs.humanoid_v0.HumanoidEnv
    options:
      show_source: false
      members: [__init__, step, reset, close]

---

## CheetahRace-v0

::: joltgym.envs.cheetah_race_v0.CheetahRaceEnv
    options:
      show_source: false
      members: [__init__, step, reset, close, get_per_agent_obs, get_per_agent_actions]
