# Environments

JoltGym provides three Gymnasium-compatible environments, all registered under the `JoltGym/` namespace.

## HalfCheetah-v0

A 2D planar cheetah with 6 actuated hinge joints (back thigh, shin, foot; front thigh, shin, foot) and a slide+hinge root.

```python
import joltgym
env = joltgym.make("JoltGym/HalfCheetah-v0")
```

### Specification

| Property | Value |
|---|---|
| Observation Space | `Box(-inf, inf, (17,))` |
| Action Space | `Box(-1.0, 1.0, (6,))` |
| Reward | `forward_velocity - 0.1 * ctrl_cost` |
| Timestep | 0.01s |
| Frame Skip | 5 (effective dt = 0.05s) |
| MJCF Model | `half_cheetah.xml` |

### Observation Layout

The observation vector is `qpos[1:]` concatenated with `qvel` (root X position is skipped):

| Index | Dim | Content |
|---|---|---|
| 0 | 1 | root Z position (height) |
| 1 | 1 | root Y rotation (torso angle) |
| 2--7 | 6 | joint angles: bthigh, bshin, bfoot, fthigh, fshin, ffoot |
| 8--10 | 3 | root velocities: vx, vz, angular_vy |
| 11--16 | 6 | joint velocities |

### Action Layout

Each action value is a normalized torque in `[-1, 1]`, scaled by the actuator's gear ratio:

| Index | Joint | Gear Ratio |
|---|---|---|
| 0 | bthigh | 120 |
| 1 | bshin | 90 |
| 2 | bfoot | 60 |
| 3 | fthigh | 120 |
| 4 | fshin | 60 |
| 5 | ffoot | 30 |

### Reward

$$
r = v_x - 0.1 \sum_{i} a_i^2
$$

Where $v_x$ is the forward (X-axis) velocity and $a_i$ are the action values.

### Constructor Parameters

```python
HalfCheetahEnv(
    render_mode=None,           # "human" or "rgb_array"
    forward_reward_weight=1.0,  # weight on forward velocity reward
    ctrl_cost_weight=0.1,       # weight on control cost penalty
    reset_noise_scale=0.1,      # std of noise added on reset
)
```

---

## Humanoid-v0

A 3D bipedal humanoid with 17 actuated joints and a free 6DOF root body. The humanoid has arms, legs, an abdomen, and sphere-geom hands and feet.

```python
import joltgym
env = joltgym.make("JoltGym/Humanoid-v0")
```

### Specification

| Property | Value |
|---|---|
| Observation Space | `Box(-inf, inf, (45,))` |
| Action Space | `Box(-0.4, 0.4, (17,))` |
| Reward | `1.25 * forward_vel + 5.0 * healthy - 0.1 * ctrl_cost` |
| Termination | root Z outside [1.0, 2.0] |
| Timestep | 0.003s |
| Frame Skip | 5 (effective dt = 0.015s) |
| MJCF Model | `humanoid.xml` |

### Observation Layout

The observation vector is `qpos[2:]` (skip root X and Y) concatenated with `qvel`:

| Index | Dim | Content |
|---|---|---|
| 0 | 1 | root Z position (height) |
| 1--4 | 4 | root quaternion (w, x, y, z) |
| 5--21 | 17 | joint angles |
| 22--24 | 3 | root linear velocity (x, y, z) |
| 25--27 | 3 | root angular velocity (x, y, z) |
| 28--44 | 17 | joint velocities |

### Joint Layout

The 17 actuated joints:

| Index | Joint | DOF |
|---|---|---|
| 0--2 | abdomen (y, z, x) | 3 |
| 3--6 | right hip (x, z, y) + knee | 4 |
| 7--10 | left hip (x, z, y) + knee | 4 |
| 11--13 | right shoulder (1, 2) + elbow | 3 |
| 14--16 | left shoulder (1, 2) + elbow | 3 |

### Reward

$$
r = 1.25 \cdot v_x + 5.0 \cdot \mathbb{1}[\text{healthy}] - 0.1 \sum_{i} a_i^2
$$

The humanoid is "healthy" when its root Z position is within [1.0, 2.0]. The episode terminates when the humanoid becomes unhealthy.

### Constructor Parameters

```python
HumanoidEnv(
    render_mode=None,
    forward_reward_weight=1.25,
    ctrl_cost_weight=0.1,
    healthy_reward=5.0,
    healthy_z_min=1.0,
    healthy_z_max=2.0,
    reset_noise_scale=0.005,
)
```

### MJCF Features Used

The humanoid exercises advanced MJCF features:

- **Free joint** -- 6DOF root with quaternion orientation
- **Multi-joint bodies** -- hip has 3 hinge joints, abdomen has 2, shoulders have 2
- **Arbitrary joint axes** -- joints can rotate around any axis
- **Sphere geoms** -- used for feet and hands
- **Quaternion body poses** -- full 3D orientation support
- **Intermediate body decomposition** -- multi-DOF joints are decomposed into kinematic chains with intermediate bodies

---

## CheetahRace-v0

N independent cheetahs placed in a shared physics world. They can physically interact (collide, push each other). All agents share a flat observation/action space, making this compatible with standard single-policy training.

```python
import joltgym
env = joltgym.make("JoltGym/CheetahRace-v0", num_agents=4)
```

![CheetahRace](../assets/race.png)

### Specification

| Property | Value |
|---|---|
| Observation Space | `Box(-inf, inf, (N*17,))` |
| Action Space | `Box(-1.0, 1.0, (N*6,))` |
| Reward | sum of all agents' forward rewards |
| Agent Spacing | 3.0 (Y-axis offset between agents) |

### Flat Layout

Observations and actions are flat concatenations of per-agent vectors:

```
obs  = [agent_0_obs(17), agent_1_obs(17), ..., agent_N_obs(17)]
act  = [agent_0_act(6),  agent_1_act(6),  ..., agent_N_act(6)]
```

### Per-Agent Access

Use the built-in helper methods to split flat arrays:

```python
per_agent_obs = env.get_per_agent_obs(obs)       # (N, 17)
per_agent_act = env.get_per_agent_actions(action)  # (N, 6)
```

### Step Info

```python
info["per_agent_reward"]  # np.array of shape (N,)
info["agent_0_x"]         # X position of agent 0
info["agent_0_xvel"]      # X velocity of agent 0
# ... same for agent_1, agent_2, etc.
```

### Constructor Parameters

```python
CheetahRaceEnv(
    num_agents=2,
    render_mode=None,
    agent_spacing=3.0,          # Y-axis offset between agents
    forward_reward_weight=1.0,
    ctrl_cost_weight=0.1,
    reset_noise_scale=0.1,
)
```
