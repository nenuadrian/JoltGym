# Training

JoltGym includes training scripts for all three environments using PPO from [Stable-Baselines3](https://stable-baselines3.readthedocs.io/).

## Prerequisites

```bash
pip install stable-baselines3 tensorboard
```

## HalfCheetah Training

`examples/train_ppo.py`

Trains a PPO agent on HalfCheetah using 8 parallel SubprocVecEnv workers.

### Usage

```bash
# Default: 500K timesteps
python examples/train_ppo.py

# Custom timesteps
python examples/train_ppo.py --timesteps 1000000

# Custom save path and seed
python examples/train_ppo.py --save-path models/my_cheetah --seed 123
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--timesteps` | 500,000 | Total training timesteps |
| `--save-path` | `models/halfcheetah_ppo` | Where to save the model |
| `--log-dir` | `logs/halfcheetah_ppo` | TensorBoard log directory |
| `--seed` | 42 | Random seed |

### Hyperparameters

| Parameter | Value |
|---|---|
| Algorithm | PPO |
| Policy | MlpPolicy |
| Learning rate | 3e-4 |
| n_steps | 2048 |
| Batch size | 64 |
| Epochs | 10 |
| Gamma | 0.99 |
| GAE lambda | 0.95 |
| Clip range | 0.2 |
| Entropy coefficient | 0.0 |
| Parallel environments | 8 |
| Evaluation frequency | Every 10K steps |

---

## Humanoid Training

`examples/train_humanoid.py`

Trains a PPO agent on the 3D Humanoid with a larger network architecture.

### Usage

```bash
# Default: 500K timesteps
python examples/train_humanoid.py

# Longer training (recommended for humanoid)
python examples/train_humanoid.py --timesteps 2000000
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--timesteps` | 500,000 | Total training timesteps |
| `--n-envs` | 8 | Number of parallel environments |
| `--eval-freq` | 25,000 | Evaluation frequency (timesteps) |

### Hyperparameters

The humanoid uses a larger policy network than the cheetah:

| Parameter | Value |
|---|---|
| Policy network | `[256, 256]` (both pi and vf) |
| Batch size | 512 |
| Value function coefficient | 0.5 |
| Max gradient norm | 0.5 |
| Other parameters | Same as HalfCheetah |

After training, the script runs a quick 1000-step evaluation and reports the episode length and total reward.

---

## Multi-Agent Training

`examples/train_multiagent.py`

Trains N cheetahs racing in a shared physics world using parameter sharing (all agents use the same PPO policy).

### Usage

```bash
# Default: 2 agents, 300K steps
python examples/train_multiagent.py

# 4-agent race, 1M steps
python examples/train_multiagent.py --agents 4 --timesteps 1000000
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--agents` | 2 | Number of cheetahs per race |
| `--timesteps` | 300,000 | Total training timesteps |
| `--save-path` | `models/cheetah_race_ppo` | Model save path |
| `--log-dir` | `logs/cheetah_race` | TensorBoard log directory |
| `--seed` | 42 | Random seed |

### How It Works

- All agents share one PPO policy (parameter sharing)
- 4 parallel race instances run as SubprocVecEnv
- The observation/action spaces are flat concatenations of all agents
- The total reward is the sum of all agents' forward rewards
- After training, a quick evaluation prints per-agent positions and identifies the winner

---

## Monitoring Training

All training scripts log to TensorBoard:

```bash
tensorboard --logdir logs/
```

This shows:

- Episode reward over time
- Episode length
- Policy loss, value loss, entropy
- Evaluation results (every `eval_freq` steps)

## Output Structure

After training, the following files are created:

```
models/
  |-- halfcheetah_ppo.zip       # Final trained model
  |-- best_model.zip            # Best model (by eval reward)
  +-- humanoid_ppo.zip
logs/
  |-- halfcheetah_ppo/
  |   +-- events.out.tfevents.* # TensorBoard logs
  |-- humanoid_ppo/
  +-- cheetah_race/
```
