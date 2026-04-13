# Unified Benchmark

`examples/benchmark.py` combines training, video/GIF recording, and throughput benchmarking into a single script for all JoltGym environments.

## Prerequisites

```bash
pip install stable-baselines3 tensorboard matplotlib
```

## Quick Start

```bash
# Full pipeline for any environment: train + record + benchmark
python examples/benchmark.py halfcheetah
python examples/benchmark.py humanoid
python examples/benchmark.py cheetah_race --agents 4
```

## Phase Control

By default, all three phases run (train, record, throughput). Use flags to run specific phases:

```bash
# Skip training, use an existing model
python examples/benchmark.py halfcheetah --no-train --model models/halfcheetah_ppo

# Only record (GIF, MP4, or both)
python examples/benchmark.py humanoid --record-only --model models/humanoid_ppo --format gif

# Only benchmark throughput
python examples/benchmark.py halfcheetah --throughput-only --num-envs 256

# Only train (no recording or benchmarking)
python examples/benchmark.py humanoid --train-only --timesteps 2000000
```

| Flag | Effect |
|---|---|
| `--no-train` | Skip training phase |
| `--no-record` | Skip recording phase |
| `--no-throughput` | Skip throughput benchmark |
| `--record-only` | Only record (implies `--no-train --no-throughput`) |
| `--throughput-only` | Only benchmark (implies `--no-train --no-record`) |
| `--train-only` | Only train (implies `--no-record --no-throughput`) |

## Output Formats

Recording supports MP4, GIF, or both:

```bash
python examples/benchmark.py halfcheetah --record-only --format mp4
python examples/benchmark.py halfcheetah --record-only --format gif
python examples/benchmark.py halfcheetah --record-only --format both  # default
```

## Arguments Reference

### Training

| Argument | Default | Description |
|---|---|---|
| `--timesteps` | env-specific | Total training timesteps |
| `--n-envs` | env-specific | Parallel training environments |
| `--model` | None | Path to existing SB3 model |

### Recording

| Argument | Default | Description |
|---|---|---|
| `--record-steps` | 300 | Simulation steps to record |
| `--fps` | 20 | Output frame rate |
| `--format` | `both` | Output format: `mp4`, `gif`, or `both` |
| `--view` | `side` | Humanoid camera view: `side`, `front`, or `3d` |

### Throughput

| Argument | Default | Description |
|---|---|---|
| `--num-envs` | sweep | Specific batch size (default: sweeps 1, 8, 64, 256) |
| `--bench-steps` | 5,000 | Steps per throughput test |

### Multi-Agent

| Argument | Default | Description |
|---|---|---|
| `--agents` | 2 | Number of agents for CheetahRace |

### General

| Argument | Default | Description |
|---|---|---|
| `--seed` | 42 | Random seed |
| `--output-json` | `benchmarks/<env>_results.json` | Save results to JSON |

## Output Structure

```
models/
  +-- <env>_ppo.zip              # Trained model
videos/
  +-- <env>_<policy>.mp4         # MP4 recording
  +-- <env>_<policy>.gif         # GIF recording
benchmarks/
  +-- <env>_results.json         # Results summary (JSON)
logs/
  +-- <env>_ppo/                 # TensorBoard logs
```

## Results JSON

Each run produces a JSON summary in `benchmarks/`:

```json
{
  "env": "halfcheetah",
  "seed": 42,
  "training": {
    "model_path": "models/halfcheetah_ppo",
    "wall_time_s": 120.5,
    "timesteps": 500000
  },
  "evaluation": {
    "episode_reward": 1234.5,
    "episode_length": 1000
  },
  "recording": {
    "files": ["videos/halfcheetah_halfcheetah_ppo.mp4", "videos/halfcheetah_halfcheetah_ppo.gif"],
    "steps": 300,
    "reward_during_recording": 456.7
  },
  "throughput": {
    "1": 11935,
    "8": 33292,
    "64": 64503,
    "256": 73606
  },
  "total_time_s": 185.3
}
```

## Individual Scripts

The original standalone scripts are still available for focused use:

- [Training scripts](training.md) (`train_ppo.py`, `train_humanoid.py`, `train_multiagent.py`)
- [Recording scripts](recording.md) (`record_video.py`, `record_humanoid.py`, `record_race.py`)
