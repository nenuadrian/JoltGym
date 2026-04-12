#!/usr/bin/env python3
"""Demo: Vectorized HalfCheetah benchmark.

Usage:
    python examples/demo_vectorized.py
"""

import os
import time
import numpy as np


def main():
    num_envs = 64
    num_steps = 10000

    print(f"JoltGym Vectorized Benchmark")
    print(f"  Environments: {num_envs}")
    print(f"  Steps: {num_steps}")
    print("=" * 40)

    asset_path = os.path.join(os.path.dirname(__file__),
                              "..", "python", "joltgym", "assets", "half_cheetah.xml")

    from joltgym.vector.jolt_vector_env import JoltVectorEnv
    envs = JoltVectorEnv(num_envs, model_path=asset_path)

    obs, info = envs.reset(seed=42)
    print(f"Observation shape: {obs.shape}")

    start = time.time()
    for step in range(num_steps):
        actions = np.random.uniform(-1, 1, (num_envs, 6)).astype(np.float32)
        obs, rewards, terms, truncs, infos = envs.step(actions)
    elapsed = time.time() - start

    total_steps = num_envs * num_steps
    fps = total_steps / elapsed

    print(f"\nResults:")
    print(f"  Total env-steps: {total_steps:,}")
    print(f"  Wall time: {elapsed:.2f}s")
    print(f"  Throughput: {fps:,.0f} env-steps/second")
    print("Done!")


if __name__ == "__main__":
    main()
