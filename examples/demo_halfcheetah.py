#!/usr/bin/env python3
"""Demo: HalfCheetah with random actions.

Usage:
    python examples/demo_halfcheetah.py
"""

import joltgym


def main():
    print("JoltGym HalfCheetah Demo")
    print("=" * 40)

    env = joltgym.make("JoltGym/HalfCheetah-v0")

    obs, info = env.reset(seed=42)
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")

    total_reward = 0
    num_steps = 1000

    for step in range(num_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if step % 100 == 0:
            print(f"  Step {step:4d} | reward={reward:7.3f} | "
                  f"x_pos={info['x_position']:7.3f} | "
                  f"x_vel={info['x_velocity']:7.3f}")

        if terminated or truncated:
            print(f"  Episode ended at step {step}")
            obs, info = env.reset()
            total_reward = 0

    print(f"\nTotal reward over {num_steps} steps: {total_reward:.3f}")
    env.close()
    print("Done!")


if __name__ == "__main__":
    main()
