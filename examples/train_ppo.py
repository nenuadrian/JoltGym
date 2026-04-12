#!/usr/bin/env python3
"""Train HalfCheetah with PPO using Stable-Baselines3.

Usage:
    python examples/train_ppo.py                     # Train for 500K steps
    python examples/train_ppo.py --timesteps 1000000  # Train for 1M steps
"""

import argparse
import os

import joltgym  # registers the env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--save-path", default="models/halfcheetah_ppo")
    parser.add_argument("--log-dir", default="logs/halfcheetah_ppo")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
    from stable_baselines3.common.callbacks import EvalCallback

    num_envs = 8

    def make_env(rank):
        def _init():
            env = joltgym.make("JoltGym/HalfCheetah-v0")
            env.reset(seed=args.seed + rank)
            return env
        return _init

    print(f"Training PPO on JoltGym/HalfCheetah-v0")
    print(f"  Timesteps: {args.timesteps:,}")
    print(f"  Parallel envs: {num_envs}")
    print(f"  Seed: {args.seed}")
    print("=" * 50)

    train_env = VecMonitor(SubprocVecEnv([make_env(i) for i in range(num_envs)]))
    eval_env = VecMonitor(SubprocVecEnv([make_env(100)]))

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        verbose=1,
        seed=args.seed,
        tensorboard_log=args.log_dir,
    )

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.dirname(args.save_path),
        log_path=args.log_dir,
        eval_freq=max(10_000 // num_envs, 1),
        n_eval_episodes=5,
        deterministic=True,
    )

    model.learn(total_timesteps=args.timesteps, callback=eval_callback)
    model.save(args.save_path)
    print(f"\nModel saved to {args.save_path}")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
