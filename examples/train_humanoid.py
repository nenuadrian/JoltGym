#!/usr/bin/env python3
"""Train a Humanoid agent with PPO.

Usage:
    python examples/train_humanoid.py
    python examples/train_humanoid.py --timesteps 2000000
"""

import argparse
import os
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback
import joltgym


def make_env(seed=0):
    def _init():
        env = joltgym.make("JoltGym/Humanoid-v0")
        env.reset(seed=seed)
        return env
    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--eval-freq", type=int, default=25_000)
    args = parser.parse_args()

    print(f"Humanoid PPO Training")
    print(f"  Parallel envs:   {args.n_envs}")
    print(f"  Total timesteps: {args.timesteps:,}")
    print("=" * 50)

    # Create training envs
    train_envs = SubprocVecEnv([make_env(seed=i) for i in range(args.n_envs)])

    # Create eval env
    eval_env = SubprocVecEnv([make_env(seed=100)])

    # PPO with tuned hyperparameters for Humanoid
    model = PPO(
        "MlpPolicy",
        train_envs,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=512,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
        ),
        verbose=1,
        tensorboard_log="logs/humanoid_ppo",
    )

    # Eval callback
    os.makedirs("logs/humanoid_ppo", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="models/",
        log_path="logs/humanoid_ppo",
        eval_freq=max(args.eval_freq // args.n_envs, 1),
        n_eval_episodes=5,
        deterministic=True,
        verbose=1,
    )

    # Train
    model.learn(total_timesteps=args.timesteps, callback=eval_callback, progress_bar=False)

    # Save final model
    model.save("models/humanoid_ppo")
    print(f"\nModel saved to models/humanoid_ppo")

    train_envs.close()
    eval_env.close()

    # Quick evaluation
    print("\n" + "=" * 50)
    print("Evaluating trained policy...")
    env = joltgym.make("JoltGym/Humanoid-v0")
    obs, _ = env.reset(seed=42)
    total_reward = 0
    steps = 0

    for step in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, term, trunc, info = env.step(action)
        total_reward += reward
        steps += 1
        if step % 100 == 0:
            print(f"  Step {step:4d}: z={info['z_position']:.3f} "
                  f"x={info['x_position']:.3f} reward={reward:.3f}")
        if term or trunc:
            break

    print(f"\nEpisode length: {steps}, Total reward: {total_reward:.1f}")
    env.close()


if __name__ == "__main__":
    main()
