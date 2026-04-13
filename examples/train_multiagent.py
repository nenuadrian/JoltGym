#!/usr/bin/env python3
"""Multi-agent CheetahRace training with PPO.

Trains N cheetahs racing in a shared physics world. All agents share one
PPO policy (parameter sharing) — the same network controls every cheetah.
Because they're in the same PhysicsWorld, they can physically interact
(collide, push each other).

Usage:
    python examples/train_multiagent.py                      # 2 agents, 300K steps
    python examples/train_multiagent.py --agents 4 --timesteps 1000000
"""

import argparse
import os
import time
import numpy as np

import joltgym


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agents", type=int, default=2)
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--save-path", default="models/cheetah_race_ppo")
    parser.add_argument("--log-dir", default="logs/cheetah_race")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
    from stable_baselines3.common.callbacks import EvalCallback

    num_agents = args.agents
    num_train_envs = 4  # parallel race instances

    def make_env(rank):
        def _init():
            env = joltgym.make("JoltGym/CheetahRace-v0", num_agents=num_agents)
            env.reset(seed=args.seed + rank)
            return env
        return _init

    print(f"Multi-Agent CheetahRace Training")
    print(f"  Agents per race: {num_agents}")
    print(f"  Parallel races:  {num_train_envs}")
    print(f"  Total timesteps: {args.timesteps:,}")
    print(f"  Policy: shared MLP (all agents use same network)")
    print("=" * 55)

    train_env = VecMonitor(SubprocVecEnv([make_env(i) for i in range(num_train_envs)]))
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
        eval_freq=max(10_000 // num_train_envs, 1),
        n_eval_episodes=5,
        deterministic=True,
    )

    model.learn(total_timesteps=args.timesteps, callback=eval_callback)
    model.save(args.save_path)
    print(f"\nModel saved to {args.save_path}")

    # --- Quick evaluation ---
    print(f"\n{'='*55}")
    print("Evaluating trained policy on a single race...")
    env = joltgym.make("JoltGym/CheetahRace-v0", num_agents=num_agents)
    obs, info = env.reset(seed=0)

    total_rewards = np.zeros(num_agents)
    for step in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, term, trunc, info = env.step(action)
        total_rewards += info["per_agent_reward"]

        if step % 200 == 199:
            positions = [info[f"agent_{i}_x"] for i in range(num_agents)]
            print(f"  Step {step+1:4d} | " +
                  " | ".join(f"Agent {i}: x={positions[i]:6.2f}" for i in range(num_agents)))

    print(f"\nFinal rewards: {total_rewards}")
    winner = np.argmax(total_rewards)
    print(f"Winner: Agent {winner} with reward {total_rewards[winner]:.1f}")

    env.close()
    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
