#!/usr/bin/env python3
"""Unified benchmark: train, record, and measure throughput for JoltGym environments.

Combines training, video/GIF recording, and throughput benchmarking into a
single entry point for all environments.

Usage:
    # Full pipeline (train + record + throughput benchmark)
    python examples/benchmark.py halfcheetah
    python examples/benchmark.py humanoid
    python examples/benchmark.py cheetah_race --agents 4

    # Skip training, use an existing model
    python examples/benchmark.py halfcheetah --no-train --model models/halfcheetah_ppo

    # Only record (GIF, MP4, or both)
    python examples/benchmark.py humanoid --record-only --model models/humanoid_ppo --format gif
    python examples/benchmark.py halfcheetah --record-only --format both

    # Only benchmark throughput
    python examples/benchmark.py halfcheetah --throughput-only --num-envs 256

    # Custom training + recording options
    python examples/benchmark.py humanoid --timesteps 2000000 --record-steps 500 --fps 30
"""

import argparse
import os
import sys
import time
import json
from pathlib import Path

import numpy as np

import joltgym

# ─── Paths ────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
VIDEOS_DIR = ROOT / "videos"
LOGS_DIR = ROOT / "logs"
RESULTS_DIR = ROOT / "benchmarks"

# ─── Environment registry ────────────────────────────────────────────────────

ENV_CONFIGS = {
    "halfcheetah": {
        "env_id": "JoltGym/HalfCheetah-v0",
        "obs_dim": 17,
        "act_dim": 6,
        "asset": "half_cheetah.xml",
        "train_defaults": {
            "timesteps": 500_000,
            "n_envs": 8,
            "batch_size": 64,
            "net_arch": None,
        },
    },
    "humanoid": {
        "env_id": "JoltGym/Humanoid-v0",
        "obs_dim": 45,
        "act_dim": 17,
        "asset": "humanoid.xml",
        "train_defaults": {
            "timesteps": 500_000,
            "n_envs": 8,
            "batch_size": 512,
            "net_arch": [256, 256],
        },
    },
    "cheetah_race": {
        "env_id": "JoltGym/CheetahRace-v0",
        "obs_dim": 17,  # per agent
        "act_dim": 6,   # per agent
        "asset": "half_cheetah.xml",
        "train_defaults": {
            "timesteps": 300_000,
            "n_envs": 4,
            "batch_size": 64,
            "net_arch": None,
        },
    },
}

# ═══════════════════════════════════════════════════════════════════════════════
#  Training
# ═══════════════════════════════════════════════════════════════════════════════

def train(env_name, args):
    """Train a PPO agent and return the path to the saved model."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
    from stable_baselines3.common.callbacks import EvalCallback

    cfg = ENV_CONFIGS[env_name]
    defaults = cfg["train_defaults"]
    timesteps = args.timesteps or defaults["timesteps"]
    n_envs = args.n_envs or defaults["n_envs"]
    batch_size = defaults["batch_size"]
    net_arch = defaults["net_arch"]

    save_path = str(MODELS_DIR / f"{env_name}_ppo")
    log_dir = str(LOGS_DIR / f"{env_name}_ppo")

    env_kwargs = {}
    if env_name == "cheetah_race":
        env_kwargs["num_agents"] = args.agents

    def make_env(rank):
        def _init():
            env = joltgym.make(cfg["env_id"], **env_kwargs)
            env.reset(seed=args.seed + rank)
            return env
        return _init

    print(f"\n{'='*60}")
    print(f"  TRAINING: {env_name}")
    print(f"  Environment: {cfg['env_id']}")
    print(f"  Timesteps:   {timesteps:,}")
    print(f"  Parallel:    {n_envs} envs")
    if env_name == "cheetah_race":
        print(f"  Agents:      {args.agents}")
    print(f"{'='*60}\n")

    train_env = VecMonitor(SubprocVecEnv([make_env(i) for i in range(n_envs)]))
    eval_env = VecMonitor(SubprocVecEnv([make_env(100)]))

    policy_kwargs = {}
    extra_kwargs = {}
    if net_arch is not None:
        policy_kwargs["net_arch"] = dict(pi=net_arch, vf=net_arch)
    if env_name == "humanoid":
        extra_kwargs["vf_coef"] = 0.5
        extra_kwargs["max_grad_norm"] = 0.5

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=batch_size,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        verbose=1,
        seed=args.seed,
        tensorboard_log=log_dir,
        policy_kwargs=policy_kwargs or {},
        **extra_kwargs,
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(MODELS_DIR),
        log_path=log_dir,
        eval_freq=max(10_000 // n_envs, 1),
        n_eval_episodes=5,
        deterministic=True,
    )

    t0 = time.time()
    model.learn(total_timesteps=timesteps, callback=eval_callback)
    train_time = time.time() - t0

    model.save(save_path)
    print(f"\nModel saved to {save_path}")
    print(f"Training wall time: {train_time:.1f}s")

    train_env.close()
    eval_env.close()

    return save_path, train_time


# ═══════════════════════════════════════════════════════════════════════════════
#  Throughput benchmark
# ═══════════════════════════════════════════════════════════════════════════════

def benchmark_throughput(env_name, args):
    """Measure environment throughput (steps/second) at various batch sizes."""
    cfg = ENV_CONFIGS[env_name]
    asset_path = str(ROOT / "python" / "joltgym" / "assets" / cfg["asset"])

    # For cheetah_race we benchmark the base halfcheetah vectorized env
    # since the WorldPool operates on single-agent envs.
    from joltgym.vector.jolt_vector_env import JoltVectorEnv

    batch_sizes = [1, 8, 64, 256] if args.num_envs is None else [args.num_envs]
    num_steps = args.bench_steps

    print(f"\n{'='*60}")
    print(f"  THROUGHPUT BENCHMARK: {env_name}")
    print(f"  Steps per batch: {num_steps:,}")
    print(f"{'='*60}\n")

    results = {}
    for n in batch_sizes:
        try:
            envs = JoltVectorEnv(n, model_path=asset_path)
            envs.reset(seed=42)

            actions = np.random.uniform(-1, 1, (n, cfg["act_dim"])).astype(np.float32)

            # Warmup
            for _ in range(100):
                envs.step(actions)

            t0 = time.time()
            for _ in range(num_steps):
                envs.step(actions)
            elapsed = time.time() - t0

            total = n * num_steps
            sps = total / elapsed
            results[n] = sps
            print(f"  {n:>4} envs: {sps:>10,.0f} steps/sec  ({elapsed:.2f}s)")
            envs.close()
        except Exception as e:
            print(f"  {n:>4} envs: FAILED ({e})")
            results[n] = 0

    print()
    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  Recording — HalfCheetah
# ═══════════════════════════════════════════════════════════════════════════════

# Cheetah FK data
_BODY_LOCAL_POS = {
    "torso":  (None, 0.0, 0.0), "bthigh": (0, -0.5, 0.0),
    "bshin":  (1, 0.16, -0.25), "bfoot":  (2, -0.28, -0.14),
    "fthigh": (0, 0.5, 0.0),    "fshin":  (4, -0.14, -0.24),
    "ffoot":  (5, 0.13, -0.18),
}
_CAPSULE_ENDPOINTS = {
    "torso": [(-0.5, 0.0), (0.5, 0.0)],
    "bthigh": [(-0.145, 0.0), (0.145, 0.0)],
    "bshin": [(-0.15, 0.0), (0.15, 0.0)],
    "bfoot": [(-0.094, 0.0), (0.094, 0.0)],
    "fthigh": [(-0.133, 0.0), (0.133, 0.0)],
    "fshin": [(-0.106, 0.0), (0.106, 0.0)],
    "ffoot": [(-0.07, 0.0), (0.07, 0.0)],
}
_GEOM_ANGLES = {
    "torso": 0.0, "bthigh": -3.8, "bshin": -2.03, "bfoot": -0.27,
    "fthigh": 0.52, "fshin": -0.6, "ffoot": -0.6,
}
_BODY_NAMES = ["torso", "bthigh", "bshin", "bfoot", "fthigh", "fshin", "ffoot"]
_JOINT_NAMES = ["bthigh", "bshin", "bfoot", "fthigh", "fshin", "ffoot"]
_BODY_COLORS = {
    "torso": "#2196F3", "bthigh": "#F44336", "bshin": "#FF8A80",
    "bfoot": "#FF8A80", "fthigh": "#4CAF50", "fshin": "#A5D6A7", "ffoot": "#A5D6A7",
}
_AGENT_PALETTES = [
    {"torso": "#2196F3", "back": "#F44336", "front": "#4CAF50"},
    {"torso": "#FF9800", "back": "#9C27B0", "front": "#00BCD4"},
    {"torso": "#E91E63", "back": "#795548", "front": "#607D8B"},
    {"torso": "#FFEB3B", "back": "#3F51B5", "front": "#009688"},
]


def _cheetah_fk(root_x, root_z, root_angle, joint_angles):
    positions, orientations = {}, {}
    positions["torso"] = np.array([root_x, root_z])
    orientations["torso"] = root_angle
    angle_map = dict(zip(_JOINT_NAMES, joint_angles))
    for name in _BODY_NAMES[1:]:
        pi, lx, lz = _BODY_LOCAL_POS[name]
        pp = positions[_BODY_NAMES[pi]]
        pa = orientations[_BODY_NAMES[pi]]
        c, s = np.cos(pa), np.sin(pa)
        positions[name] = np.array([pp[0] + c*lx - s*lz, pp[1] + s*lx + c*lz])
        orientations[name] = pa + angle_map.get(name, 0.0)
    return positions, orientations


def _draw_cheetah(ax, root_x, root_z, root_angle, joint_angles, colors=None, label=None):
    if colors is None:
        colors = _BODY_COLORS
    positions, orientations = _cheetah_fk(root_x, root_z, root_angle, joint_angles)
    for name in _BODY_NAMES:
        pos, angle = positions[name], orientations[name]
        ta = angle + _GEOM_ANGLES[name]
        ep = _CAPSULE_ENDPOINTS[name]
        c, s = np.cos(ta), np.sin(ta)
        for i in range(len(ep) - 1):
            x0, z0 = ep[i]; x1, z1 = ep[i+1]
            ax.plot([pos[0]+c*x0-s*z0, pos[0]+c*x1-s*z1],
                    [pos[1]+s*x0+c*z0, pos[1]+s*x1+c*z1],
                    color=colors[name], linewidth=7, solid_capstyle="round", zorder=2)
        ax.plot(pos[0], pos[1], "o", color="white", markersize=3,
                markeredgecolor="black", markeredgewidth=0.5, zorder=3)
    if label:
        ax.text(root_x, root_z + 0.4, label, ha="center", fontsize=8,
                fontweight="bold", color=colors["torso"],
                bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.8))


def _get_agent_colors(agent_idx):
    p = _AGENT_PALETTES[agent_idx % len(_AGENT_PALETTES)]
    return {
        "torso": p["torso"],
        "bthigh": p["back"], "bshin": p["back"], "bfoot": p["back"],
        "fthigh": p["front"], "fshin": p["front"], "ffoot": p["front"],
    }


def _collect_halfcheetah(env, policy, steps, seed):
    obs, _ = env.reset(seed=seed)
    base_env = env.unwrapped
    frames, total_reward = [], 0.0

    for _ in range(steps):
        root_x = base_env._core.get_root_x()
        rz, ra, ja = float(obs[0]), float(obs[1]), obs[2:8].astype(float)
        frames.append((root_x, rz, ra, ja))

        if policy is not None:
            action, _ = policy.predict(obs, deterministic=True)
        else:
            action = env.action_space.sample()
        obs, reward, term, trunc, _ = env.step(action)
        total_reward += reward

    return frames, total_reward


def _collect_cheetah_race(env, policy, steps, seed, n_agents):
    obs, info = env.reset(seed=seed)
    frames, total_rewards = [], np.zeros(n_agents)

    for _ in range(steps):
        frame = []
        for i in range(n_agents):
            agent_obs = obs[i * 17 : (i + 1) * 17]
            rx = info.get(f"agent_{i}_x", 0.0)
            rz, ra = float(agent_obs[0]), float(agent_obs[1])
            ja = agent_obs[2:8].astype(float)
            frame.append((rx, rz, ra, ja))
        frames.append(frame)

        if policy is not None:
            action, _ = policy.predict(obs, deterministic=True)
        else:
            action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        if "per_agent_reward" in info:
            total_rewards += info["per_agent_reward"]

    return frames, total_rewards


# ═══════════════════════════════════════════════════════════════════════════════
#  Recording — Humanoid
# ═══════════════════════════════════════════════════════════════════════════════

_SKELETON_EDGES = [
    ("torso", "lwaist"), ("lwaist", "pelvis"),
    ("pelvis", "right_thigh"), ("right_thigh", "right_shin"), ("right_shin", "right_foot"),
    ("pelvis", "left_thigh"), ("left_thigh", "left_shin"), ("left_shin", "left_foot"),
    ("torso", "right_upper_arm"), ("right_upper_arm", "right_lower_arm"),
    ("torso", "left_upper_arm"), ("left_upper_arm", "left_lower_arm"),
]
_HEAD_OFFSET = np.array([0, 0, 0.19])
_EDGE_COLORS = {
    ("torso", "lwaist"): "#42A5F5", ("lwaist", "pelvis"): "#42A5F5",
    ("pelvis", "right_thigh"): "#EF5350", ("right_thigh", "right_shin"): "#EF5350",
    ("right_shin", "right_foot"): "#EF9A9A",
    ("pelvis", "left_thigh"): "#66BB6A", ("left_thigh", "left_shin"): "#66BB6A",
    ("left_shin", "left_foot"): "#A5D6A7",
    ("torso", "right_upper_arm"): "#FFA726", ("right_upper_arm", "right_lower_arm"): "#FFB74D",
    ("torso", "left_upper_arm"): "#AB47BC", ("left_upper_arm", "left_lower_arm"): "#CE93D8",
}


def _collect_humanoid(env, policy, steps, seed):
    obs, _ = env.reset(seed=seed)
    base_env = env.unwrapped
    body_names = base_env._core.get_body_names()
    name_to_idx = {name: i for i, name in enumerate(body_names)}

    frames, total_reward = [], 0.0

    for _ in range(steps):
        positions = base_env._core.get_body_positions()
        root_x = float(positions[0, 0])
        root_z = float(positions[0, 2])
        frames.append((positions.copy(), root_x, root_z))

        if policy is not None:
            action, _ = policy.predict(obs, deterministic=True)
        else:
            action = env.action_space.sample()
        obs, reward, term, trunc, _ = env.step(action)
        total_reward += reward
        if term or trunc:
            obs, _ = env.reset()

    return frames, total_reward, name_to_idx


# ═══════════════════════════════════════════════════════════════════════════════
#  Rendering
# ═══════════════════════════════════════════════════════════════════════════════

def _save_animation(anim, output_path, fps, formats):
    """Save animation as MP4, GIF, or both. Returns list of saved paths."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.animation import FFMpegWriter, PillowWriter

    saved = []
    base = str(output_path).rsplit(".", 1)[0]

    if "mp4" in formats:
        mp4_path = base + ".mp4"
        try:
            writer = FFMpegWriter(fps=fps, metadata={"title": "JoltGym Benchmark"})
            anim.save(mp4_path, writer=writer)
            saved.append(mp4_path)
        except Exception as e:
            print(f"  MP4 failed ({e}), skipping.")

    if "gif" in formats:
        gif_path = base + ".gif"
        writer = PillowWriter(fps=fps)
        anim.save(gif_path, writer=writer)
        saved.append(gif_path)

    return saved


def record(env_name, model_path, args):
    """Record video/GIF of a policy and return (saved_paths, eval_reward)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    cfg = ENV_CONFIGS[env_name]
    steps = args.record_steps
    fps = args.fps
    seed = args.seed

    # Determine output formats
    fmt = args.format
    formats = ["mp4", "gif"] if fmt == "both" else [fmt]

    # Load policy
    policy = None
    if model_path:
        from stable_baselines3 import PPO
        policy = PPO.load(model_path)
        policy_label = Path(model_path).stem
    else:
        policy_label = "random"

    output_base = VIDEOS_DIR / f"{env_name}_{policy_label}"
    os.makedirs(VIDEOS_DIR, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  RECORDING: {env_name}")
    print(f"  Policy:  {policy_label}")
    print(f"  Steps:   {steps}")
    print(f"  FPS:     {fps}")
    print(f"  Formats: {', '.join(formats)}")
    print(f"{'='*60}\n")

    # ── HalfCheetah ───────────────────────────────────────────────────────
    if env_name == "halfcheetah":
        env = joltgym.make(cfg["env_id"])
        frames, total_reward = _collect_halfcheetah(env, policy, steps, seed)
        env.close()

        fig, ax = plt.subplots(1, 1, figsize=(12, 4), dpi=100)
        fig.patch.set_facecolor("#E8E0D0")

        def animate(fi):
            ax.clear()
            root_x, root_z, root_angle, ja = frames[fi]
            _draw_cheetah(ax, root_x, root_z, root_angle, ja)
            ax.set_xlim(root_x - 2.5, root_x + 2.5)
            ax.set_ylim(-0.5, 1.5)
            ax.set_aspect("equal")
            ax.set_facecolor("#D4E6F1")
            ax.fill_between([root_x - 3, root_x + 3], -0.5, 0,
                            color="#C4A882", alpha=0.5, zorder=0)
            ax.set_title(f"JoltGym HalfCheetah  |  Step {fi}  |  "
                         f"x={root_x:.2f}  z={root_z:.2f}",
                         fontsize=11, fontfamily="monospace")
            ax.tick_params(labelsize=8)

        anim = FuncAnimation(fig, animate, frames=len(frames), interval=1000 // fps)
        saved = _save_animation(anim, output_base, fps, formats)
        plt.close()
        return saved, total_reward

    # ── Humanoid ──────────────────────────────────────────────────────────
    elif env_name == "humanoid":
        env = joltgym.make(cfg["env_id"])
        frames, total_reward, name_to_idx = _collect_humanoid(env, policy, steps, seed)
        env.close()

        view = args.view
        if view == "3d":
            fig = plt.figure(figsize=(10, 8), dpi=100)
            ax = fig.add_subplot(111, projection="3d")
        else:
            fig, ax = plt.subplots(1, 1, figsize=(12, 6), dpi=100)
        fig.patch.set_facecolor("#1a1a2e")

        def draw_skeleton(ax, positions, view, root_x):
            for (na, nb) in _SKELETON_EDGES:
                if na not in name_to_idx or nb not in name_to_idx:
                    continue
                pa = positions[name_to_idx[na]]
                pb = positions[name_to_idx[nb]]
                color = _EDGE_COLORS.get((na, nb), "#FFFFFF")
                if view == "3d":
                    ax.plot([pa[0], pb[0]], [pa[1], pb[1]], [pa[2], pb[2]],
                            color=color, linewidth=5, solid_capstyle="round", zorder=2)
                elif view == "front":
                    ax.plot([pa[1], pb[1]], [pa[2], pb[2]],
                            color=color, linewidth=6, solid_capstyle="round", zorder=2)
                else:
                    ax.plot([pa[0], pb[0]], [pa[2], pb[2]],
                            color=color, linewidth=6, solid_capstyle="round", zorder=2)

            for idx in range(len(positions)):
                p = positions[idx]
                if view == "3d":
                    ax.scatter(p[0], p[1], p[2], color="white", s=25, zorder=5,
                               edgecolors="#333", linewidths=0.5)
                elif view == "front":
                    ax.plot(p[1], p[2], "o", color="white", markersize=5,
                            markeredgecolor="#333", markeredgewidth=0.5, zorder=5)
                else:
                    ax.plot(p[0], p[2], "o", color="white", markersize=5,
                            markeredgecolor="#333", markeredgewidth=0.5, zorder=5)

            torso = positions[name_to_idx["torso"]]
            head = torso + _HEAD_OFFSET
            if view == "3d":
                ax.scatter(head[0], head[1], head[2], color="#FFD54F", s=120,
                           zorder=6, edgecolors="#333", linewidths=0.5)
            elif view == "front":
                ax.plot(head[1], head[2], "o", color="#FFD54F", markersize=12,
                        markeredgecolor="#333", markeredgewidth=1, zorder=6)
            else:
                ax.plot(head[0], head[2], "o", color="#FFD54F", markersize=12,
                        markeredgecolor="#333", markeredgewidth=1, zorder=6)

        def animate(fi):
            ax.clear()
            positions, root_x, root_z = frames[fi]

            if view == "3d":
                draw_skeleton(ax, positions, "3d", root_x)
                gx = np.linspace(root_x - 2, root_x + 2, 2)
                gy = np.linspace(-2, 2, 2)
                gx, gy = np.meshgrid(gx, gy)
                ax.plot_surface(gx, gy, np.zeros_like(gx), alpha=0.15, color="#8B7355")
                ax.set_xlim(root_x - 1.5, root_x + 1.5)
                ax.set_ylim(-1.5, 1.5)
                ax.set_zlim(-0.1, 2.2)
                ax.set_box_aspect([3, 3, 2.3])
                ax.view_init(elev=15, azim=-60 + fi * 0.3)
                ax.set_facecolor("#1a1a2e")
                ax.xaxis.pane.fill = False
                ax.yaxis.pane.fill = False
                ax.zaxis.pane.fill = False
            elif view == "front":
                draw_skeleton(ax, positions, "front", root_x)
                ax.fill_between([-2, 2], -0.1, 0, color="#8B7355", alpha=0.4)
                ax.axhline(y=0, color="#8B7355", linewidth=2)
                ax.set_xlim(-1.0, 1.0)
                ax.set_ylim(-0.2, 2.2)
                ax.set_aspect("equal")
                ax.set_facecolor("#16213e")
            else:
                draw_skeleton(ax, positions, "side", root_x)
                ax.fill_between([root_x - 3, root_x + 3], -0.1, 0,
                                color="#8B7355", alpha=0.4)
                ax.axhline(y=0, color="#8B7355", linewidth=2)
                ax.set_xlim(root_x - 1.5, root_x + 1.5)
                ax.set_ylim(-0.2, 2.2)
                ax.set_aspect("equal")
                ax.set_facecolor("#16213e")

            ax.set_title(
                f"JoltGym Humanoid  |  Step {fi}  |  x={root_x:.2f}  z={root_z:.2f}",
                fontsize=11, fontfamily="monospace", color="#e0e0e0", pad=10)
            ax.tick_params(colors="#888888", labelsize=8)

        anim = FuncAnimation(fig, animate, frames=len(frames), interval=1000 // fps)
        saved = _save_animation(anim, output_base, fps, formats)
        plt.close()
        return saved, total_reward

    # ── CheetahRace ───────────────────────────────────────────────────────
    elif env_name == "cheetah_race":
        n = args.agents
        env = joltgym.make(cfg["env_id"], num_agents=n)
        frames, total_rewards = _collect_cheetah_race(env, policy, steps, seed, n)
        env.close()

        agent_colors = [_get_agent_colors(i) for i in range(n)]
        fig_height = max(4, 2 + n * 1.2)
        fig, ax = plt.subplots(1, 1, figsize=(14, fig_height), dpi=100)
        fig.patch.set_facecolor("#E8E0D0")

        z_offset = 1.5

        def animate(fi):
            ax.clear()
            frame = frames[fi]
            xs = [f[0] for f in frame]
            cam_x = np.mean(xs)

            ax.fill_between([cam_x - 8, cam_x + 8], -1.5, 0,
                            color="#C4A882", alpha=0.5, zorder=0)
            ax.axhline(y=0, color="#8B7355", linewidth=2, zorder=1)

            for i, (rx, rz, ra, ja) in enumerate(frame):
                _draw_cheetah(ax, rx, rz + i * z_offset, ra, ja,
                              agent_colors[i], label=f"Agent {i}")

            for i in range(n):
                gz = i * z_offset
                ax.axhline(y=gz, color="#8B7355", linewidth=1.5, alpha=0.4, zorder=0)
                ax.fill_between([cam_x - 10, cam_x + 10],
                                gz - 0.3, gz, color="#C4A882", alpha=0.2, zorder=0)

            view_w = max(6, (max(xs) - min(xs)) + 6)
            ax.set_xlim(cam_x - view_w/2, cam_x + view_w/2)
            ax.set_ylim(-0.5, 1.0 + n * z_offset)
            ax.set_aspect("equal")
            ax.set_facecolor("#D4E6F1")

            score_parts = [f"Ag{i}: x={frame[i][0]:+.1f}" for i in range(n)]
            leader = max(range(n), key=lambda i: frame[i][0])
            score_parts.append(f"Lead: Ag{leader}")
            ax.set_title(
                f"JoltGym CheetahRace ({n} agents)  |  Step {fi}  |  "
                + "  ".join(score_parts),
                fontsize=10, fontfamily="monospace")
            ax.tick_params(labelsize=8)

        anim = FuncAnimation(fig, animate, frames=len(frames), interval=1000 // fps)
        saved = _save_animation(anim, output_base, fps, formats)
        plt.close()
        return saved, float(np.sum(total_rewards))


# ═══════════════════════════════════════════════════════════════════════════════
#  Evaluation (quick deterministic rollout)
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate(env_name, model_path, args):
    """Run a deterministic evaluation episode and return stats."""
    cfg = ENV_CONFIGS[env_name]
    env_kwargs = {}
    if env_name == "cheetah_race":
        env_kwargs["num_agents"] = args.agents

    policy = None
    if model_path:
        from stable_baselines3 import PPO
        policy = PPO.load(model_path)

    env = joltgym.make(cfg["env_id"], **env_kwargs)
    obs, info = env.reset(seed=args.seed)

    total_reward = 0.0
    steps = 0
    for step in range(1000):
        if policy is not None:
            action, _ = policy.predict(obs, deterministic=True)
        else:
            action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        total_reward += reward
        steps += 1
        if term or trunc:
            break

    env.close()
    return {"episode_reward": total_reward, "episode_length": steps}


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark JoltGym environments: train, record, and measure throughput.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  %(prog)s halfcheetah                                  # full pipeline
  %(prog)s humanoid --no-train --model models/humanoid_ppo
  %(prog)s cheetah_race --agents 4 --timesteps 1000000
  %(prog)s halfcheetah --record-only --format gif
  %(prog)s halfcheetah --throughput-only --num-envs 256
""")

    parser.add_argument("env", choices=list(ENV_CONFIGS.keys()),
                        help="Environment to benchmark")

    # Phase control
    phase = parser.add_argument_group("phase control")
    phase.add_argument("--no-train", action="store_true",
                       help="Skip training (use --model for recording/eval)")
    phase.add_argument("--no-record", action="store_true",
                       help="Skip recording")
    phase.add_argument("--no-throughput", action="store_true",
                       help="Skip throughput benchmark")
    phase.add_argument("--record-only", action="store_true",
                       help="Only record (implies --no-train --no-throughput)")
    phase.add_argument("--throughput-only", action="store_true",
                       help="Only benchmark throughput (implies --no-train --no-record)")
    phase.add_argument("--train-only", action="store_true",
                       help="Only train (implies --no-record --no-throughput)")

    # Training
    tr = parser.add_argument_group("training")
    tr.add_argument("--timesteps", type=int, default=None,
                    help="Total training timesteps (default: env-specific)")
    tr.add_argument("--n-envs", type=int, default=None,
                    help="Parallel training envs (default: env-specific)")
    tr.add_argument("--model", type=str, default=None,
                    help="Path to existing SB3 model (skip .zip extension)")

    # Recording
    rec = parser.add_argument_group("recording")
    rec.add_argument("--record-steps", type=int, default=300,
                     help="Simulation steps to record (default: 300)")
    rec.add_argument("--fps", type=int, default=20,
                     help="Output frame rate (default: 20)")
    rec.add_argument("--format", choices=["mp4", "gif", "both"], default="both",
                     help="Output format (default: both)")
    rec.add_argument("--view", choices=["side", "front", "3d"], default="side",
                     help="Humanoid camera view (default: side)")

    # Throughput
    tp = parser.add_argument_group("throughput")
    tp.add_argument("--num-envs", type=int, default=None,
                    help="Specific batch size to test (default: sweep 1,8,64,256)")
    tp.add_argument("--bench-steps", type=int, default=5000,
                    help="Steps per throughput test (default: 5000)")

    # Multi-agent
    ma = parser.add_argument_group("multi-agent")
    ma.add_argument("--agents", type=int, default=2,
                    help="Number of agents for CheetahRace (default: 2)")

    # General
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-json", type=str, default=None,
                        help="Save results summary to JSON file")

    args = parser.parse_args()

    # Resolve phase flags
    if args.record_only:
        args.no_train = True
        args.no_throughput = True
    if args.throughput_only:
        args.no_train = True
        args.no_record = True
    if args.train_only:
        args.no_record = True
        args.no_throughput = True

    do_train = not args.no_train
    do_record = not args.no_record
    do_throughput = not args.no_throughput
    env_name = args.env

    print(f"\n{'#'*60}")
    print(f"  JoltGym Benchmark — {env_name}")
    print(f"  Phases: ", end="")
    phases = []
    if do_train:
        phases.append("train")
    if do_record:
        phases.append("record")
    if do_throughput:
        phases.append("throughput")
    print(", ".join(phases) or "(none)")
    print(f"{'#'*60}")

    results = {"env": env_name, "seed": args.seed}
    model_path = args.model
    t_total = time.time()

    # ── Train ─────────────────────────────────────────────────────────────
    if do_train:
        save_path, train_time = train(env_name, args)
        model_path = save_path
        results["training"] = {
            "model_path": save_path,
            "wall_time_s": round(train_time, 1),
            "timesteps": args.timesteps or ENV_CONFIGS[env_name]["train_defaults"]["timesteps"],
        }

    # ── Evaluate ──────────────────────────────────────────────────────────
    if model_path and (do_train or do_record):
        print(f"\nEvaluating policy ({model_path})...")
        eval_stats = evaluate(env_name, model_path, args)
        results["evaluation"] = eval_stats
        print(f"  Episode reward: {eval_stats['episode_reward']:.1f}  "
              f"length: {eval_stats['episode_length']}")

    # ── Record ────────────────────────────────────────────────────────────
    if do_record:
        saved_paths, rec_reward = record(env_name, model_path, args)
        results["recording"] = {
            "files": saved_paths,
            "steps": args.record_steps,
            "reward_during_recording": round(rec_reward, 1),
        }
        for p in saved_paths:
            print(f"  Saved: {p}")

    # ── Throughput ────────────────────────────────────────────────────────
    if do_throughput:
        tp_results = benchmark_throughput(env_name, args)
        results["throughput"] = {str(k): round(v) for k, v in tp_results.items()}

    # ── Summary ───────────────────────────────────────────────────────────
    total_time = time.time() - t_total

    print(f"\n{'='*60}")
    print(f"  BENCHMARK SUMMARY: {env_name}")
    print(f"{'='*60}")
    if "training" in results:
        t = results["training"]
        print(f"  Training:    {t['timesteps']:,} steps in {t['wall_time_s']:.1f}s")
        print(f"               Model: {t['model_path']}")
    if "evaluation" in results:
        e = results["evaluation"]
        print(f"  Eval reward: {e['episode_reward']:.1f}  (length: {e['episode_length']})")
    if "recording" in results:
        r = results["recording"]
        print(f"  Recordings:  {', '.join(r['files'])}")
    if "throughput" in results:
        print(f"  Throughput:")
        for k, v in results["throughput"].items():
            print(f"    {k:>4} envs: {v:>10,} steps/sec")
    print(f"  Total time:  {total_time:.1f}s")
    print(f"{'='*60}\n")

    # ── Save JSON ─────────────────────────────────────────────────────────
    json_path = args.output_json
    if json_path is None:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        json_path = str(RESULTS_DIR / f"{env_name}_results.json")

    results["total_time_s"] = round(total_time, 1)
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {json_path}")


if __name__ == "__main__":
    main()
