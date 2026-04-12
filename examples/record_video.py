#!/usr/bin/env python3
"""Record a video of a trained (or random) HalfCheetah agent.

Renders a 2D skeletal visualization of the cheetah using matplotlib,
saves as MP4 video.

Usage:
    python examples/record_video.py                          # Random policy
    python examples/record_video.py --model models/halfcheetah_ppo  # Trained model
    python examples/record_video.py --steps 500 --fps 30     # Custom settings
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

import joltgym


# HalfCheetah kinematic chain (from MJCF body positions)
# Each link: (parent_idx, local_x, local_z)
# Body tree: torso(0) -> bthigh(1) -> bshin(2) -> bfoot(3)
#            torso(0) -> fthigh(4) -> fshin(5) -> ffoot(6)
BODY_LOCAL_POS = {
    "torso":  (None,    0.0,    0.0),
    "bthigh": (0,      -0.5,    0.0),
    "bshin":  (1,       0.16,  -0.25),
    "bfoot":  (2,      -0.28,  -0.14),
    "fthigh": (0,       0.5,    0.0),
    "fshin":  (4,      -0.14,  -0.24),
    "ffoot":  (5,       0.13,  -0.18),
}

# Capsule visual endpoints (local to body, from MJCF geom fromto/pos+size)
CAPSULE_ENDPOINTS = {
    "torso":  [(-0.5, 0.0), (0.5, 0.0)],
    "bthigh": [(-0.145, 0.0), (0.145, 0.0)],
    "bshin":  [(-0.15, 0.0), (0.15, 0.0)],
    "bfoot":  [(-0.094, 0.0), (0.094, 0.0)],
    "fthigh": [(-0.133, 0.0), (0.133, 0.0)],
    "fshin":  [(-0.106, 0.0), (0.106, 0.0)],
    "ffoot":  [(-0.07, 0.0), (0.07, 0.0)],
}

# Capsule geom axis angles (from MJCF axisangle Y component)
GEOM_ANGLES = {
    "torso":  0.0,
    "bthigh": -3.8,
    "bshin":  -2.03,
    "bfoot":  -0.27,
    "fthigh": 0.52,
    "fshin":  -0.6,
    "ffoot":  -0.6,
}

BODY_COLORS = {
    "torso":  "#2196F3",
    "bthigh": "#F44336",
    "bshin":  "#FF8A80",
    "bfoot":  "#FF8A80",
    "fthigh": "#4CAF50",
    "fshin":  "#A5D6A7",
    "ffoot":  "#A5D6A7",
}

BODY_NAMES = ["torso", "bthigh", "bshin", "bfoot", "fthigh", "fshin", "ffoot"]
JOINT_NAMES = ["bthigh", "bshin", "bfoot", "fthigh", "fshin", "ffoot"]

CAPSULE_RADIUS = 0.046


def forward_kinematics(root_x, root_z, root_angle, joint_angles):
    """Compute world positions and orientations of all bodies."""
    positions = {}
    orientations = {}

    # Torso
    positions["torso"] = np.array([root_x, root_z])
    orientations["torso"] = root_angle

    angle_map = dict(zip(JOINT_NAMES, joint_angles))

    for name in BODY_NAMES[1:]:
        parent_idx, lx, lz = BODY_LOCAL_POS[name]
        parent_name = BODY_NAMES[parent_idx]
        parent_pos = positions[parent_name]
        parent_angle = orientations[parent_name]

        # Rotate local offset by parent orientation
        c, s = np.cos(parent_angle), np.sin(parent_angle)
        wx = parent_pos[0] + c * lx - s * lz
        wz = parent_pos[1] + s * lx + c * lz

        # This body's orientation = parent + joint angle
        joint_angle = angle_map.get(name, 0.0)
        body_angle = parent_angle + joint_angle

        positions[name] = np.array([wx, wz])
        orientations[name] = body_angle

    return positions, orientations


def draw_cheetah(ax, root_x, root_z, root_angle, joint_angles):
    """Draw the cheetah skeleton on the given axes."""
    positions, orientations = forward_kinematics(
        root_x, root_z, root_angle, joint_angles
    )

    artists = []
    for name in BODY_NAMES:
        pos = positions[name]
        angle = orientations[name]
        geom_angle = GEOM_ANGLES[name]
        total_angle = angle + geom_angle

        # Capsule endpoints in world frame
        endpoints = CAPSULE_ENDPOINTS[name]
        c, s = np.cos(total_angle), np.sin(total_angle)
        for i in range(len(endpoints) - 1):
            x0, z0 = endpoints[i]
            x1, z1 = endpoints[i + 1]
            wx0 = pos[0] + c * x0 - s * z0
            wz0 = pos[1] + s * x0 + c * z0
            wx1 = pos[0] + c * x1 - s * z1
            wz1 = pos[1] + s * x1 + c * z1

            line, = ax.plot([wx0, wx1], [wz0, wz1],
                           color=BODY_COLORS[name], linewidth=8,
                           solid_capstyle="round", zorder=2)
            artists.append(line)

        # Joint dot
        dot, = ax.plot(pos[0], pos[1], "o", color="white",
                      markersize=4, markeredgecolor="black",
                      markeredgewidth=0.5, zorder=3)
        artists.append(dot)

    # Ground line
    ground = ax.axhline(y=0, color="#8B7355", linewidth=3, zorder=0)
    artists.append(ground)

    return artists


def obs_to_state(obs, root_x):
    """Extract state from observation vector.

    obs layout (17 dims): qpos[1:] ++ qvel
    qpos[1:] = [rootz, rooty, bthigh, bshin, bfoot, fthigh, fshin, ffoot]
    """
    root_z = obs[0]
    root_angle = obs[1]
    joint_angles = obs[2:8]
    return root_x, root_z, root_angle, joint_angles


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None,
                        help="Path to trained SB3 model (omit for random policy)")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--output", type=str, default="videos/halfcheetah.mp4")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load model if provided
    policy = None
    if args.model:
        from stable_baselines3 import PPO
        policy = PPO.load(args.model)
        print(f"Loaded model from {args.model}")
    else:
        print("No model specified — using random policy")

    env = joltgym.make("JoltGym/HalfCheetah-v0")
    # Unwrap to access _core through any gymnasium wrappers
    base_env = env.unwrapped
    obs, info = env.reset(seed=args.seed)

    # Collect trajectory
    print(f"Collecting {args.steps} steps...")
    frames = []
    total_reward = 0
    root_x = 0.0

    for step in range(args.steps):
        root_x = base_env._core.get_root_x()
        state = obs_to_state(obs, root_x)
        frames.append(state)

        if policy is not None:
            action, _ = policy.predict(obs, deterministic=True)
        else:
            action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

    env.close()
    print(f"Total reward: {total_reward:.1f}")

    # Render video
    print(f"Rendering {len(frames)} frames...")

    fig, ax = plt.subplots(1, 1, figsize=(12, 4), dpi=100)
    fig.patch.set_facecolor("#E8E0D0")

    def animate(frame_idx):
        ax.clear()
        root_x, root_z, root_angle, joint_angles = frames[frame_idx]

        draw_cheetah(ax, root_x, root_z, root_angle, joint_angles)

        # Camera follows the cheetah
        ax.set_xlim(root_x - 2.5, root_x + 2.5)
        ax.set_ylim(-0.5, 1.5)
        ax.set_aspect("equal")
        ax.set_facecolor("#D4E6F1")

        # Ground fill
        ax.fill_between([root_x - 3, root_x + 3], -0.5, 0,
                        color="#C4A882", alpha=0.5, zorder=0)

        # Info text
        ax.set_title(f"JoltGym HalfCheetah  |  Step {frame_idx}  |  "
                     f"x={root_x:.2f}  z={root_z:.2f}",
                     fontsize=11, fontfamily="monospace")
        ax.set_xlabel("x position", fontsize=9)
        ax.tick_params(labelsize=8)

    anim = FuncAnimation(fig, animate, frames=len(frames), interval=1000 // args.fps)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # Try FFmpeg first, fall back to Pillow (GIF)
    try:
        writer = FFMpegWriter(fps=args.fps, metadata={"title": "JoltGym HalfCheetah"})
        anim.save(args.output, writer=writer)
        print(f"Video saved to {args.output}")
    except Exception:
        gif_path = args.output.replace(".mp4", ".gif")
        writer = PillowWriter(fps=args.fps)
        anim.save(gif_path, writer=writer)
        print(f"GIF saved to {gif_path} (install ffmpeg for MP4)")

    plt.close()


if __name__ == "__main__":
    main()
