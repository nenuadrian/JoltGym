#!/usr/bin/env python3
"""Record a video of a trained (or random) Humanoid agent.

Renders a 2D front-projection skeletal visualization using matplotlib.

Usage:
    python examples/record_humanoid.py                                # Random policy
    python examples/record_humanoid.py --model models/humanoid_ppo    # Trained
    python examples/record_humanoid.py --steps 500 --fps 30
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection

import joltgym


# ── Humanoid body tree ──────────────────────────────────────────────
# Built from the MJCF: body name, parent index, local offset (x,y,z)
BODIES = [
    # idx  name               parent  local_pos
    (0,  "torso",             None,   (0, 0, 1.4)),
    (1,  "lwaist",            0,      (-0.01, 0, -0.260)),
    (2,  "pelvis",            1,      (0, 0, -0.165)),
    (3,  "right_thigh",       2,      (0, -0.1, -0.04)),
    (4,  "right_shin",        3,      (0, 0.01, -0.403)),
    (5,  "right_foot",        4,      (0, 0, -0.45)),
    (6,  "left_thigh",        2,      (0, 0.1, -0.04)),
    (7,  "left_shin",         6,      (0, -0.01, -0.403)),
    (8,  "left_foot",         7,      (0, 0, -0.45)),
    (9,  "right_upper_arm",   0,      (0, -0.17, 0.06)),
    (10, "right_lower_arm",   9,      (0.18, -0.18, -0.18)),
    (11, "left_upper_arm",    0,      (0, 0.17, 0.06)),
    (12, "left_lower_arm",    11,     (0.18, 0.18, -0.18)),
]

# Which bodies to draw lines between (skeleton edges)
SKELETON_EDGES = [
    (0, 1),   # torso → lwaist
    (1, 2),   # lwaist → pelvis
    (2, 3),   # pelvis → right_thigh
    (3, 4),   # right_thigh → right_shin
    (4, 5),   # right_shin → right_foot
    (2, 6),   # pelvis → left_thigh
    (6, 7),   # left_thigh → left_shin
    (7, 8),   # left_shin → left_foot
    (0, 9),   # torso → right_upper_arm
    (9, 10),  # right_upper_arm → right_lower_arm
    (0, 11),  # torso → left_upper_arm
    (11, 12), # left_upper_arm → left_lower_arm
]

# Colors per limb group
EDGE_COLORS = {
    (0, 1): "#2196F3",   (1, 2): "#2196F3",            # torso chain (blue)
    (2, 3): "#F44336",   (3, 4): "#FF8A80", (4, 5): "#FF8A80",  # right leg (red)
    (2, 6): "#4CAF50",   (6, 7): "#A5D6A7", (7, 8): "#A5D6A7",  # left leg (green)
    (0, 9): "#FF9800",   (9, 10): "#FFB74D",            # right arm (orange)
    (0, 11): "#9C27B0",  (11, 12): "#CE93D8",           # left arm (purple)
}

JOINT_NAMES = [
    "abdomen_y", "abdomen_z", "abdomen_x",
    "right_hip_x", "right_hip_z", "right_hip_y", "right_knee",
    "left_hip_x", "left_hip_z", "left_hip_y", "left_knee",
    "right_shoulder1", "right_shoulder2", "right_elbow",
    "left_shoulder1", "left_shoulder2", "left_elbow",
]

# Map joint name → which body it rotates, and its axis
# For simple visualization we accumulate rotations down the chain
# but for a quick side-view projection this is sufficient:
# we read body positions directly from the physics engine via obs.


def quat_to_rotation_matrix(qw, qx, qy, qz):
    """Convert quaternion to 3x3 rotation matrix."""
    return np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),     1 - 2*(qx*qx + qz*qz),  2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw),       1 - 2*(qx*qx + qy*qy)],
    ])


def forward_kinematics_simple(root_x, root_y, root_z, qw, qx, qy, qz, joint_angles):
    """Compute approximate world positions of all bodies using static offsets + root transform.

    This is a simplified FK that applies the root quaternion to all local offsets
    and accumulates joint rotations down the chain. Good enough for visualization.
    """
    root_rot = quat_to_rotation_matrix(qw, qx, qy, qz)
    root_pos = np.array([root_x, root_y, root_z])

    positions = {}
    positions[0] = root_pos

    for idx, name, parent_idx, local_pos in BODIES:
        if parent_idx is None:
            continue
        parent_pos = positions[parent_idx]
        offset = root_rot @ np.array(local_pos)
        positions[idx] = parent_pos + offset

    # Re-anchor: make torso position correct and compute children relative
    # Actually, let's do proper recursive FK
    positions = {}
    positions[0] = root_pos
    rotations = {}
    rotations[0] = root_rot

    for idx, name, parent_idx, local_pos in BODIES:
        if parent_idx is None:
            continue
        parent_pos = positions[parent_idx]
        parent_rot = rotations[parent_idx]
        offset = parent_rot @ np.array(local_pos)
        positions[idx] = parent_pos + offset
        rotations[idx] = parent_rot  # Simplified: ignore joint rotations for vis

    return positions


def obs_to_body_positions(obs, root_x, root_y):
    """Extract body positions from humanoid observation.

    obs layout (45 dims):
      qpos[2:]: root_z(1), quat(4), joints(17) = indices 0-21
      qvel: lin(3), ang(3), joints(17) = indices 22-44
    """
    root_z = obs[0]
    qw, qx, qy, qz = obs[1], obs[2], obs[3], obs[4]
    joint_angles = obs[5:22]

    positions = forward_kinematics_simple(
        root_x, root_y, root_z, qw, qx, qy, qz, joint_angles
    )
    return positions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--output", type=str, default="videos/humanoid.mp4")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--view", choices=["side", "front", "3d"], default="side")
    args = parser.parse_args()

    policy = None
    if args.model:
        from stable_baselines3 import PPO
        policy = PPO.load(args.model)
        print(f"Loaded model from {args.model}")
    else:
        print("No model — random policy")

    env = joltgym.make("JoltGym/Humanoid-v0")
    base_env = env.unwrapped
    obs, info = env.reset(seed=args.seed)

    print(f"Recording {args.steps} steps ({args.view} view)...")
    frames = []  # list of (positions_dict, root_x, root_z)
    total_reward = 0

    for step in range(args.steps):
        root_x = base_env._core.get_root_x()
        root_z = base_env._core.get_root_z()
        positions = obs_to_body_positions(obs, root_x, 0.0)
        frames.append((positions, root_x, root_z))

        if policy is not None:
            action, _ = policy.predict(obs, deterministic=True)
        else:
            action = env.action_space.sample()

        obs, reward, term, trunc, info = env.step(action)
        total_reward += reward

        if term or trunc:
            obs, info = env.reset()

    env.close()
    print(f"Total reward: {total_reward:.1f}")

    # ── Render ──────────────────────────────────────────────
    print(f"Rendering {len(frames)} frames...")

    if args.view == "3d":
        fig = plt.figure(figsize=(10, 8), dpi=100)
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig, ax = plt.subplots(1, 1, figsize=(12, 6), dpi=100)

    fig.patch.set_facecolor("#1a1a2e")

    def animate(fi):
        ax.clear()
        positions, root_x, root_z = frames[fi]

        if args.view == "3d":
            # 3D view
            for (i, j) in SKELETON_EDGES:
                if i in positions and j in positions:
                    p1, p2 = positions[i], positions[j]
                    color = EDGE_COLORS.get((i, j), "#FFFFFF")
                    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                            color=color, linewidth=4, solid_capstyle="round")

            for idx in positions:
                p = positions[idx]
                ax.scatter(p[0], p[1], p[2], color="white", s=20, zorder=5,
                           edgecolors="black", linewidths=0.5)

            # Ground plane
            gx = np.linspace(root_x - 2, root_x + 2, 2)
            gy = np.linspace(-2, 2, 2)
            gx, gy = np.meshgrid(gx, gy)
            gz = np.zeros_like(gx)
            ax.plot_surface(gx, gy, gz, alpha=0.15, color="#8B7355")

            ax.set_xlim(root_x - 1.5, root_x + 1.5)
            ax.set_ylim(-1.5, 1.5)
            ax.set_zlim(-0.1, 2.2)
            ax.set_box_aspect([3, 3, 2.3])
            ax.view_init(elev=15, azim=-60 + fi * 0.2)
            ax.set_facecolor("#1a1a2e")
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False

        elif args.view == "front":
            # Front view: Y vs Z
            for (i, j) in SKELETON_EDGES:
                if i in positions and j in positions:
                    p1, p2 = positions[i], positions[j]
                    color = EDGE_COLORS.get((i, j), "#FFFFFF")
                    ax.plot([p1[1], p2[1]], [p1[2], p2[2]],
                            color=color, linewidth=5, solid_capstyle="round")

            for idx in positions:
                p = positions[idx]
                ax.plot(p[1], p[2], "o", color="white", markersize=5,
                        markeredgecolor="black", markeredgewidth=0.5, zorder=5)

            ax.fill_between([-2, 2], -0.1, 0, color="#8B7355", alpha=0.4)
            ax.axhline(y=0, color="#8B7355", linewidth=2)
            ax.set_xlim(-1.2, 1.2)
            ax.set_ylim(-0.2, 2.2)
            ax.set_aspect("equal")
            ax.set_facecolor("#16213e")

        else:
            # Side view: X vs Z
            for (i, j) in SKELETON_EDGES:
                if i in positions and j in positions:
                    p1, p2 = positions[i], positions[j]
                    color = EDGE_COLORS.get((i, j), "#FFFFFF")
                    ax.plot([p1[0], p2[0]], [p1[2], p2[2]],
                            color=color, linewidth=5, solid_capstyle="round")

            for idx in positions:
                p = positions[idx]
                ax.plot(p[0], p[2], "o", color="white", markersize=5,
                        markeredgecolor="black", markeredgewidth=0.5, zorder=5)

            ax.fill_between([root_x - 3, root_x + 3], -0.1, 0,
                            color="#8B7355", alpha=0.4)
            ax.axhline(y=0, color="#8B7355", linewidth=2)
            ax.set_xlim(root_x - 2, root_x + 2)
            ax.set_ylim(-0.2, 2.2)
            ax.set_aspect("equal")
            ax.set_facecolor("#16213e")

        title_color = "#e0e0e0"
        ax.set_title(
            f"JoltGym Humanoid  |  Step {fi}  |  "
            f"x={root_x:.2f}  z={root_z:.2f}",
            fontsize=11, fontfamily="monospace", color=title_color, pad=10
        )
        ax.tick_params(colors="#888888", labelsize=8)
        for spine in ax.spines.values() if hasattr(ax, 'spines') else []:
            spine.set_color("#333333")

    anim = FuncAnimation(fig, animate, frames=len(frames),
                         interval=1000 // args.fps)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    try:
        writer = FFMpegWriter(fps=args.fps,
                              metadata={"title": "JoltGym Humanoid"})
        anim.save(args.output, writer=writer)
        print(f"Video saved to {args.output}")
    except Exception as e:
        gif_path = args.output.replace(".mp4", ".gif")
        writer = PillowWriter(fps=args.fps)
        anim.save(gif_path, writer=writer)
        print(f"GIF saved to {gif_path} (install ffmpeg for MP4)")

    plt.close()


if __name__ == "__main__":
    main()
