#!/usr/bin/env python3
"""Record a video of a trained (or random) Humanoid agent.

Uses actual body positions from the Jolt physics engine for accurate rendering.

Usage:
    python examples/record_humanoid.py                                # Random policy
    python examples/record_humanoid.py --model models/humanoid_ppo    # Trained
    python examples/record_humanoid.py --view 3d --steps 500
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

import joltgym


# ── Skeleton edges (by body name) ──────────────────────────────────
SKELETON_EDGES = [
    ("torso", "lwaist"),
    ("lwaist", "pelvis"),
    ("pelvis", "right_thigh"),
    ("right_thigh", "right_shin"),
    ("right_shin", "right_foot"),
    ("pelvis", "left_thigh"),
    ("left_thigh", "left_shin"),
    ("left_shin", "left_foot"),
    ("torso", "right_upper_arm"),
    ("right_upper_arm", "right_lower_arm"),
    ("torso", "left_upper_arm"),
    ("left_upper_arm", "left_lower_arm"),
]

# Head is not a separate body — it's a geom on torso.
# We'll draw it as an offset from torso.
HEAD_OFFSET = np.array([0, 0, 0.19])

EDGE_COLORS = {
    ("torso", "lwaist"):              "#42A5F5",
    ("lwaist", "pelvis"):             "#42A5F5",
    ("pelvis", "right_thigh"):        "#EF5350",
    ("right_thigh", "right_shin"):    "#EF5350",
    ("right_shin", "right_foot"):     "#EF9A9A",
    ("pelvis", "left_thigh"):         "#66BB6A",
    ("left_thigh", "left_shin"):      "#66BB6A",
    ("left_shin", "left_foot"):       "#A5D6A7",
    ("torso", "right_upper_arm"):     "#FFA726",
    ("right_upper_arm", "right_lower_arm"): "#FFB74D",
    ("torso", "left_upper_arm"):      "#AB47BC",
    ("left_upper_arm", "left_lower_arm"):   "#CE93D8",
}


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

    # Get body name→index mapping
    body_names = base_env._core.get_body_names()
    name_to_idx = {name: i for i, name in enumerate(body_names)}
    print(f"Bodies: {body_names}")

    print(f"Recording {args.steps} steps ({args.view} view)...")
    frames = []
    total_reward = 0

    for step in range(args.steps):
        # Get actual body world positions from physics engine
        positions = base_env._core.get_body_positions()  # (N, 3)
        root_x = float(positions[0, 0])
        root_z = float(positions[0, 2])
        frames.append((positions.copy(), root_x, root_z))

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

    def draw_skeleton(ax, positions, view, root_x):
        # Draw edges
        for (name_a, name_b) in SKELETON_EDGES:
            if name_a not in name_to_idx or name_b not in name_to_idx:
                continue
            pa = positions[name_to_idx[name_a]]
            pb = positions[name_to_idx[name_b]]
            color = EDGE_COLORS.get((name_a, name_b), "#FFFFFF")

            if view == "3d":
                ax.plot([pa[0], pb[0]], [pa[1], pb[1]], [pa[2], pb[2]],
                        color=color, linewidth=5, solid_capstyle="round", zorder=2)
            elif view == "front":
                ax.plot([pa[1], pb[1]], [pa[2], pb[2]],
                        color=color, linewidth=6, solid_capstyle="round", zorder=2)
            else:  # side
                ax.plot([pa[0], pb[0]], [pa[2], pb[2]],
                        color=color, linewidth=6, solid_capstyle="round", zorder=2)

        # Draw joints
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

        # Draw head (offset from torso)
        torso = positions[name_to_idx["torso"]]
        head = torso + HEAD_OFFSET
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

        if args.view == "3d":
            draw_skeleton(ax, positions, "3d", root_x)

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
            ax.view_init(elev=15, azim=-60 + fi * 0.3)
            ax.set_facecolor("#1a1a2e")
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False

        elif args.view == "front":
            draw_skeleton(ax, positions, "front", root_x)
            ax.fill_between([-2, 2], -0.1, 0, color="#8B7355", alpha=0.4)
            ax.axhline(y=0, color="#8B7355", linewidth=2)
            ax.set_xlim(-1.0, 1.0)
            ax.set_ylim(-0.2, 2.2)
            ax.set_aspect("equal")
            ax.set_facecolor("#16213e")

        else:  # side
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
            fontsize=11, fontfamily="monospace", color="#e0e0e0", pad=10
        )
        ax.tick_params(colors="#888888", labelsize=8)

    anim = FuncAnimation(fig, animate, frames=len(frames),
                         interval=1000 // args.fps)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    try:
        writer = FFMpegWriter(fps=args.fps, metadata={"title": "JoltGym Humanoid"})
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
