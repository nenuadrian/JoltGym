#!/usr/bin/env python3
"""Record a video of a multi-agent CheetahRace.

Renders a 2D skeletal visualization of multiple cheetahs racing
side-by-side in a shared physics world.

Usage:
    python examples/record_race.py                                      # Random policy
    python examples/record_race.py --model models/cheetah_race_ppo      # Trained
    python examples/record_race.py --agents 4 --steps 500 --fps 30
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

import joltgym

# ── Cheetah kinematics (reused from record_video.py) ────────────────

BODY_LOCAL_POS = {
    "torso":  (None,    0.0,    0.0),
    "bthigh": (0,      -0.5,    0.0),
    "bshin":  (1,       0.16,  -0.25),
    "bfoot":  (2,      -0.28,  -0.14),
    "fthigh": (0,       0.5,    0.0),
    "fshin":  (4,      -0.14,  -0.24),
    "ffoot":  (5,       0.13,  -0.18),
}

CAPSULE_ENDPOINTS = {
    "torso":  [(-0.5, 0.0), (0.5, 0.0)],
    "bthigh": [(-0.145, 0.0), (0.145, 0.0)],
    "bshin":  [(-0.15, 0.0), (0.15, 0.0)],
    "bfoot":  [(-0.094, 0.0), (0.094, 0.0)],
    "fthigh": [(-0.133, 0.0), (0.133, 0.0)],
    "fshin":  [(-0.106, 0.0), (0.106, 0.0)],
    "ffoot":  [(-0.07, 0.0), (0.07, 0.0)],
}

GEOM_ANGLES = {
    "torso": 0.0, "bthigh": -3.8, "bshin": -2.03,
    "bfoot": -0.27, "fthigh": 0.52, "fshin": -0.6, "ffoot": -0.6,
}

BODY_NAMES = ["torso", "bthigh", "bshin", "bfoot", "fthigh", "fshin", "ffoot"]
JOINT_NAMES = ["bthigh", "bshin", "bfoot", "fthigh", "fshin", "ffoot"]

# Color palettes per agent
AGENT_PALETTES = [
    {"torso": "#2196F3", "back": "#F44336", "front": "#4CAF50"},  # blue/red/green
    {"torso": "#FF9800", "back": "#9C27B0", "front": "#00BCD4"},  # orange/purple/cyan
    {"torso": "#E91E63", "back": "#795548", "front": "#607D8B"},  # pink/brown/grey
    {"torso": "#FFEB3B", "back": "#3F51B5", "front": "#009688"},  # yellow/indigo/teal
]

def get_agent_colors(agent_idx):
    p = AGENT_PALETTES[agent_idx % len(AGENT_PALETTES)]
    return {
        "torso": p["torso"],
        "bthigh": p["back"], "bshin": p["back"], "bfoot": p["back"],
        "fthigh": p["front"], "fshin": p["front"], "ffoot": p["front"],
    }


def forward_kinematics(root_x, root_z, root_angle, joint_angles):
    positions, orientations = {}, {}
    positions["torso"] = np.array([root_x, root_z])
    orientations["torso"] = root_angle
    angle_map = dict(zip(JOINT_NAMES, joint_angles))
    for name in BODY_NAMES[1:]:
        pi, lx, lz = BODY_LOCAL_POS[name]
        pp = positions[BODY_NAMES[pi]]
        pa = orientations[BODY_NAMES[pi]]
        c, s = np.cos(pa), np.sin(pa)
        positions[name] = np.array([pp[0] + c*lx - s*lz, pp[1] + s*lx + c*lz])
        orientations[name] = pa + angle_map.get(name, 0.0)
    return positions, orientations


def draw_cheetah(ax, root_x, root_z, root_angle, joint_angles, colors, label=None):
    positions, orientations = forward_kinematics(root_x, root_z, root_angle, joint_angles)
    for name in BODY_NAMES:
        pos, angle = positions[name], orientations[name]
        ta = angle + GEOM_ANGLES[name]
        ep = CAPSULE_ENDPOINTS[name]
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--agents", type=int, default=2)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--output", type=str, default="videos/cheetah_race.mp4")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    n = args.agents
    policy = None
    if args.model:
        from stable_baselines3 import PPO
        policy = PPO.load(args.model)
        print(f"Loaded model from {args.model}")
    else:
        print("No model — random policy")

    env = joltgym.make("JoltGym/CheetahRace-v0", num_agents=n)
    obs, info = env.reset(seed=args.seed)
    obs_dim = 17  # per agent

    print(f"Recording {n}-agent race for {args.steps} steps...")

    frames = []  # list of [(root_x, root_z, root_angle, joint_angles) per agent]
    total_rewards = np.zeros(n)

    for step in range(args.steps):
        frame = []
        for i in range(n):
            agent_obs = obs[i * obs_dim : (i + 1) * obs_dim]
            rx = info.get(f"agent_{i}_x", 0.0)
            rz = float(agent_obs[0])
            ra = float(agent_obs[1])
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

    env.close()
    print(f"Total rewards: {total_rewards}")

    # ── Render ──────────────────────────────────────────
    print(f"Rendering {len(frames)} frames...")
    fig_height = max(4, 2 + n * 1.2)
    fig, ax = plt.subplots(1, 1, figsize=(14, fig_height), dpi=100)
    fig.patch.set_facecolor("#E8E0D0")

    agent_colors = [get_agent_colors(i) for i in range(n)]

    def animate(fi):
        ax.clear()
        frame = frames[fi]
        xs = [f[0] for f in frame]
        cam_x = np.mean(xs)

        # Ground
        ax.fill_between([cam_x - 8, cam_x + 8], -1.5, 0,
                        color="#C4A882", alpha=0.5, zorder=0)
        ax.axhline(y=0, color="#8B7355", linewidth=2, zorder=1)

        # Lane lines
        spacing = 3.0
        for i in range(n + 1):
            y = (i - 0.5) * spacing - (n - 1) * spacing / 2
            # We only have X (forward) and Z (up) in 2D side view,
            # but agents are offset in Y. We draw them at different Z offsets.

        # Draw each agent (offset vertically for visual clarity)
        z_offset_per_agent = 1.5
        for i, (rx, rz, ra, ja) in enumerate(frame):
            # Visual vertical offset so agents don't overlap in 2D
            visual_z = rz + i * z_offset_per_agent
            draw_cheetah(ax, rx, visual_z, ra, ja, agent_colors[i],
                         label=f"Agent {i}")

        view_w = max(6, (max(xs) - min(xs)) + 6)
        ax.set_xlim(cam_x - view_w/2, cam_x + view_w/2)
        ax.set_ylim(-0.5, 1.0 + n * z_offset_per_agent)
        ax.set_aspect("equal")
        ax.set_facecolor("#D4E6F1")

        # Draw multiple ground lines for each lane
        for i in range(n):
            gz = i * z_offset_per_agent
            ax.axhline(y=gz, color="#8B7355", linewidth=1.5, alpha=0.4, zorder=0)
            ax.fill_between([cam_x - view_w, cam_x + view_w],
                            gz - 0.3, gz, color="#C4A882", alpha=0.2, zorder=0)

        # Scoreboard
        score_lines = []
        for i, (rx, rz, ra, ja) in enumerate(frame):
            score_lines.append(f"Agent {i}: x={rx:+.2f}")
        leader = max(range(n), key=lambda i: frame[i][0])
        score_lines.append(f"Leader: Agent {leader}")

        ax.set_title(
            f"JoltGym CheetahRace ({n} agents)  |  Step {fi}/{len(frames)}  |  "
            + "  ".join(score_lines),
            fontsize=10, fontfamily="monospace"
        )
        ax.tick_params(labelsize=8)

    anim = FuncAnimation(fig, animate, frames=len(frames), interval=1000 // args.fps)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    try:
        writer = FFMpegWriter(fps=args.fps, metadata={"title": "JoltGym CheetahRace"})
        anim.save(args.output, writer=writer)
        print(f"Video saved to {args.output}")
    except Exception:
        gif_path = args.output.replace(".mp4", ".gif")
        writer = PillowWriter(fps=args.fps)
        anim.save(gif_path, writer=writer)
        print(f"GIF saved to {gif_path} (install ffmpeg for MP4)")

    plt.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
