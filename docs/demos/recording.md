# Recording & Visualization

JoltGym includes scripts for recording environment visualizations as MP4 videos or GIFs using matplotlib-based 2D/3D skeletal rendering.

## Prerequisites

```bash
pip install matplotlib

# For MP4 output (optional, falls back to GIF)
brew install ffmpeg      # macOS
sudo apt install ffmpeg  # Ubuntu/Debian
```

## HalfCheetah Recording

`examples/record_video.py`

Renders a 2D side-view skeletal visualization of the cheetah using forward kinematics.

### Usage

```bash
# Random policy
python examples/record_video.py

# Trained model
python examples/record_video.py --model models/halfcheetah_ppo

# Custom settings
python examples/record_video.py --steps 500 --fps 30 --output videos/my_cheetah.mp4
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--model` | None | Path to trained SB3 model (random policy if omitted) |
| `--steps` | 300 | Number of simulation steps to record |
| `--fps` | 20 | Video frame rate |
| `--output` | `videos/halfcheetah.mp4` | Output file path |
| `--seed` | 42 | Random seed |

### Visualization Details

The renderer uses forward kinematics to compute world positions of each body segment from the observation vector:

- **Torso**: blue
- **Back legs**: red shades
- **Front legs**: green shades
- Camera follows the cheetah's X position
- Ground plane rendered at Z=0

---

## Humanoid Recording

`examples/record_humanoid.py`

Renders the 3D humanoid as a skeletal stick figure with multiple view options.

### Usage

```bash
# Random policy, side view
python examples/record_humanoid.py

# Trained model, 3D rotating view
python examples/record_humanoid.py --model models/humanoid_ppo --view 3d

# Front view
python examples/record_humanoid.py --view front --steps 500
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--model` | None | Path to trained SB3 model |
| `--steps` | 300 | Number of steps |
| `--fps` | 30 | Frame rate |
| `--output` | `videos/humanoid.mp4` | Output path |
| `--seed` | 42 | Random seed |
| `--view` | `side` | View mode: `side`, `front`, or `3d` |

### View Modes

=== "Side View"

    X vs Z projection. Camera follows the humanoid horizontally. Best for observing forward locomotion.

=== "Front View"

    Y vs Z projection. Static camera. Best for observing lateral stability and arm movement.

=== "3D View"

    Full 3D matplotlib projection with slowly rotating camera. Shows the complete spatial structure.

### Skeleton Structure

The humanoid skeleton consists of 13 bodies and 12 edges:

| Body | Color |
|---|---|
| Torso chain (torso, lwaist, pelvis) | Blue |
| Right leg (thigh, shin, foot) | Red |
| Left leg (thigh, shin, foot) | Green |
| Right arm (upper, lower) | Orange |
| Left arm (upper, lower) | Purple |

---

## CheetahRace Recording

`examples/record_race.py`

Renders multiple cheetahs racing side-by-side with per-agent color coding and a live scoreboard.

### Usage

```bash
# 2-agent random race
python examples/record_race.py

# 4-agent trained race
python examples/record_race.py --model models/cheetah_race_ppo --agents 4

# Custom settings
python examples/record_race.py --agents 3 --steps 500 --fps 30
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--model` | None | Path to trained SB3 model |
| `--agents` | 2 | Number of racing cheetahs |
| `--steps` | 500 | Number of steps |
| `--fps` | 20 | Frame rate |
| `--output` | `videos/cheetah_race.mp4` | Output path |
| `--seed` | 42 | Random seed |

### Visualization Details

- Each agent is drawn in a distinct color palette
- Agents are offset vertically for visual clarity (since the physics is 2D side-view)
- Each lane has its own ground line
- A live scoreboard in the title shows each agent's X position and the current leader
- Camera follows the mean X position of all agents

### Agent Color Palettes

| Agent | Torso | Back Legs | Front Legs |
|---|---|---|---|
| 0 | Blue | Red | Green |
| 1 | Orange | Purple | Cyan |
| 2 | Pink | Brown | Grey |
| 3 | Yellow | Indigo | Teal |

---

## Output Formats

All recording scripts try FFmpeg first for MP4 output, then fall back to Pillow for GIF:

```
videos/
  |-- halfcheetah.mp4      # or .gif
  |-- humanoid.mp4         # or .gif
  +-- cheetah_race.mp4     # or .gif
```

!!! tip
    Install FFmpeg for significantly smaller file sizes and better quality compared to GIF output.
