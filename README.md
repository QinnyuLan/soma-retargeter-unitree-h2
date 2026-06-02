# SOMA Retargeter - Unitree H2 Adaptation
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

![SOMA Retargeter Unitree H2 Banner](assets/docs/banner.gif)

Convert [SOMA](https://github.com/NVlabs/SOMA-X) human motion captures into humanoid robot joint animation. Takes BVH motion files as input and produces robot-playable CSV joint data as output using GPU-optimized inverse kinematics via [Newton](https://github.com/newton-physics/newton) and high-performance computation with [NVIDIA Warp](https://github.com/NVIDIA/warp).

The retargeting pipeline handles proportional human-to-robot scaling, multi-objective IK solving with joint limits, feet stabilization to maintain ground contact, and per-DOF joint limit clamping. Currently supports SOMA as the input skeleton and Unitree G1 (29 DOF), Unitree H2 (31 DOF), and the Sonic H2 training-asset variant (31 DOF) as output robots. Additional robot targets are planned.

SOMA Retargeter is part of the [SOMA body model](https://github.com/NVlabs/SOMA-X) ecosystem for humanoid motion data.

> **Note:** This project is in active development. The API may change between releases as the design is refined.

## Unitree H2 Fork Changes

This fork adds a functional Unitree H2 target on top of SOMA Retargeter:

- Added local Unitree H2 robot assets, including MJCF, URDF, meshes, license notes, and packaging/LFS rules.
- Added SOMA-to-H2 retargeting, scaling, and feet-stabilization configs under `soma_retargeter/configs/unitree_h2/`.
- Added `assets/h2_bvh_to_csv_config.json` for H2 viewer and batch conversion workflows.
- Added `unitree_h2_sonic`, a second H2 target using the H2 assets from the Sonic
  training workspace, with separate assets, configs, and conversion entry point.
- Added 31-DOF H2 CSV export matching the MuJoCo `qpos[7:]` joint order documented in `soma_retargeter/robot_assets/unitree_h2/JOINT_ORDER.md`.
- Generalized the pipeline and viewer from a G1-only target path to explicit robot target selection for Unitree G1, Unitree H2, and Sonic H2.
- Added viewer startup options for loading BVH/CSV files directly and retargeting on launch, plus a separated default layout for comparing SOMA and H2 motion.
- Added light head/neck retargeting for the Sonic H2 target so `head_pitch` and
  `head_yaw` are driven by SOMA `Neck1` rotation instead of staying fixed.

The current H2 and Sonic H2 parameters are intended for visualization and offline CSV generation. Before hardware deployment, verify joint order, joint signs, command units, limits, and controller expectations against the target robot stack.

## Requirements

- **Python:** 3.12
- **Git LFS:** Installed and initialized for asset downloads
- **OS:** Windows (x86-64) and Linux (x86-64, aarch64)
- **GPU:** NVIDIA GPU (Maxwell or newer), driver 545+ (CUDA 12). No local CUDA Toolkit installation required.

## Installation

<details>

<summary>Setup instructions</summary>

### Method 1 (conda + pip)

#### 1. Create and Activate Conda Environment

```bash
conda create -n soma-retargeter python=3.12 -y
conda activate soma-retargeter
```

#### 2. Download LFS Assets

```bash
git lfs pull
```

#### 3. Install the Library

```bash
pip install .
```

### Method 2 (uv)

#### 1. Install uv

Follow the [official installation guide](https://docs.astral.sh/uv/getting-started/installation/) if `uv` is not yet installed.

#### 2. Download LFS Assets

```bash
git lfs pull
```

#### 3. Sync the Project

`uv sync` creates an isolated `.venv` virtual environment inside the project directory, installs the correct Python version and resolves all dependencies.

```bash
uv sync
```

### Platform-specific notes

**Note (Linux):** For the GUI viewer to work, install `tkinter`

```bash
sudo apt-get install python3.12-tk
```

**Note (Windows):** If `imgui-bundle` fails to install, the Microsoft Visual C++ Redistributables may be missing. Download from the [official Microsoft documentation](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist).

</details>

## Motion Data

This repo includes 10 sample BVH/CSV pairs in `assets/motions/` for immediate testing.

For large-scale motion data, see the [SEED dataset](https://huggingface.co/datasets/bones-studio/seed) (Skeletal Everyday Embodiment Dataset) published by [Bones Studio](https://huggingface.co/bones-studio). SEED provides a large-scale collection of human motions on the SOMA uniform-proportion skeleton, which is the expected input format for this tool. The G1 robot motion data included in SEED was retargeted using SOMA Retargeter.

## Quick Start

> When using **uv** (Method 2), replace `python` with `uv run` in the commands below.

### Interactive viewer (OpenGL)

```bash
python ./app/bvh_to_csv_converter.py --config ./assets/default_bvh_to_csv_converter_config.json --viewer gl
```

The viewer displays the source SOMA motion alongside the retargeted robot in a 3D viewport. Use the right panel to load BVH files, run retargeting, and save CSV output. Playback controls at the bottom allow scrubbing, speed adjustment, and looping. Toggle visibility of the skinned mesh, skeleton, joint axes, and positioning gizmos.

The `--bvh` and `--csv` options load files at startup. `--retarget-on-load` immediately runs retargeting for the loaded BVH. These options are useful on Linux systems without `tkinter`, where the viewer still works but native file dialogs are unavailable.

### Batch conversion (headless)

Process a folder of BVH files without a display. Set `import_folder` and `export_folder` in the config file, then run:

```bash
python ./app/bvh_to_csv_converter.py --config ./assets/default_bvh_to_csv_converter_config.json --viewer null
```

Batch mode recursively finds all `.bvh` files in the import folder, processes them in configurable batch sizes, and writes CSV files to the export folder mirroring the input directory structure.

For Unitree H2 export, use the H2 config:

```bash
python ./app/bvh_to_csv_converter.py --config ./assets/h2_bvh_to_csv_config.json --viewer null
```

For the Sonic H2 training-asset variant, use:

```bash
python ./app/bvh_to_csv_converter.py --config ./assets/h2_sonic_bvh_to_csv_config.json --viewer null
```

## Unitree H2 Support

Unitree H2 support includes a local MJCF model, mesh assets, 31-DOF CSV output order, SOMA-to-H2 retargeting settings, human-to-robot scaling, and feet stabilization. The H2 assets live under `soma_retargeter/robot_assets/unitree_h2/` and are distributed under Unitree Robotics' BSD 3-Clause license included in that directory.

The H2 CSV joint order follows the MuJoCo `qpos[7:]` order documented in `soma_retargeter/robot_assets/unitree_h2/JOINT_ORDER.md`. Verify SDK motor order, joint signs, command units, and whether the deployed controller expects head, waist, and wrist joints before sending exported motion to hardware.

To open the viewer directly with the Unitree H2 target and an example motion:

```bash
python ./app/bvh_to_csv_converter.py \
  --config ./assets/h2_bvh_to_csv_config.json \
  --viewer gl \
  --bvh assets/motions/bvh/dance_hiphop_shuffle_square_R_fast_002__A318.bvh \
  --retarget-on-load
```

![Interactive H2 viewer interface](assets/docs/interactive_viewer_h2.gif)

### H2 Sonic Training Assets

`unitree_h2_sonic` uses the H2 robot description from the local Sonic training
workspace:

`/home/sky/workspace/GR00T-WholeBodyControl-UnitreeH2-FT/gear_sonic/data/assets/robot_description`

The copied assets live under `soma_retargeter/robot_assets/unitree_h2_sonic/`.
The retargeter configs live under `soma_retargeter/configs/unitree_h2_sonic/`,
and `assets/h2_sonic_bvh_to_csv_config.json` selects this target for viewer and
batch conversion runs.

To open the viewer directly with the Sonic H2 target:

```bash
python ./app/bvh_to_csv_converter.py \
  --config ./assets/h2_sonic_bvh_to_csv_config.json \
  --viewer gl \
  --bvh assets/motions/bvh/dance_hiphop_shuffle_square_R_fast_002__A318.bvh \
  --retarget-on-load
```

To retarget the full SEED BVH tree with the tuned Sonic H2 parameters, use the
resume-friendly sharded exporter. This command writes only the new Sonic H2
dataset and leaves existing datasets untouched:

```bash
ROOT=/mnt/data/seed/h2_sonic_full
IMPORT_ROOT=/mnt/data/seed/soma_uniform/bvh
NUM_SHARDS=4
BATCH_SIZE=32

mkdir -p "$ROOT/csv" "$ROOT/logs"

for shard in $(seq 0 $((NUM_SHARDS - 1))); do
  CUDA_VISIBLE_DEVICES=0 uv run python tools/dataset_export/batch_bvh_to_csv.py \
    --import-root "$IMPORT_ROOT" \
    --export-root "$ROOT/csv" \
    --robot-type unitree_h2_sonic \
    --num-shards "$NUM_SHARDS" \
    --shard-index "$shard" \
    --batch-size "$BATCH_SIZE" \
    --device cuda:0 \
    --progress "$ROOT/logs/progress_${shard}.csv" \
    --failures "$ROOT/logs/failures_${shard}.csv" \
    > "$ROOT/logs/shard_${shard}.log" 2>&1 &
done

wait
```

The exporter sorts clips by size before round-robin sharding so long motions are
balanced across workers. It mirrors the input folder structure under
`$ROOT/csv`, skips non-empty existing CSVs on rerun, and records per-shard
progress and failures under `$ROOT/logs`.

The Sonic H2 output uses the same 31-DOF CSV column layout as the base H2
target. Its MuJoCo `qpos[7:]` and actuator order are documented in
`soma_retargeter/robot_assets/unitree_h2_sonic/JOINT_ORDER.md`. Compared with
the base H2 target, the Sonic config lightly tracks SOMA `Neck1` rotation through
`head_yaw_link` (`r_weight = 0.35`, `t_weight = 0.0`) and smooths both head
links with a low mask (`0.2`). This keeps the head DOFs active without letting
head position pull the torso or arms.

### H2 Mesh Video Export

To save Newton-style mesh review videos without using an on-screen OpenGL viewer,
use the helper scripts in `tools/h2_newton_mesh_videos/`. They render the white
Unitree H2 mesh next to the yellow SOMA skinned mesh with Blender.

```bash
uv run python tools/h2_newton_mesh_videos/generate_mesh_video_data.py \
  --fps 24 \
  --out-dir outputs/h2_newton_videos/data

blender --background \
  --python tools/h2_newton_mesh_videos/render_mesh_videos_blender.py -- \
  --out-dir outputs/h2_newton_videos \
  --width 1280 \
  --height 720 \
  --fps 24 \
  outputs/h2_newton_videos/data/*.npz
```

The default export uses five representative sample motions. Generated videos and
intermediate mesh-pose caches are written under `outputs/`, which is ignored by
Git.

### H2 Tuning

Most H2 retargeting behavior is controlled by JSON configs:

- `soma_retargeter/configs/unitree_h2/soma_to_h2_retargeter_config.json`: IK iterations, joint limit/smoothing weights, post-processing, and per-body IK target weights.
- `soma_retargeter/configs/unitree_h2/soma_to_h2_scaler_config.json`: body-part scale factors and per-joint target offsets from SOMA into the H2 tracking frame.
- `soma_retargeter/configs/unitree_h2/h2_feet_stabilizer_config.json`: post-process foot stabilization weights and two-bone leg IK hints.

The Sonic H2 equivalents are under `soma_retargeter/configs/unitree_h2_sonic/`.
The selected Sonic parameters keep the tuned H2 pelvis, limb, foot, and smoothing
settings, then add the low-weight neck/head target described above. On five
representative sample motions, this preserved the H2 tracking baseline while
adding about 16 degrees of average head-pitch range and 15 degrees of average
head-yaw range.

Useful first adjustments:

- Raise/lower the retargeted root by changing the `Hips` z offset in `soma_to_h2_scaler_config.json`.
- Adjust foot clearance by changing the `LeftFoot` and `RightFoot` z offsets in `soma_to_h2_scaler_config.json`.
- If the robot over-crouches to satisfy both pelvis and feet, reduce `LeftFoot` and `RightFoot` `t_weight` values in `soma_to_h2_retargeter_config.json` before changing joint limits.
- If motion is jittery, increase `smooth_joint_filter_weight`; if it feels too sluggish, decrease it.

The current H2 parameters are intended as a functional baseline. They are suitable for visualization and offline CSV generation, but should be calibrated further before hardware deployment.

## Code Overview

### `app/`

| File | Description |
|------|-------------|
| `bvh_to_csv_converter.py` | Main entry point. Drives both interactive and headless batch retargeting modes. |

### `soma_retargeter/`

| Module | Description |
|--------|-------------|
| `animation/` | Core data structures for skeletons, animation buffers, IK, and skinned meshes. |
| `assets/` | File I/O for BVH, CSV, and USD formats. |
| `pipelines/` | Retargeting pipeline: IK solving, feet stabilization, and joint limit clamping. |
| `robotics/` | Human-to-robot scaling and robot output formatting. |
| `renderers/` | Visualization for the interactive viewer. |
| `utils/` | Math, pose, coordinate conversion, Newton and Warp helpers. |
| `configs/` | JSON configuration for retargeting, scaling, and feet stabilization parameters. |

## Related Work

SOMA Retargeter is a support tool within the SOMA ecosystem for humanoid motion data:

* [SOMA Body Model](https://github.com/NVlabs/SOMA-X) - Parametric human body model with standardized skeleton, mesh, and shape parameters
* [GEM-X](https://github.com/NVlabs/GEM-X) - Human motion estimation from video
* [Kimodo](https://github.com/nv-tlabs/kimodo) - Kinematic motion diffusion model for text and constraint-driven 3D human and robot motion generation
* [ProtoMotions](https://github.com/NVlabs/ProtoMotions) - GPU-accelerated simulation and learning framework for training physically simulated digital humans and humanoid robots
* [SONIC](https://nvlabs.github.io/GEAR-SONIC/) - Whole-body control for humanoid robots, training locomotion and interaction policies

## Acknowledgments

This project draws inspiration and builds upon excellent open-source work, including:
* [GMR](https://github.com/YanjieZe/GMR) - General Motion Retargeting
* [PyRoki](https://pyroki-toolkit.github.io/) - A Modular Toolkit for Robot Kinematic Optimization

## License

This codebase is licensed under [Apache-2.0](LICENSE).

This project will download and install additional third-party open source software projects. Review the license terms of these open source projects before use.
