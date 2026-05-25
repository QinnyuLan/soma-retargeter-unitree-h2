# SOMA Retargeter - Unitree H2 Adaptation
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

![SOMA Retargeter Unitree H2 Banner](assets/docs/banner.gif)

Convert [SOMA](https://github.com/NVlabs/SOMA-X) human motion captures into humanoid robot joint animation. Takes BVH motion files as input and produces robot-playable CSV joint data as output using GPU-optimized inverse kinematics via [Newton](https://github.com/newton-physics/newton) and high-performance computation with [NVIDIA Warp](https://github.com/NVIDIA/warp).

The retargeting pipeline handles proportional human-to-robot scaling, multi-objective IK solving with joint limits, feet stabilization to maintain ground contact, and per-DOF joint limit clamping. Currently supports SOMA as the input skeleton and Unitree G1 (29 DOF) and Unitree H2 (31 DOF) as output robots. Additional robot targets are planned.

SOMA Retargeter is part of the [SOMA body model](https://github.com/NVlabs/SOMA-X) ecosystem for humanoid motion data.

> **Note:** This project is in active development. The API may change between releases as the design is refined.

## Unitree H2 Fork Changes

This fork adds a functional Unitree H2 target on top of SOMA Retargeter:

- Added local Unitree H2 robot assets, including MJCF, URDF, meshes, license notes, and packaging/LFS rules.
- Added SOMA-to-H2 retargeting, scaling, and feet-stabilization configs under `soma_retargeter/configs/unitree_h2/`.
- Added `assets/h2_bvh_to_csv_config.json` for H2 viewer and batch conversion workflows.
- Added 31-DOF H2 CSV export matching the MuJoCo `qpos[7:]` joint order documented in `soma_retargeter/robot_assets/unitree_h2/JOINT_ORDER.md`.
- Generalized the pipeline and viewer from a G1-only target path to explicit robot target selection for Unitree G1 and Unitree H2.
- Added viewer startup options for loading BVH/CSV files directly and retargeting on launch, plus a separated default layout for comparing SOMA and H2 motion.

The current H2 parameters are intended for visualization and offline CSV generation. Before hardware deployment, verify joint order, joint signs, command units, limits, and controller expectations against the target robot stack.

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

### H2 Tuning

Most H2 retargeting behavior is controlled by JSON configs:

- `soma_retargeter/configs/unitree_h2/soma_to_h2_retargeter_config.json`: IK iterations, joint limit/smoothing weights, post-processing, and per-body IK target weights.
- `soma_retargeter/configs/unitree_h2/soma_to_h2_scaler_config.json`: body-part scale factors and per-joint target offsets from SOMA into the H2 tracking frame.
- `soma_retargeter/configs/unitree_h2/h2_feet_stabilizer_config.json`: post-process foot stabilization weights and two-bone leg IK hints.

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
