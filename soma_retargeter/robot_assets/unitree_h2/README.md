# Unitree H2 Description (URDF and MJCF)

This directory contains the Unitree H2 robot description used by SOMA Retargeter.

- `H2_original.urdf` is a line-ending-normalized copy of Unitree Robotics'
  `unitree_ros/robots/h2_description/H2.urdf` at repository commit
  `d96d8f63ae17a7108d4f7229c00ef875ba7129c9` (H2 asset commit
  `1cb2d0b66946460435f88458d29584e02f3ff92f`, 2026-07-08).
- `h2.urdf` is the MuJoCo-compatible URDF used to generate the MJCF: mesh paths are normalized and the floating base is enabled.
- `h2.xml` is the generated MJCF model loaded by SOMA Retargeter. It preserves
  the upstream camera and torso/pelvis IMU frames, and adds fixed
  `left_sole_link` and `right_sole_link` frames for foot retargeting.
- `scene.xml` includes the H2 MJCF with viewer lighting and a ground plane.
- `h2_physics.xml` is a separate physics-oriented MJCF with normal gravity, joint damping, position actuators, primitive sole contacts, and a `stand` keyframe.
- `physics_scene.xml` includes `h2_physics.xml` with the ground plane and viewer lighting.
- `h2_mujoco_joint_order.json` and `JOINT_ORDER.md` document the MuJoCo `qpos[7:]`/`dof_pos` and actuator order used by exported H2 motions.

Inspect the physics scene with MuJoCo's standard viewer:

```bash
python -m mujoco.viewer --mjcf=soma_retargeter/robot_assets/unitree_h2/physics_scene.xml
```

The robot asset files are distributed under Unitree Robotics' BSD 3-Clause license in `LICENSE`.
