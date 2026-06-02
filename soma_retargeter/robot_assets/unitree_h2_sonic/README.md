# Unitree H2 Sonic Description

This directory contains the H2 robot description copied from the Sonic training
workspace:

`/home/sky/workspace/GR00T-WholeBodyControl-UnitreeH2-FT/gear_sonic/data/assets/robot_description`

- `h2.xml` is copied from `mjcf/h2.xml` and is the model loaded by SOMA
  Retargeter for the `unitree_h2_sonic` target.
- `h2.urdf` and `meshes/` are copied from `urdf/h2/`.
- `h2.xml` adds fixed `left_sole_link` and `right_sole_link` marker bodies
  under the ankle pitch links so existing foot retargeting and stabilization
  can target a sole frame. These markers do not add joints, mass, or geometry.
- `scene.xml` includes `h2.xml` with viewer lighting and a ground plane.
- `h2_sonic_mujoco_joint_order.json` and `JOINT_ORDER.md` document the
  MuJoCo `qpos[7:]`/`dof_pos` order used by exported Sonic H2 motions.

The joint order is compatible with the existing 31-DOF H2 CSV export. Before
hardware deployment, verify the Sonic training/control stack's motor order,
joint signs, command units, and whether the deployed controller expects head,
waist, and wrist joints.
