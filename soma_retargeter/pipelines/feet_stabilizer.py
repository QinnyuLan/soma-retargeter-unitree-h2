# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import warp as wp
from scipy.ndimage import gaussian_filter1d

import newton
import soma_retargeter.utils.newton_utils as newton_utils
import soma_retargeter.animation.ik as ik_utils
import soma_retargeter.utils.io_utils as io_utils
import soma_retargeter.robotics.robot_model as robot_model
from soma_retargeter.pipelines.ik_objectives import (
    IKCenterOfMassHorizontal,
    IKMinimumJointAngle,
)

_LIMB_DATA_IDX_NAME = 0
_LIMB_DATA_IDX_EFFECTOR_INDICES = 1
_LIMB_DATA_IDX_HINT_REF = 2
_LIMB_DATA_IDX_HINT_OFFSET = 3


def _slerp_quaternions(
    start: np.ndarray, end: np.ndarray, blend: np.ndarray
) -> np.ndarray:
    dots = np.sum(start * end, axis=2, keepdims=True)
    end = np.where(dots < 0.0, -end, end)
    dots = np.clip(np.abs(dots), 0.0, 1.0)
    angles = np.arccos(dots)
    sin_angles = np.sin(angles)
    regular = sin_angles > 1.0e-6
    start_scale = np.where(
        regular,
        np.sin((1.0 - blend) * angles) / np.maximum(sin_angles, 1.0e-8),
        1.0 - blend,
    )
    end_scale = np.where(
        regular,
        np.sin(blend * angles) / np.maximum(sin_angles, 1.0e-8),
        blend,
    )
    result = start_scale * start + end_scale * end
    return result / np.maximum(
        np.linalg.norm(result, axis=2, keepdims=True), 1.0e-8
    )


@wp.kernel
def _copy_effector_targets_kernel(
    in_effectors: wp.array2d(dtype=wp.transform),
    in_effector_idx: wp.int32,
    in_body_q: wp.array2d(dtype=wp.transform),
    in_body_idx: wp.int32,
    in_foot_rotation_weights: wp.array2d(dtype=wp.float32),
    in_foot_idx: wp.int32,
    out_target_positions: wp.array1d(dtype=wp.vec3),
    out_target_rotations: wp.array1d(dtype=wp.vec4),
):
    env = wp.tid()
    tx = in_effectors[env, in_effector_idx]
    q = wp.transform_get_rotation(tx)
    if in_foot_idx >= 0:
        current_q = wp.transform_get_rotation(in_body_q[env, in_body_idx])
        q = wp.quat_slerp(
            current_q,
            q,
            in_foot_rotation_weights[env, in_foot_idx],
        )
    out_target_positions[env] = wp.transform_get_translation(tx)
    out_target_rotations[env] = wp.vec4(q[0], q[1], q[2], q[3])


@wp.kernel
def _compute_com_target_offsets_kernel(
    feet_targets: wp.array2d(dtype=wp.transform),
    com_target_positions: wp.array1d(dtype=wp.vec3),
    out_offsets: wp.array1d(dtype=wp.vec3),
):
    env = wp.tid()
    target_support_center = 0.5 * (
        wp.transform_get_translation(feet_targets[env, 0])
        + wp.transform_get_translation(feet_targets[env, 1])
    )
    out_offsets[env] = com_target_positions[env] - target_support_center


@wp.kernel
def _solve_two_bone_ik_batched_kernel(
    in_body_q: wp.array2d(dtype=wp.transform),
    in_joint_q: wp.array2d(dtype=wp.float32),
    in_pelvis_index: wp.int32,
    in_left_knee_coord: wp.int32,
    in_right_knee_coord: wp.int32,
    in_root_lift_start_rad: wp.float32,
    in_root_lift_full_rad: wp.float32,
    in_root_lift_max_m: wp.float32,
    in_root_lift_smoothing_alpha: wp.float32,
    inout_root_lift_weights: wp.array1d(dtype=wp.float32),
    in_num_ik_chains: wp.int32,
    in_chain_indices: wp.array2d(dtype=wp.int32),
    in_chain_parent_indices: wp.array1d(dtype=wp.int32),
    in_chain_hint_indices: wp.array1d(dtype=wp.int32),
    in_chain_hint_offsets: wp.array1d(dtype=wp.vec3),
    in_ik_targets: wp.array2d(dtype=wp.transform),
    out_result: wp.array2d(dtype=wp.transform),
):
    env = wp.tid()
    body_q = in_body_q[env]

    pelvis = body_q[in_pelvis_index]
    both_knees = wp.min(
        in_joint_q[env, in_left_knee_coord],
        in_joint_q[env, in_right_knee_coord],
    )
    desired_lift_weight = wp.clamp(
        (both_knees - in_root_lift_start_rad)
        / (in_root_lift_full_rad - in_root_lift_start_rad),
        0.0,
        1.0,
    )
    desired_lift_weight = desired_lift_weight * desired_lift_weight * (
        3.0 - 2.0 * desired_lift_weight
    )
    lift_weight = inout_root_lift_weights[env] + in_root_lift_smoothing_alpha * (
        desired_lift_weight - inout_root_lift_weights[env]
    )
    inout_root_lift_weights[env] = lift_weight
    pelvis_position = wp.transform_get_translation(pelvis)
    pelvis_rotation = wp.transform_get_rotation(pelvis)
    out_result[env, 0] = wp.transform(
        pelvis_position + wp.vec3(0.0, 0.0, in_root_lift_max_m * lift_weight),
        pelvis_rotation,
    )
    offset = wp.int32(1)
    for i in range(in_num_ik_chains):
        chain_indices = in_chain_indices[i]
        chain_hint_idx = in_chain_hint_indices[i]

        use_hint = chain_hint_idx != -1
        chain_hint_world = wp.vec3(0.0, 0.0, 0.0)
        if use_hint:
            chain_hint_world = wp.transform_point(
                body_q[chain_hint_idx], in_chain_hint_offsets[i]
            )

        result = ik_utils.wp_solve_two_bone_ik(
            1.0,
            body_q[in_chain_parent_indices[i]],
            body_q[chain_indices[0]],
            body_q[chain_indices[1]],
            body_q[chain_indices[2]],
            in_ik_targets[env, i],
            use_hint,
            chain_hint_world,
        )

        out_result[env, offset + 0] = result.root
        out_result[env, offset + 1] = result.mid
        out_result[env, offset + 2] = result.tip
        offset += wp.int32(3)


class FeetStabilizer:
    """
    FeetStabilizer class for managing inverse kinematics and feet stabilization for robotic motion transfer.
    """
    def __init__(self, config: str):
        """
        Initialize the feet stabilizer with the specified configuration.
        Args:
            config (str): Path to the configuration file.
        Raises:
            ValueError: If the robot type specified in the config is unknown.
        """
        self._load_config(config)

        self.robot_builder = robot_model.create_robot_builder(self.robot_type)

        self.num_body_count = self.robot_builder.body_count
        self.ik_model = self._build_model(1)

        body_names = [newton_utils.get_name_from_label(label) for label in self.robot_builder.body_label]
        self.effector_mapped_indices = [body_names.index(body_name) for (body_name, _) in self.effectors.items()]
        self.effector_weights = [wp.vec2(*tr_weights) for (_, tr_weights) in self.effectors.items()]
        effector_parent_indices = [self.robot_builder.joint_parent[idx] for idx in self.effector_mapped_indices]

        self.pelvis_idx = self.effector_mapped_indices[self.ik_root]
        joint_q_start = self.ik_model.joint_q_start.numpy()
        self.knee_coord_indices = (
            int(joint_q_start[body_names.index("left_knee_link")]),
            int(joint_q_start[body_names.index("right_knee_link")]),
        )
        self.two_bone_ik_chains = wp.array2d([[self.effector_mapped_indices[i] for i in limb[_LIMB_DATA_IDX_EFFECTOR_INDICES]] for limb in self.ik_limb_data], dtype=wp.int32)
        self.two_bone_ik_chain_parent = wp.array([effector_parent_indices[limb[_LIMB_DATA_IDX_EFFECTOR_INDICES][0]] for limb in self.ik_limb_data], dtype=wp.int32)
        self.two_bone_ik_hint_references = wp.array([self.effector_mapped_indices[limb[_LIMB_DATA_IDX_HINT_REF]] for limb in self.ik_limb_data], dtype=wp.int32)
        self.two_bone_ik_hint_offsets = wp.array([limb[_LIMB_DATA_IDX_HINT_OFFSET] for limb in self.ik_limb_data], dtype=wp.vec3)
        self.effector_foot_indices = [-1] * self.num_effectors
        for foot_idx, limb in enumerate(self.ik_limb_data):
            self.effector_foot_indices[
                limb[_LIMB_DATA_IDX_EFFECTOR_INDICES][-1]
            ] = foot_idx

        self.num_envs = -1

    def prepare_foot_targets(
        self, targets: np.ndarray, sample_rate: float = 120.0
    ) -> tuple[np.ndarray, np.ndarray]:
        """Flatten stationary near-ground feet and smoothly track their rotation."""
        targets = np.asarray(targets, dtype=np.float32)
        if targets.ndim != 3 or targets.shape[1:] != (len(self.ik_limb_data), 7):
            raise ValueError(
                "foot targets must have shape "
                f"[frames, {len(self.ik_limb_data)}, 7], got {targets.shape}"
            )
        if not self.ground_contact_rotation_enabled:
            return targets, np.ones(targets.shape[:2], dtype=np.float32)
        if not np.isfinite(sample_rate) or sample_rate <= 0.0:
            raise ValueError(f"sample_rate must be positive and finite, got {sample_rate}")

        prepared = targets.copy()
        detection_positions = prepared[:, :, :3]
        if self.ground_contact_detection_smoothing_sigma_frames > 0.0:
            detection_positions = gaussian_filter1d(
                detection_positions,
                sigma=self.ground_contact_detection_smoothing_sigma_frames,
                axis=0,
                mode="nearest",
                truncate=4.0,
            )
        heights = prepared[:, :, 2].copy()
        quaternions = prepared[:, :, 3:7]
        norms = np.linalg.norm(quaternions, axis=2, keepdims=True)
        quaternions = quaternions / np.maximum(norms, 1.0e-8)

        def support_min_heights(
            positions_z: np.ndarray, candidate_quaternions: np.ndarray
        ) -> np.ndarray:
            if self.ground_contact_support_vertices_local_m is None:
                return positions_z
            qx, qy, qz, qw = np.moveaxis(candidate_quaternions, -1, 0)
            world_z_rows = np.stack(
                (
                    2.0 * (qx * qz - qy * qw),
                    2.0 * (qy * qz + qx * qw),
                    1.0 - 2.0 * (qx * qx + qy * qy),
                ),
                axis=2,
            )
            rotated_z = np.einsum(
                "fsc,svc->fsv",
                world_z_rows,
                self.ground_contact_support_vertices_local_m,
                optimize=True,
            )
            return positions_z + np.min(rotated_z, axis=2)

        detection_heights = support_min_heights(
            detection_positions[:, :, 2], quaternions
        )
        if (
            self.ground_contact_support_vertices_local_m is not None
            and self.ground_contact_detection_smoothing_sigma_frames > 0.0
        ):
            detection_heights = gaussian_filter1d(
                detection_heights,
                sigma=self.ground_contact_detection_smoothing_sigma_frames,
                axis=0,
                mode="nearest",
                truncate=4.0,
            )
        floor_height = float(np.percentile(detection_heights, 2.0))
        relative_height = detection_heights - floor_height
        transition = (
            self.ground_contact_rotation_release_height_m
            - self.ground_contact_rotation_full_height_m
        )
        weights = np.clip(
            (
                self.ground_contact_rotation_release_height_m
                - relative_height
            )
            / transition,
            0.0,
            1.0,
        )
        weights = weights * weights * (3.0 - 2.0 * weights)

        horizontal_speed = np.zeros(heights.shape, dtype=np.float32)
        if len(prepared) > 1:
            segment_speed = (
                np.linalg.norm(np.diff(detection_positions[:, :, :2], axis=0), axis=2)
                * float(sample_rate)
            )
            horizontal_speed[:-1] = segment_speed
            horizontal_speed[1:] = np.maximum(
                horizontal_speed[1:], segment_speed
            )
        speed_transition = (
            self.ground_contact_rotation_release_speed_mps
            - self.ground_contact_rotation_full_speed_mps
        )
        speed_weights = np.clip(
            (
                self.ground_contact_rotation_release_speed_mps
                - horizontal_speed
            )
            / speed_transition,
            0.0,
            1.0,
        )
        speed_weights = speed_weights * speed_weights * (3.0 - 2.0 * speed_weights)
        if self.ground_contact_temporal_smoothing_sigma_frames > 0.0:
            weights = gaussian_filter1d(
                weights,
                sigma=self.ground_contact_temporal_smoothing_sigma_frames,
                axis=0,
                mode="nearest",
                truncate=4.0,
            )
            speed_weights = gaussian_filter1d(
                speed_weights,
                sigma=self.ground_contact_temporal_smoothing_sigma_frames,
                axis=0,
                mode="nearest",
                truncate=4.0,
            )
        weights *= speed_weights
        source_up_z = 1.0 - 2.0 * (
            quaternions[:, :, 0] * quaternions[:, :, 0]
            + quaternions[:, :, 1] * quaternions[:, :, 1]
        )
        source_tilt_deg = np.rad2deg(
            np.arccos(np.clip(source_up_z, -1.0, 1.0))
        )
        tilt_transition = (
            self.ground_contact_rotation_release_tilt_deg
            - self.ground_contact_rotation_full_tilt_deg
        )
        level_weights = np.clip(
            (
                self.ground_contact_rotation_release_tilt_deg
                - source_tilt_deg
            )
            / tilt_transition,
            0.0,
            1.0,
        )
        level_weights = level_weights * level_weights * (3.0 - 2.0 * level_weights)
        motion_weights = 1.0 - speed_weights
        prepared[:, :, 2] = (
            heights * (1.0 - weights)
            + self.ground_contact_rotation_target_height_m * weights
        )
        minimum_swing_height = (
            self.ground_contact_rotation_target_height_m
            + self.ground_contact_minimum_swing_height_m * motion_weights
        )
        prepared[:, :, 2] = np.maximum(
            prepared[:, :, 2], minimum_swing_height
        )

        x, y, z, w = np.moveaxis(quaternions, -1, 0)
        yaw = np.arctan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y * y + z * z),
        )
        half_yaw = 0.5 * yaw
        level = np.zeros_like(quaternions)
        level[:, :, 2] = np.sin(half_yaw)
        level[:, :, 3] = np.cos(half_yaw)

        blend = (weights * level_weights)[:, :, None]
        conditioned_quaternions = _slerp_quaternions(quaternions, level, blend)
        if self.ground_contact_support_vertices_local_m is not None:
            current_min = support_min_heights(
                prepared[:, :, 2], conditioned_quaternions
            )
            prepared[:, :, 2] += np.maximum(
                0.0,
                self.ground_contact_target_clearance_m - current_min,
            )
        prepared[:, :, 3:7] = conditioned_quaternions
        return prepared, weights.astype(np.float32)

    def setup_num_envs(self, num_envs):
        """
        Initialize the setup for the feet stabilizer with the specified number of environments.

        This method configures the model, state, joint parameters, and effectors for inverse kinematics
        computation across multiple parallel environments. It also initializes the objectives and solver.

        Args:
            num_envs (int): The number of parallel environments to set up.
        """
        self.num_envs = num_envs
        self.model = self._build_model(num_envs)
        self.state = self.model.state()
        self.joint_q = wp.array(self.model.joint_q, shape=(self.num_envs, self.ik_model.joint_coord_count))
        self._joint_q_flat = self.joint_q.reshape((self.model.joint_coord_count,))
        self.out_effectors = wp.empty(shape=[self.num_envs, self.num_effectors], dtype=wp.transform)
        self.com_target_offsets = wp.zeros(self.num_envs, dtype=wp.vec3)
        self.full_foot_rotation_weights = wp.ones(
            shape=(self.num_envs, len(self.ik_limb_data)), dtype=wp.float32
        )
        self.deep_knee_root_lift_weights = wp.zeros(
            shape=self.num_envs, dtype=wp.float32
        )
        self.reset_state()
        self._create_objectives_and_solver()

    def reset_state(self, joint_q=None):
        """
        Resets the current state of the feet stabilizer, optionally with a provided joint configuration..

        Args:
            joint_q (wp.array, optional): The joint configuration to reset to. If None, the default configuration is used.

        Raises:
            ValueError: If the provided joint_q does not match the expected shape for the model's joint coordinates.
        """
        assert self.num_envs != -1, "[ERROR]: Environments have not been initialized. Call setup_num_envs to create a valid model."
        if joint_q is not None:
            if joint_q.shape != self.joint_q.shape:
                raise ValueError(f"[ERROR]: joint_q size mismatch. Expected joint_q shape of [{self.joint_q.shape}] but received [{joint_q.shape}]")

            wp.copy(self.joint_q, joint_q)

        newton.eval_fk(self.model, self._joint_q_flat, self.model.joint_qd, self.state)

    def current_state(self):
        """Returns the current joint configuration of the model."""
        return self.joint_q

    def solve(self, targets_tx):
        """
        Solves the inverse kinematics problem for the specified target transforms of the effectors.

        Args:
            targets_tx (np.ndarray): An array of shape (num_envs, num_effectors, wp.transform) containing the target transforms for each effector in each environment.

        Raises:
            ValueError: If the number of environments has not been initialized or if the shape of targets_tx
                    does not match the expected shape based on the number of environments and effectors.
        """
        assert self.num_envs != -1, "[ERROR]: Environments have not been initialized. Call setup_num_envs to create a valid model."
        if targets_tx.shape != (self.num_envs, self.two_bone_ik_chains.shape[0], 7):
            raise ValueError(f"[ERROR]: targets_tx size mismatch. Expected targets_tx shape is [{(self.num_envs, self.two_bone_ik_chains.shape[0], 7)}] but received [{targets_tx.shape}]")

        self.solve_warp(wp.array2d(targets_tx, dtype=wp.transform))

    def solve_warp(
        self,
        targets_tx,
        com_target_positions=None,
        com_weights=None,
        foot_rotation_weights=None,
    ):
        """
        Solves the inverse kinematics problem from a Warp transform array.
        """
        assert self.num_envs != -1, "[ERROR]: Environments have not been initialized. Call setup_num_envs to create a valid model."
        if targets_tx.shape != (self.num_envs, self.two_bone_ik_chains.shape[0]):
            raise ValueError(f"[ERROR]: targets_tx size mismatch. Expected targets_tx shape is [{(self.num_envs, self.two_bone_ik_chains.shape[0])}] but received [{targets_tx.shape}]")
        if foot_rotation_weights is None:
            foot_rotation_weights = self.full_foot_rotation_weights
        expected_rotation_weight_shape = (
            self.num_envs,
            self.two_bone_ik_chains.shape[0],
        )
        if foot_rotation_weights.shape != expected_rotation_weight_shape:
            raise ValueError(
                "[ERROR]: foot_rotation_weights size mismatch. Expected "
                f"{expected_rotation_weight_shape}, received "
                f"{foot_rotation_weights.shape}"
            )

        if self.center_of_mass_objective is not None:
            if com_target_positions is None or com_weights is None:
                self.center_of_mass_objective.frame_weights.zero_()
            else:
                wp.launch(
                    _compute_com_target_offsets_kernel,
                    dim=self.num_envs,
                    inputs=[targets_tx, com_target_positions],
                    outputs=[self.com_target_offsets],
                )
                self.center_of_mass_objective.set_targets(
                    self.com_target_offsets, com_weights
                )

        wp.launch(
            _solve_two_bone_ik_batched_kernel,
            dim=self.num_envs,
            inputs=[
                self.state.body_q.reshape(shape=[self.num_envs, self.num_body_count]),
                self.joint_q,
                self.pelvis_idx,
                self.knee_coord_indices[0],
                self.knee_coord_indices[1],
                self.deep_knee_root_lift_start_rad,
                self.deep_knee_root_lift_full_rad,
                self.deep_knee_root_lift_max_m,
                self.deep_knee_root_lift_smoothing_alpha,
                self.deep_knee_root_lift_weights,
                self.two_bone_ik_chains.shape[0],
                self.two_bone_ik_chains,
                self.two_bone_ik_chain_parent,
                self.two_bone_ik_hint_references,
                self.two_bone_ik_hint_offsets,
                targets_tx],
                outputs=[self.out_effectors])

        for i in range(self.num_effectors):
            body_idx = self.effector_mapped_indices[i]
            wp.launch(
                _copy_effector_targets_kernel,
                dim=self.num_envs,
                inputs=[
                    self.out_effectors,
                    i,
                    self.state.body_q.reshape(
                        shape=[self.num_envs, self.num_body_count]
                    ),
                    body_idx,
                    foot_rotation_weights,
                    self.effector_foot_indices[i],
                ],
                outputs=[
                    self.position_objectives[i].target_positions,
                    self.rotation_objectives[i].target_rotations,
                ])

        if self.captured_graph is not None:
            wp.capture_launch(self.captured_graph)
        else:
            self.ik_solver.step(self.joint_q, self.joint_q, iterations=self.ik_iterations)

    def _load_config(self, config: str):
        data = io_utils.load_json(config)
        self.robot_type = data['robot_type']
        self.ik_iterations = data['ik_iterations']
        self.joint_limit_weight = data['joint_limit_weight']
        self.center_of_mass_weight = max(
            0.0, float(data.get('center_of_mass_weight', 0.0))
        )
        ground_rotation = data.get("ground_contact_foot_rotation", {})
        self.ground_contact_rotation_enabled = bool(
            ground_rotation.get("enabled", False)
        )
        self.ground_contact_rotation_full_height_m = max(
            0.0, float(ground_rotation.get("full_height_margin_m", 0.035))
        )
        self.ground_contact_rotation_release_height_m = max(
            self.ground_contact_rotation_full_height_m + 1.0e-6,
            float(ground_rotation.get("release_height_margin_m", 0.08)),
        )
        self.ground_contact_rotation_target_height_m = float(
            ground_rotation.get("target_height_m", 0.001)
        )
        self.ground_contact_rotation_full_speed_mps = max(
            0.0,
            float(ground_rotation.get("full_horizontal_speed_mps", 0.03)),
        )
        self.ground_contact_rotation_release_speed_mps = max(
            self.ground_contact_rotation_full_speed_mps + 1.0e-6,
            float(ground_rotation.get("release_horizontal_speed_mps", 0.12)),
        )
        self.ground_contact_rotation_full_tilt_deg = max(
            0.0, float(ground_rotation.get("full_tilt_deg", 180.0))
        )
        self.ground_contact_rotation_release_tilt_deg = max(
            self.ground_contact_rotation_full_tilt_deg + 1.0e-6,
            float(ground_rotation.get("release_tilt_deg", 181.0)),
        )
        self.ground_contact_minimum_swing_height_m = max(
            0.0,
            float(ground_rotation.get("minimum_swing_height_m", 0.0)),
        )
        self.ground_contact_root_drop_m = max(
            0.0, float(ground_rotation.get("root_drop_m", 0.0))
        )
        self.ground_contact_temporal_smoothing_sigma_frames = max(
            0.0,
            float(ground_rotation.get("temporal_smoothing_sigma_frames", 0.0)),
        )
        self.ground_contact_detection_smoothing_sigma_frames = max(
            0.0,
            float(
                ground_rotation.get(
                    "detection_smoothing_sigma_frames",
                    self.ground_contact_temporal_smoothing_sigma_frames,
                )
            ),
        )
        support_vertices = ground_rotation.get("support_vertices_local_m")
        self.ground_contact_support_vertices_local_m = None
        if support_vertices is not None:
            support_vertices = np.asarray(support_vertices, dtype=np.float32)
            expected_prefix = (len(data["ik_limbs"]),)
            if (
                support_vertices.ndim != 3
                or support_vertices.shape[:1] != expected_prefix
                or support_vertices.shape[2] != 3
                or support_vertices.shape[1] < 3
                or not np.isfinite(support_vertices).all()
            ):
                raise ValueError(
                    "support_vertices_local_m must have shape [feet, vertices, 3]"
                )
            self.ground_contact_support_vertices_local_m = support_vertices
        self.ground_contact_target_clearance_m = float(
            ground_rotation.get("target_clearance_m", 0.001)
        )

        deep_knee_lift = data.get("deep_knee_root_lift", {})
        lift_start_deg = float(deep_knee_lift.get("start_deg", 140.0))
        lift_full_deg = max(
            lift_start_deg + 1.0e-3,
            float(deep_knee_lift.get("full_deg", 144.0)),
        )
        self.deep_knee_root_lift_start_rad = float(np.deg2rad(lift_start_deg))
        self.deep_knee_root_lift_full_rad = float(np.deg2rad(lift_full_deg))
        self.deep_knee_root_lift_max_m = (
            max(0.0, float(deep_knee_lift.get("max_lift_m", 0.0)))
            if bool(deep_knee_lift.get("enabled", False))
            else 0.0
        )
        self.deep_knee_root_lift_smoothing_alpha = float(
            np.clip(deep_knee_lift.get("smoothing_alpha", 1.0), 1.0e-4, 1.0)
        )
        minimum_knee_bend = data.get("minimum_knee_bend", {})
        self.minimum_knee_bend_enabled = bool(
            minimum_knee_bend.get("enabled", False)
        )
        self.minimum_knee_bend_target_rad = float(
            np.deg2rad(minimum_knee_bend.get("target_deg", 5.0))
        )
        self.minimum_knee_bend_weight = max(
            0.0, float(minimum_knee_bend.get("weight", 0.0))
        )
        self.minimum_knee_bend_transition_rad = float(
            np.deg2rad(
                max(0.0, float(minimum_knee_bend.get("transition_deg", 0.0)))
            )
        )

        self.effectors = data['effectors']
        self.num_effectors = len(self.effectors)

        self.ik_root = data['ik_root']
        self.ik_limb_data = []
        for label, values in data['ik_limbs'].items():
            self.ik_limb_data.append([label, values['effectors'], values['hint_reference'], wp.vec3(*values['hint_offset'])])

    def _create_objectives_and_solver(self):
        body_q_np = self.state.body_q.numpy().reshape(self.num_envs, self.num_body_count, 7)
        pos_effector_arrays, rot_effector_arrays = [], []
        for i in range(self.num_effectors):
            body_idx = self.effector_mapped_indices[i]
            pos_effector_arrays.append(wp.array(body_q_np[:, body_idx, 0:3], dtype=wp.vec3))
            rot_effector_arrays.append(wp.array(body_q_np[:, body_idx, 3:7], dtype=wp.vec4))

        self.position_objectives = []
        self.rotation_objectives = []
        for i in range(self.num_effectors):
            body_idx = self.effector_mapped_indices[i]
            t_weight = self.effector_weights[i][0]
            r_weight = self.effector_weights[i][1]
            self.position_objectives.append(
                newton.ik.IKObjectivePosition(
                    link_index=body_idx,
                    link_offset=wp.vec3(0.0, 0.0, 0.0),
                    target_positions=pos_effector_arrays[i],
                    weight=t_weight
                    )
                )
            self.rotation_objectives.append(
                newton.ik.IKObjectiveRotation(
                    link_index=body_idx,
                    link_offset_rotation=wp.quat_identity(),
                    target_rotations=rot_effector_arrays[i],
                    weight=r_weight
                    )
                )

        # Joint limit objective
        self.joint_limit_objective = newton.ik.IKObjectiveJointLimit(
            joint_limit_lower=self.ik_model.joint_limit_lower,
            joint_limit_upper=self.ik_model.joint_limit_upper,
            weight=self.joint_limit_weight)

        self.center_of_mass_objective = None
        if self.center_of_mass_weight > 0.0:
            support_body_indices = tuple(
                self.effector_mapped_indices[limb[_LIMB_DATA_IDX_EFFECTOR_INDICES][-1]]
                for limb in self.ik_limb_data
            )
            self.center_of_mass_objective = IKCenterOfMassHorizontal(
                target_positions=wp.zeros(self.num_envs, dtype=wp.vec3),
                frame_weights=wp.zeros(self.num_envs, dtype=wp.float32),
                weight=self.center_of_mass_weight,
                support_body_indices=support_body_indices,
            )

        self.minimum_knee_bend_objective = None
        if self.minimum_knee_bend_enabled and self.minimum_knee_bend_weight > 0.0:
            targets = np.zeros(self.ik_model.joint_coord_count, dtype=np.float32)
            masks = np.zeros(self.ik_model.joint_coord_count, dtype=np.float32)
            for coord in self.knee_coord_indices:
                targets[coord] = self.minimum_knee_bend_target_rad
                masks[coord] = 1.0
            self.minimum_knee_bend_objective = IKMinimumJointAngle(
                target_angles=targets,
                coord_masks=masks,
                n_dofs=self.ik_model.joint_dof_count,
                weight=self.minimum_knee_bend_weight,
                transition_angle=self.minimum_knee_bend_transition_rad,
            )

        active_objectives = [
            *self.position_objectives,
            *self.rotation_objectives,
            self.joint_limit_objective,
        ]
        if self.center_of_mass_objective is not None:
            active_objectives.append(self.center_of_mass_objective)
        if self.minimum_knee_bend_objective is not None:
            active_objectives.append(self.minimum_knee_bend_objective)

        self.ik_solver = newton.ik.IKSolver(
            model=self.ik_model,
            objectives=active_objectives,
            lambda_initial=0.1,
            n_problems=self.num_envs,
            jacobian_mode=newton.ik.IKJacobianType.ANALYTIC)

        self.ik_solver.reset()
        self.captured_graph = None
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as cap:
                self.ik_solver.step(self.joint_q, self.joint_q, iterations=self.ik_iterations)
            self.captured_graph = cap.graph

    def _build_model(self, num_envs: int):
        builder = newton.ModelBuilder()
        for _ in range(num_envs):
            builder.add_builder(self.robot_builder, xform=wp.transform_identity())
        builder.add_ground_plane()
        return builder.finalize()
