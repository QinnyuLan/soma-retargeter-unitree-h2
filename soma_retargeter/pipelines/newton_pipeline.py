# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import warp as wp
import numpy as np
import newton
import newton.ik as ik
from tqdm import trange

import soma_retargeter.assets.bvh as bvh_utils
import soma_retargeter.utils.newton_utils as newton_utils
import soma_retargeter.utils.io_utils as io_utils
import soma_retargeter.pipelines.utils as pipeline_utils
import soma_retargeter.robotics.robot_model as robot_model
from soma_retargeter.pipelines.ik_objectives import (
    IKCenterOfMassHorizontal,
    IKSmoothJointFilter,
)
from soma_retargeter.animation.skeleton import Skeleton, SkeletonInstance
from soma_retargeter.animation.animation_buffer import AnimationBuffer
from soma_retargeter.robotics.human_to_robot_scaler import HumanToRobotScaler
from soma_retargeter.robotics.csv_animation_buffer import CSVAnimationBuffer
from soma_retargeter.pipelines.feet_stabilizer import FeetStabilizer
from soma_retargeter.pipelines.joint_limit_clamper import JointLimitClamper

_DEFAULT_IK_SOLVER_ITERATIONS = 24
_DEFAULT_JOINT_LIMIT_OBJECTIVE_WEIGHT = 10.0
_DEFAULT_SMOOTH_JOINT_FILTER_OBJECTIVE_WEIGHT = 5.5
_DEFAULT_NUM_INITIALIZATION_FRAMES = 10
_DEFAULT_NUM_STABILIZATION_FRAMES = 5


@wp.kernel
def _copy_joint_q_frame_kernel(
    in_joint_q: wp.array2d(dtype=wp.float32),
    in_frame: wp.int32,
    out_joint_q_frames: wp.array3d(dtype=wp.float32),
):
    env, coord = wp.tid()
    out_joint_q_frames[env, in_frame, coord] = in_joint_q[env, coord]


@wp.kernel
def _copy_ik_targets_frame_kernel(
    in_targets: wp.array3d(dtype=wp.transform),
    in_frame: wp.int32,
    in_last_frames: wp.array1d(dtype=wp.int32),
    in_target_effector_idx: wp.int32,
    out_target_positions: wp.array1d(dtype=wp.vec3),
    out_target_rotations: wp.array1d(dtype=wp.vec4),
):
    env = wp.tid()
    frame = wp.min(in_frame, in_last_frames[env])
    tx = in_targets[env, frame, in_target_effector_idx]
    q = wp.transform_get_rotation(tx)
    out_target_positions[env] = wp.transform_get_translation(tx)
    out_target_rotations[env] = wp.vec4(q[0], q[1], q[2], q[3])


@wp.kernel
def _copy_feet_targets_frame_kernel(
    in_feet_targets: wp.array3d(dtype=wp.transform),
    in_rotation_weights: wp.array3d(dtype=wp.float32),
    in_frame: wp.int32,
    in_last_frames: wp.array1d(dtype=wp.int32),
    out_feet_targets: wp.array2d(dtype=wp.transform),
    out_rotation_weights: wp.array2d(dtype=wp.float32),
):
    env, foot = wp.tid()
    frame = wp.min(in_frame, in_last_frames[env])
    out_feet_targets[env, foot] = in_feet_targets[env, frame, foot]
    out_rotation_weights[env, foot] = in_rotation_weights[env, frame, foot]


@wp.kernel
def _copy_static_com_targets_frame_kernel(
    in_targets: wp.array3d(dtype=wp.transform),
    in_weights: wp.array2d(dtype=wp.float32),
    in_frame: wp.int32,
    in_last_frames: wp.array1d(dtype=wp.int32),
    in_root_effector_idx: wp.int32,
    in_left_foot_effector_idx: wp.int32,
    in_right_foot_effector_idx: wp.int32,
    in_forward_offset_m: wp.float32,
    out_positions: wp.array1d(dtype=wp.vec3),
    out_weights: wp.array1d(dtype=wp.float32),
):
    env = wp.tid()
    frame = wp.min(in_frame, in_last_frames[env])
    left = wp.transform_get_translation(in_targets[env, frame, in_left_foot_effector_idx])
    right = wp.transform_get_translation(in_targets[env, frame, in_right_foot_effector_idx])
    root_rotation = wp.transform_get_rotation(in_targets[env, frame, in_root_effector_idx])
    forward = wp.quat_rotate(root_rotation, wp.vec3(1.0, 0.0, 0.0))
    horizontal_forward = wp.normalize(wp.vec3(forward[0], forward[1], 0.0))
    out_positions[env] = (left + right) * 0.5 + horizontal_forward * in_forward_offset_m
    out_weights[env] = in_weights[env, frame]


class NewtonPipeline:
    """
    Newton-based motion retargeting pipeline.

    This pipeline retargets human motion captured on a common skeleton
    to a target robot using inverse kinematics (IK),
    custom objectives, and optional post-processing filters such as
    joint limit clamping and feet stabilization.
    """
    def __init__(
        self,
        skeleton: Skeleton,
        source_type='soma',
        robot_type='unitree_g1',
        retarget_config: dict = None,
        show_progress: bool = True,
    ):
        """
        Initialize the Newton retargeting pipeline.

        Args:
            skeleton: Common skeleton definition used by the input clips to be retargeted.
            source_type: Source skeleton type name. Currently only "soma" is supported.
            robot_type: Target robot type name.
            retarget_config: Optional configuration dictionary. If None, a
                configuration is loaded from disk based on the source/target
                types.

        Raises:
            ValueError: If the target robot type is not supported.
        """
        self.source_type = pipeline_utils.get_source_type_from_str(source_type)
        self.target_type = pipeline_utils.get_target_type_from_str(robot_type)
        self.input_targets = []
        self.input_sample_rates = []
        self.input_static_com_weights = []
        self.max_frames = -1
        self.show_progress = show_progress

        if retarget_config is None:
            retargeter_config = pipeline_utils.get_retargeter_config(self.source_type, self.target_type)
        else:
            retargeter_config = retarget_config

        self.ik_iterations = retargeter_config.get('ik_iterations', _DEFAULT_IK_SOLVER_ITERATIONS)
        self.joint_limit_weight = retargeter_config.get('joint_limit_weight', _DEFAULT_JOINT_LIMIT_OBJECTIVE_WEIGHT)
        self.smooth_joint_filter_weight = retargeter_config.get('smooth_joint_filter_weight', _DEFAULT_SMOOTH_JOINT_FILTER_OBJECTIVE_WEIGHT)
        self.post_processing_enabled = retargeter_config.get('enable_post_processing', True)
        self.condition_foot_targets = bool(
            retargeter_config.get('condition_foot_targets', False)
        )
        self.enable_self_penetration = False
        self.smooth_joint_filter_coord_masks = None
        self.joint_limit_clamper = None

        static_com_config = retargeter_config.get('static_center_of_mass', {})
        self.static_com_weight = max(0.0, float(static_com_config.get('weight', 0.0)))
        self.static_com_enabled = bool(
            static_com_config.get('enabled', self.static_com_weight > 0.0)
        )
        self.static_com_velocity_threshold = max(
            0.0, float(static_com_config.get('velocity_threshold_mps', 0.05))
        )
        self.static_com_velocity_transition = max(
            0.0, float(static_com_config.get('velocity_transition_mps', 0.0))
        )
        self.static_com_smoothing_window_s = max(
            0.0, float(static_com_config.get('smoothing_window_s', 0.0))
        )
        self.static_com_stationary_hold_s = max(
            0.0, float(static_com_config.get('stationary_hold_s', 0.0))
        )
        self.static_com_moving_hold_s = max(
            0.0, float(static_com_config.get('moving_hold_s', 0.0))
        )
        self.static_com_forward_offset = float(static_com_config.get('forward_offset_m', 0.0))

        robot_type_str = pipeline_utils.get_target_str_from_type(self.target_type)
        self.robot_builder = robot_model.create_robot_builder(robot_type_str)

        self.human_robot_scaler = HumanToRobotScaler(
            skeleton, retargeter_config['model_height'], io_utils.get_config_file(retargeter_config['human_robot_scaler_config']))

        self.num_body_count = self.robot_builder.body_count
        self.num_dofs = self.robot_builder.joint_dof_count
        self.ik_model = self._build_model(1)

        (
            self.mapped_joints,
            self.mapped_joint_indices,
            self.mapped_body_link_pos_data,
            self.mapped_body_link_rot_data
        ) = self._build_target_mapping(
            self.ik_model,
            self.human_robot_scaler.skeleton,
            retargeter_config)

        smooth_joint_filter_objective_body_masks = retargeter_config.get('smooth_joint_filter_objective_body_masks', None)
        if smooth_joint_filter_objective_body_masks is not None:
            self.smooth_joint_filter_coord_masks = newton_utils.create_joint_coord_masks(
                self.ik_model, smooth_joint_filter_objective_body_masks, 0.0)

        effector_names = self.human_robot_scaler.effector_names()
        self.target_effector_indices = [effector_names.index(name) for name in self.mapped_joints]
        self.feet_effector_indices = [
            self.mapped_joints.index("LeftFoot"),
            self.mapped_joints.index("RightFoot")]
        self.root_effector_index = self.mapped_joints.index("Hips")

        self.feet_stabilizer = FeetStabilizer(io_utils.get_config_file(retargeter_config['feet_stabilizer_config']))
        self.joint_limit_clamper = JointLimitClamper(self.ik_model)

        self.initialization_pose = None
        self.num_initialization_frames = 0
        self.num_stabilization_frames = 0
        if (retargeter_config['initialization_pose']):
            init_skel, init_anim = bvh_utils.load_bvh(io_utils.get_config_file(retargeter_config['initialization_pose']))
            self.initialization_pose = SkeletonInstance(init_skel, [0, 0, 0], wp.transform_identity())
            self.initialization_pose.set_local_transforms(init_anim.get_local_transforms(0))
            self.num_initialization_frames = retargeter_config.get('num_initialization_frames', _DEFAULT_NUM_INITIALIZATION_FRAMES)
            self.num_stabilization_frames = retargeter_config.get('num_stabilization_frames', _DEFAULT_NUM_STABILIZATION_FRAMES)

    def clear(self):
        """
        Clear all accumulated input motions and reset internal state.

        This removes all previously added motions set for retargeting.
        It does not modify static configuration such as the robot model or IK settings.
        """
        self.input_targets = []
        self.input_sample_rates = []
        self.input_static_com_weights = []
        self.max_frames = -1

    def _compute_static_com_weights(self, targets: np.ndarray, sample_rate: float) -> np.ndarray:
        weights = np.ones(len(targets), dtype=np.float32)
        if not self.static_com_enabled or len(targets) <= 1:
            return weights

        tracked_indices = [self.root_effector_index, *self.feet_effector_indices]
        positions = targets[:, tracked_indices, 0:3]
        frame_speeds = np.linalg.norm(np.diff(positions, axis=0), axis=2) * float(sample_rate)
        speeds = np.zeros((len(targets), len(tracked_indices)), dtype=np.float32)
        speeds[1:] = frame_speeds
        speeds[:-1] = np.maximum(speeds[:-1], frame_speeds)
        max_speed = np.max(speeds, axis=1)

        transition = min(
            self.static_com_velocity_transition,
            self.static_com_velocity_threshold,
        )
        stationary_threshold = self.static_com_velocity_threshold - transition
        stationary_hold_frames = max(
            1, int(round(self.static_com_stationary_hold_s * sample_rate))
        )
        moving_hold_frames = max(
            1, int(round(self.static_com_moving_hold_s * sample_rate))
        )
        weights = np.empty(len(max_speed), dtype=np.float32)
        is_static = bool(max_speed[0] <= stationary_threshold)
        candidate_frames = 0
        for frame, speed in enumerate(max_speed):
            if is_static:
                candidate_frames = (
                    candidate_frames + 1
                    if speed >= self.static_com_velocity_threshold
                    else 0
                )
                if candidate_frames >= moving_hold_frames:
                    is_static = False
                    candidate_frames = 0
            else:
                candidate_frames = candidate_frames + 1 if speed <= stationary_threshold else 0
                if candidate_frames >= stationary_hold_frames:
                    is_static = True
                    candidate_frames = 0
            weights[frame] = float(is_static)

        window_frames = int(round(self.static_com_smoothing_window_s * sample_rate))
        if window_frames > 1:
            if window_frames % 2 == 0:
                window_frames += 1
            window = np.hanning(window_frames + 2)[1:-1]
            window /= np.sum(window)
            radius = window_frames // 2
            weights = np.convolve(
                np.pad(weights, (radius, radius), mode="edge"),
                window,
                mode="valid",
            )
        return np.asarray(weights, dtype=np.float32)

    def add_input_targets(
        self,
        targets: np.ndarray,
        sample_rate: float,
        add_initialization_frames: bool = True,
    ) -> None:
        """Add robot-aligned effector transforms without passing through the human scaler."""
        targets = np.asarray(targets, dtype=np.float32)
        expected_shape = (len(self.mapped_joints), 7)
        if targets.ndim != 3 or targets.shape[1:] != expected_shape:
            raise ValueError(
                f"targets must have shape (frames, {expected_shape[0]}, 7), got {targets.shape}"
            )
        if len(targets) == 0 or not np.all(np.isfinite(targets)):
            raise ValueError("targets must contain at least one finite frame")
        if not np.isfinite(sample_rate) or sample_rate <= 0.0:
            raise ValueError(f"sample_rate must be positive and finite, got {sample_rate}")

        if add_initialization_frames:
            padding = self.num_initialization_frames + self.num_stabilization_frames
            if padding > 0:
                targets = np.concatenate(
                    [np.repeat(targets[:1], padding, axis=0), targets], axis=0
                )

        self.max_frames = max(self.max_frames, len(targets))
        self.input_targets.append(targets)
        self.input_sample_rates.append(float(sample_rate))
        self.input_static_com_weights.append(
            self._compute_static_com_weights(targets, float(sample_rate))
        )

    def add_input_motions(self, buffers: list[AnimationBuffer], offsets: list[wp.transform], scale_animation: bool):
        """
        Add input motions to be retargeted.
        Each buffer is converted into IK targets using the human-to-robot scaler.

        Args:
            buffers: List of input animation buffers defined on the common skeleton.
            offsets: List of root transforms applied to each buffer. If the
                length does not match `buffers`, identity transforms are used
                for all.
            scale_animation: Whether to rescale the source motion using the
                configured HumanToRobotScaler.
        """
        offsets = offsets if len(offsets) == len(buffers) else [wp.transform_identity()] * len(buffers)
        for i in trange(len(buffers), desc="[INFO] Converting Motions for Newton", disable=not self.show_progress):
            buffer = buffers[i]
            if self.initialization_pose and self.num_initialization_frames > 0:
                buffer = newton_utils.create_buffer_with_initialization_frames(
                    self.initialization_pose, buffers[i], self.num_initialization_frames, self.num_stabilization_frames)

            self.max_frames = max(self.max_frames, buffer.num_frames)
            buffer_effectors = self.human_robot_scaler.compute_effectors_from_buffer(buffer, scale_animation, offsets[i])
            targets = buffer_effectors[:, self.target_effector_indices, :]
            self.input_targets.append(targets)
            self.input_sample_rates.append(buffers[i].sample_rate)
            self.input_static_com_weights.append(
                self._compute_static_com_weights(targets, buffers[i].sample_rate)
            )

    def execute(self):
        """
        Run the retargeting pipeline on all added input motions.

        This method builds a multi-environment Newton model, sets up IK
        objectives, and performs frame-by-frame IK solving.

        Returns:
            list[CSVAnimationBuffer]: A list of retargeted robot motions, one per input motion.
        """
        num_envs = len(self.input_targets)
        if num_envs == 0:
            self.retargeted_motions = []
            return

        # Clamp objective weights to valid values
        self.ik_iterations = max(1, self.ik_iterations)
        self.joint_limit_weight = max(0.0, self.joint_limit_weight)
        self.smooth_joint_filter_weight = max(0.0, self.smooth_joint_filter_weight)

        print("[INFO] Newton Retargeter Settings: ")
        print(f"[INFO]\t  Source Skeleton Type: {pipeline_utils.get_source_str_from_type(self.source_type)}")
        print(f"[INFO]\t  Target Robot Type: {pipeline_utils.get_target_str_from_type(self.target_type)}")
        print(f"[INFO]\t  Post-Processing Enabled: {self.post_processing_enabled}")
        print(f"[INFO]\t  Foot Target Conditioning: {self.condition_foot_targets}")
        print(f"[INFO]\t  Initialization Pose: {self.initialization_pose is not None}")
        print(f"[INFO]\t  Initialization Frame Count: {self.num_initialization_frames}")
        print(f"[INFO]\t  Constraint Stabilization Frame Count: {self.num_stabilization_frames}")
        print(f"[INFO]\t  IK Solver Iterations: {self.ik_iterations}")
        print(f"[INFO]\t  Joint Limit Objective Weight: {self.joint_limit_weight}")
        print(f"[INFO]\t  Smooth Joint Filter Objective Weight: {self.smooth_joint_filter_weight}")

        model = self._build_model(num_envs)
        state = model.state()

        if self.post_processing_enabled:
            self.feet_stabilizer.setup_num_envs(num_envs)
            feet_targets = wp.empty(shape=(num_envs, len(self.feet_effector_indices)), dtype=wp.transform)
            feet_rotation_weights = wp.empty(
                shape=(num_envs, len(self.feet_effector_indices)),
                dtype=wp.float32,
            )

        (
            position_objectives,
            rotation_objectives,
            joint_limit_objective,
            smooth_joint_filter_objective,
            static_com_objective,
        ) = self._create_ik_objectives(num_envs, model, state)

        # Add optional objectives
        ik_solver_active_objectives = [*position_objectives, *rotation_objectives]
        if self.joint_limit_weight > 0.0:
            ik_solver_active_objectives.append(joint_limit_objective)
        if self.smooth_joint_filter_weight > 0.0:
            ik_solver_active_objectives.append(smooth_joint_filter_objective)
        if static_com_objective is not None and self.static_com_weight > 0.0:
            ik_solver_active_objectives.append(static_com_objective)
        ik_solver = ik.IKSolver(
            model=self.ik_model,
            n_problems=num_envs,
            objectives=ik_solver_active_objectives,
            lambda_initial=0.1,
            jacobian_mode=ik.IKJacobianType.ANALYTIC)

        joint_q = wp.empty(shape=(num_envs, self.ik_model.joint_coord_count))
        wp.copy(joint_q, model.joint_q)

        # Solver initialization
        ik_solver.reset()

        graph_capture = None

        def single_step():
            ik_solver.step(joint_q, joint_q, iterations=self.ik_iterations)

        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as cap:
                single_step()
            graph_capture = cap.graph
        else:
            ik_solver.step(joint_q, joint_q, iterations=self.ik_iterations)

        #import time
        num_frames_to_remove = self.num_initialization_frames + self.num_stabilization_frames
        output_frame_counts = [
            max(0, len(self.input_targets[i]) - num_frames_to_remove)
            for i in range(num_envs)]
        max_output_frames = max(output_frame_counts) if output_frame_counts else 0
        joint_q_frames = wp.empty(
            shape=(num_envs, max_output_frames, self.ik_model.joint_coord_count),
            dtype=wp.float32)
        target_frame_counts = np.array([len(targets) for targets in self.input_targets], dtype=np.int32)
        last_target_frames = wp.array(target_frame_counts - 1, dtype=wp.int32)
        target_sequences = self.input_targets
        prepared_feet = None
        if self.post_processing_enabled or self.condition_foot_targets:
            prepared_feet = [
                self.feet_stabilizer.prepare_foot_targets(
                    targets[:, self.feet_effector_indices],
                    self.input_sample_rates[index],
                )
                for index, targets in enumerate(self.input_targets)
            ]
        if self.condition_foot_targets:
            assert prepared_feet is not None
            target_sequences = []
            for targets, (foot_targets, _) in zip(
                self.input_targets, prepared_feet, strict=True
            ):
                conditioned = targets.copy()
                conditioned[:, self.feet_effector_indices] = foot_targets
                target_sequences.append(conditioned)

        if self.post_processing_enabled:
            assert prepared_feet is not None
            if self.feet_stabilizer.ground_contact_root_drop_m > 0.0:
                dropped_sequences = []
                for targets, (_, rotation_weights) in zip(
                    target_sequences, prepared_feet, strict=True
                ):
                    conditioned = targets.copy()
                    contact_weight = np.max(rotation_weights, axis=1)
                    conditioned[:, :, 2] -= (
                        self.feet_stabilizer.ground_contact_root_drop_m
                        * contact_weight[:, None]
                    )
                    conditioned[:, self.feet_effector_indices, 2] = targets[
                        :, self.feet_effector_indices, 2
                    ]
                    dropped_sequences.append(conditioned)
                target_sequences = dropped_sequences
        input_targets_gpu = wp.array3d(
            np.stack([
                np.pad(
                    targets,
                    ((0, self.max_frames - len(targets)), (0, 0), (0, 0)),
                    mode="edge",
                )
                for targets in target_sequences
            ]),
            dtype=wp.transform)
        if self.post_processing_enabled:
            assert prepared_feet is not None
            input_feet_targets_gpu = wp.array3d(
                np.stack(
                    [
                        np.pad(
                            targets,
                            ((0, self.max_frames - len(targets)), (0, 0), (0, 0)),
                            mode="edge",
                        )
                        for targets, _ in prepared_feet
                    ]
                ),
                dtype=wp.transform,
            )
            input_feet_rotation_weights_gpu = wp.array3d(
                np.stack(
                    [
                        np.pad(
                            weights,
                            ((0, self.max_frames - len(weights)), (0, 0)),
                            mode="edge",
                        )
                        for _, weights in prepared_feet
                    ]
                ),
                dtype=wp.float32,
            )
        if static_com_objective is not None:
            input_static_com_weights_gpu = wp.array2d(
                np.stack([
                    np.pad(
                        weights,
                        (0, self.max_frames - len(weights)),
                        mode="edge",
                    )
                    for weights in self.input_static_com_weights
                ]),
                dtype=wp.float32,
            )
        for frame in trange(self.max_frames, desc="[INFO] Retargeting Motions", disable=not self.show_progress):
            if frame <= num_frames_to_remove:
                smooth_joint_filter_objective.set_weight(self.smooth_joint_filter_weight * (frame / float(num_frames_to_remove)))

            #start_time = time.time()
            for i in range(len(position_objectives)):
                wp.launch(
                    _copy_ik_targets_frame_kernel,
                    dim=num_envs,
                    inputs=[
                        input_targets_gpu,
                        frame,
                        last_target_frames,
                        i,
                    ],
                    outputs=[
                        position_objectives[i].target_positions,
                        rotation_objectives[i].target_rotations,
                    ])

            if static_com_objective is not None:
                wp.launch(
                    _copy_static_com_targets_frame_kernel,
                    dim=num_envs,
                    inputs=[
                        input_targets_gpu,
                        input_static_com_weights_gpu,
                        frame,
                        last_target_frames,
                        self.root_effector_index,
                        self.feet_effector_indices[0],
                        self.feet_effector_indices[1],
                        self.static_com_forward_offset,
                    ],
                    outputs=[
                        static_com_objective.target_positions,
                        static_com_objective.frame_weights,
                    ],
                )

            if graph_capture is not None:
                wp.capture_launch(graph_capture)
            else:
                single_step()

            if self.post_processing_enabled:
                self.feet_stabilizer.reset_state(joint_q)
                wp.launch(
                    _copy_feet_targets_frame_kernel,
                    dim=[num_envs, len(self.feet_effector_indices)],
                    inputs=[
                        input_feet_targets_gpu,
                        input_feet_rotation_weights_gpu,
                        frame,
                        last_target_frames,
                    ],
                    outputs=[feet_targets, feet_rotation_weights])
                if static_com_objective is not None:
                    self.feet_stabilizer.solve_warp(
                        feet_targets,
                        static_com_objective.target_positions,
                        static_com_objective.frame_weights,
                        feet_rotation_weights,
                    )
                else:
                    self.feet_stabilizer.solve_warp(
                        feet_targets,
                        foot_rotation_weights=feet_rotation_weights,
                    )
                output_joint_q = self.joint_limit_clamper.apply(self.feet_stabilizer.current_state())
            else:
                output_joint_q = self.joint_limit_clamper.apply(joint_q)

            output_frame = frame - num_frames_to_remove
            if output_frame >= 0:
                wp.launch(
                    _copy_joint_q_frame_kernel,
                    dim=[num_envs, self.ik_model.joint_coord_count],
                    inputs=[output_joint_q, output_frame],
                    outputs=[joint_q_frames])

            #end_time = time.time()
            #print(f"Time taken for frame {frame}: {end_time - start_time} seconds")

        joint_q_data = joint_q_frames.numpy()
        return [
            CSVAnimationBuffer.create_from_raw_data(joint_q_data[i, :output_frame_counts[i]], self.input_sample_rates[i])
            for i in range(num_envs)]

    def _build_model(self, num_envs: int):
        builder = newton.ModelBuilder()
        for _ in range(num_envs):
            builder.add_builder(self.robot_builder, xform=wp.transform_identity())

        builder.add_ground_plane()
        model = builder.finalize(requires_grad=True)

        return model

    def _build_target_mapping(self, model, skeleton, retargeter_config):
        mapped_joints = []
        mapped_joint_indices = []
        mapped_body_link_pos_data = []
        mapped_body_link_rot_data = []
        body_names = [newton_utils.get_name_from_label(label) for label in self.robot_builder.body_label]
        for joint, mapping_data in retargeter_config["ik_map"].items():
            mapped_joints.append(joint)
            mapped_joint_indices.append(skeleton.joint_index(joint))
            mapped_body_link_pos_data.append((body_names.index(mapping_data['t_body']), mapping_data['t_weight']))
            mapped_body_link_rot_data.append((body_names.index(mapping_data['r_body']), mapping_data['r_weight']))

        return (
            mapped_joints,
            mapped_joint_indices,
            mapped_body_link_pos_data,
            mapped_body_link_rot_data)

    def _create_ik_objectives(self, num_envs, model, state):
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)

        # Gather default body position and rotation based on model state to initialize
        # position and rotation objectives
        num_body_link_pos = len(self.mapped_body_link_pos_data)
        num_body_link_rot = len(self.mapped_body_link_rot_data)
        pos_targets = np.zeros((num_envs, num_body_link_pos), dtype=wp.vec3)
        rot_targets = np.zeros((num_envs, num_body_link_rot), dtype=wp.quat)

        body_q = state.body_q.numpy()
        for env in range(num_envs):
            base = env * self.num_body_count
            for ee_idx, (link_idx, _) in enumerate(self.mapped_body_link_pos_data):
                pos_targets[env, ee_idx] = body_q[base + link_idx][0:3]

            for ee_idx, (link_idx, _) in enumerate(self.mapped_body_link_rot_data):
                rot_wp = wp.quat(body_q[base + link_idx][3:7])
                rot_targets[env, ee_idx] = wp.normalize(rot_wp)

        pos_num_ees = len(self.mapped_body_link_pos_data)
        rot_num_ees = len(self.mapped_body_link_rot_data)
        pos_target_arrays, rot_target_arrays = [], []
        for ee_idx in range(pos_num_ees):
            pos_wp = wp.array(pos_targets[:, ee_idx], dtype=wp.vec3)
            pos_target_arrays.append(pos_wp)

        for ee_idx in range(rot_num_ees):
            rot_wp = wp.array(rot_targets[:, ee_idx], dtype=wp.vec4)
            rot_target_arrays.append(rot_wp)

        position_objectives = []
        for i, (link_idx, w) in enumerate(self.mapped_body_link_pos_data):
            objective = ik.IKObjectivePosition(
                link_index=link_idx,
                link_offset=wp.vec3(0.0, 0.0, 0.0),
                target_positions=pos_target_arrays[i],
                weight=w)
            position_objectives.append(objective)

        rotation_objectives = []
        for i, (link_idx, w) in enumerate(self.mapped_body_link_rot_data):
            objective = ik.IKObjectiveRotation(
                link_index=link_idx,
                link_offset_rotation=wp.quat_identity(),
                target_rotations=rot_target_arrays[i],
                weight=w)
            rotation_objectives.append(objective)

        joint_limit_objective = ik.IKObjectiveJointLimit(
            joint_limit_lower=self.ik_model.joint_limit_lower,
            joint_limit_upper=self.ik_model.joint_limit_upper,
            weight=self.joint_limit_weight)

        # Weight is set to desired value once initialization frames have been processed
        smooth_joint_limiter_objective = IKSmoothJointFilter(
            joint_limit_lower=self.ik_model.joint_limit_lower,
            joint_limit_upper=self.ik_model.joint_limit_upper,
            weight=0.0,
            coord_masks=self.smooth_joint_filter_coord_masks)

        static_com_objective = None
        if self.static_com_enabled:
            static_com_objective = IKCenterOfMassHorizontal(
                target_positions=wp.zeros(num_envs, dtype=wp.vec3),
                frame_weights=wp.zeros(num_envs, dtype=wp.float32),
                weight=self.static_com_weight,
            )

        return (
            position_objectives,
            rotation_objectives,
            joint_limit_objective,
            smooth_joint_limiter_objective,
            static_com_objective,
        )
