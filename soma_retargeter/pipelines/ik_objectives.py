# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import warp as wp

import newton.ik as ik
from newton._src.sim.ik.ik_common import IKJacobianType


@wp.kernel
def _minimum_joint_angle_residuals(
    joint_q: wp.array2d(dtype=wp.float32),
    dof_to_coord: wp.array1d(dtype=wp.int32),
    target_angles: wp.array1d(dtype=wp.float32),
    coord_masks: wp.array1d(dtype=wp.float32),
    transition_angle: wp.float32,
    weight: wp.float32,
    start_idx: wp.int32,
    residuals: wp.array2d(dtype=wp.float32),
):
    problem, dof = wp.tid()
    coord = dof_to_coord[dof]
    if coord < 0:
        return
    mask = coord_masks[coord]
    delta = joint_q[problem, coord] - target_angles[coord]
    error = wp.min(delta, 0.0)
    if transition_angle > 0.0:
        if delta <= -transition_angle:
            error = delta
        elif delta < transition_angle:
            distance = transition_angle - delta
            error = -(distance * distance) / (4.0 * transition_angle)
        else:
            error = 0.0
    residuals[problem, start_idx + dof] = weight * mask * error


@wp.kernel
def _minimum_joint_angle_jacobian(
    joint_q: wp.array2d(dtype=wp.float32),
    dof_to_coord: wp.array1d(dtype=wp.int32),
    target_angles: wp.array1d(dtype=wp.float32),
    coord_masks: wp.array1d(dtype=wp.float32),
    transition_angle: wp.float32,
    weight: wp.float32,
    start_idx: wp.int32,
    jacobian: wp.array3d(dtype=wp.float32),
):
    problem, dof = wp.tid()
    coord = dof_to_coord[dof]
    if coord < 0:
        return
    delta = joint_q[problem, coord] - target_angles[coord]
    derivative = wp.float32(0.0)
    if transition_angle > 0.0:
        if delta <= -transition_angle:
            derivative = 1.0
        elif delta < transition_angle:
            derivative = (transition_angle - delta) / (2.0 * transition_angle)
    elif delta < 0.0:
        derivative = 1.0
    jacobian[problem, start_idx + dof, dof] = (
        weight * coord_masks[coord] * derivative
    )


class IKMinimumJointAngle(ik.IKObjective):
    """Apply a one-sided preference only while selected joints are too straight."""

    def __init__(
        self,
        target_angles,
        coord_masks,
        n_dofs,
        weight=1.0,
        transition_angle=0.0,
    ):
        super().__init__()
        self.target_angles_np = np.asarray(target_angles, dtype=np.float32)
        self.coord_masks_np = np.asarray(coord_masks, dtype=np.float32)
        self.weight = float(weight)
        self.transition_angle = max(0.0, float(transition_angle))
        self.n_dofs = int(n_dofs)
        self.dof_to_coord = None
        self.target_angles = None
        self.coord_masks = None

    def residual_dim(self):
        return self.n_dofs

    def supports_analytic(self):
        return True

    def init_buffers(self, model, jacobian_mode):
        self._require_batch_layout()
        if jacobian_mode != IKJacobianType.ANALYTIC:
            raise ValueError("IKMinimumJointAngle requires the analytic Jacobian mode")
        if self.target_angles_np.shape != (model.joint_coord_count,):
            raise ValueError("target_angles must match model.joint_coord_count")
        if self.coord_masks_np.shape != (model.joint_coord_count,):
            raise ValueError("coord_masks must match model.joint_coord_count")

        if self.n_dofs != model.joint_dof_count:
            raise ValueError("n_dofs must match model.joint_dof_count")
        dof_to_coord = np.full(self.n_dofs, -1, dtype=np.int32)
        q_start = model.joint_q_start.numpy()
        qd_start = model.joint_qd_start.numpy()
        dof_dims = model.joint_dof_dim.numpy()
        for joint in range(model.joint_count):
            linear, angular = dof_dims[joint]
            for offset in range(linear + angular):
                dof_to_coord[qd_start[joint] + offset] = q_start[joint] + offset
        self.dof_to_coord = wp.array(dof_to_coord, dtype=wp.int32, device=self.device)
        self.target_angles = wp.array(
            self.target_angles_np, dtype=wp.float32, device=self.device
        )
        self.coord_masks = wp.array(
            self.coord_masks_np, dtype=wp.float32, device=self.device
        )

    def compute_residuals(
        self, body_q, joint_q, model, residuals, start_idx, problem_idx
    ):
        wp.launch(
            _minimum_joint_angle_residuals,
            dim=[joint_q.shape[0], self.n_dofs],
            inputs=[
                joint_q,
                self.dof_to_coord,
                self.target_angles,
                self.coord_masks,
                self.transition_angle,
                self.weight,
                start_idx,
            ],
            outputs=[residuals],
            device=self.device,
        )

    def compute_jacobian_analytic(
        self, body_q, joint_q, model, jacobian, joint_S_s, start_idx
    ):
        wp.launch(
            _minimum_joint_angle_jacobian,
            dim=[joint_q.shape[0], self.n_dofs],
            inputs=[
                joint_q,
                self.dof_to_coord,
                self.target_angles,
                self.coord_masks,
                self.transition_angle,
                self.weight,
                start_idx,
            ],
            outputs=[jacobian],
            device=self.device,
        )


@wp.kernel
def _center_of_mass_horizontal_residuals(
    body_q: wp.array2d(dtype=wp.transform),
    body_mass: wp.array1d(dtype=wp.float32),
    body_com: wp.array1d(dtype=wp.vec3),
    total_mass: wp.float32,
    target_positions: wp.array1d(dtype=wp.vec3),
    frame_weights: wp.array1d(dtype=wp.float32),
    objective_weight: wp.float32,
    left_support_body: wp.int32,
    right_support_body: wp.int32,
    start_idx: wp.int32,
    problem_idx_map: wp.array1d(dtype=wp.int32),
    residuals: wp.array2d(dtype=wp.float32),
):
    row = wp.tid()
    base = problem_idx_map[row]
    weighted_position = wp.vec3(0.0, 0.0, 0.0)
    for body in range(body_q.shape[1]):
        weighted_position += (
            wp.transform_point(body_q[row, body], body_com[body]) * body_mass[body]
        )
    center_of_mass = weighted_position / total_mass
    target = target_positions[base]
    if left_support_body >= 0 and right_support_body >= 0:
        target += 0.5 * (
            wp.transform_get_translation(body_q[row, left_support_body])
            + wp.transform_get_translation(body_q[row, right_support_body])
        )
    error = target - center_of_mass
    weight = objective_weight * frame_weights[base]
    residuals[row, start_idx] = weight * error[0]
    residuals[row, start_idx + 1] = weight * error[1]


@wp.kernel
def _center_of_mass_horizontal_jacobian(
    body_q: wp.array2d(dtype=wp.transform),
    joint_S_s: wp.array2d(dtype=wp.spatial_vector),
    body_mass: wp.array1d(dtype=wp.float32),
    body_com: wp.array1d(dtype=wp.vec3),
    affects_body: wp.array2d(dtype=wp.uint8),
    total_mass: wp.float32,
    frame_weights: wp.array1d(dtype=wp.float32),
    objective_weight: wp.float32,
    left_support_body: wp.int32,
    right_support_body: wp.int32,
    start_idx: wp.int32,
    jacobian: wp.array3d(dtype=wp.float32),
):
    problem, dof = wp.tid()
    linear = wp.vec3(0.0, 0.0, 0.0)
    motion = joint_S_s[problem, dof]
    origin_velocity = wp.vec3(motion[0], motion[1], motion[2])
    angular_velocity = wp.vec3(motion[3], motion[4], motion[5])
    for body in range(body_q.shape[1]):
        if affects_body[dof, body] != wp.uint8(0):
            point = wp.transform_point(body_q[problem, body], body_com[body])
            point_velocity = origin_velocity + wp.cross(angular_velocity, point)
            linear += point_velocity * body_mass[body]
    linear /= total_mass
    support_velocity = wp.vec3(0.0, 0.0, 0.0)
    if left_support_body >= 0 and right_support_body >= 0:
        if affects_body[dof, left_support_body] != wp.uint8(0):
            left_position = wp.transform_get_translation(
                body_q[problem, left_support_body]
            )
            support_velocity += 0.5 * (
                origin_velocity + wp.cross(angular_velocity, left_position)
            )
        if affects_body[dof, right_support_body] != wp.uint8(0):
            right_position = wp.transform_get_translation(
                body_q[problem, right_support_body]
            )
            support_velocity += 0.5 * (
                origin_velocity + wp.cross(angular_velocity, right_position)
            )
    weight = objective_weight * frame_weights[problem]
    relative_velocity = support_velocity - linear
    jacobian[problem, start_idx, dof] = weight * relative_velocity[0]
    jacobian[problem, start_idx + 1, dof] = weight * relative_velocity[1]


@wp.kernel
def _update_center_of_mass_targets(
    positions: wp.array1d(dtype=wp.vec3),
    weights: wp.array1d(dtype=wp.float32),
    target_positions: wp.array1d(dtype=wp.vec3),
    frame_weights: wp.array1d(dtype=wp.float32),
):
    problem = wp.tid()
    target_positions[problem] = positions[problem]
    frame_weights[problem] = weights[problem]


class IKCenterOfMassHorizontal(ik.IKObjective):
    """Keep the model COM horizontally over a per-problem support target."""

    def __init__(
        self,
        target_positions,
        frame_weights,
        weight=1.0,
        support_body_indices: tuple[int, int] | None = None,
    ):
        super().__init__()
        self.target_positions = target_positions
        self.frame_weights = frame_weights
        self.weight = float(weight)
        self.body_mass = None
        self.body_com = None
        self.affects_body = None
        self.total_mass = 0.0
        if support_body_indices is None:
            self.support_body_indices = (-1, -1)
        else:
            if len(support_body_indices) != 2:
                raise ValueError("support_body_indices must contain exactly two bodies")
            self.support_body_indices = tuple(int(index) for index in support_body_indices)

    def residual_dim(self):
        return 2

    def supports_analytic(self):
        return True

    def init_buffers(self, model, jacobian_mode):
        self._require_batch_layout()
        if jacobian_mode != IKJacobianType.ANALYTIC:
            raise ValueError("IKCenterOfMassHorizontal requires the analytic Jacobian mode")

        body_mass_np = model.body_mass.numpy().astype(np.float32)
        total_mass = float(np.sum(body_mass_np))
        if not np.isfinite(total_mass) or total_mass <= 0.0:
            raise ValueError("center-of-mass objective requires positive finite body mass")
        self.total_mass = total_mass
        self.body_mass = wp.array(body_mass_np, dtype=wp.float32, device=self.device)
        self.body_com = wp.array(model.body_com.numpy(), dtype=wp.vec3, device=self.device)

        qd_start = model.joint_qd_start.numpy()
        dof_to_joint = np.empty(model.joint_dof_count, dtype=np.int32)
        for joint in range(model.joint_count):
            dof_to_joint[qd_start[joint] : qd_start[joint + 1]] = joint

        joint_child = model.joint_child.numpy()
        joint_parent = model.joint_parent.numpy()
        body_to_joint = np.full(model.body_count, -1, dtype=np.int32)
        for joint, child in enumerate(joint_child):
            if child >= 0:
                body_to_joint[child] = joint

        affected = np.zeros((model.joint_dof_count, model.body_count), dtype=np.uint8)
        for body in range(model.body_count):
            ancestor_joints = set()
            cursor = body
            while cursor >= 0:
                joint = int(body_to_joint[cursor])
                if joint < 0:
                    break
                ancestor_joints.add(joint)
                cursor = int(joint_parent[joint])
            for dof, joint in enumerate(dof_to_joint):
                affected[dof, body] = joint in ancestor_joints
        self.affects_body = wp.array2d(affected, dtype=wp.uint8, device=self.device)

    def set_targets(self, positions, weights):
        wp.launch(
            _update_center_of_mass_targets,
            dim=self.target_positions.shape[0],
            inputs=[positions, weights],
            outputs=[self.target_positions, self.frame_weights],
            device=self.device,
        )

    def compute_residuals(self, body_q, joint_q, model, residuals, start_idx, problem_idx):
        wp.launch(
            _center_of_mass_horizontal_residuals,
            dim=body_q.shape[0],
            inputs=[
                body_q,
                self.body_mass,
                self.body_com,
                self.total_mass,
                self.target_positions,
                self.frame_weights,
                self.weight,
                self.support_body_indices[0],
                self.support_body_indices[1],
                start_idx,
                problem_idx,
            ],
            outputs=[residuals],
            device=self.device,
        )

    def compute_jacobian_analytic(self, body_q, joint_q, model, jacobian, joint_S_s, start_idx):
        wp.launch(
            _center_of_mass_horizontal_jacobian,
            dim=[body_q.shape[0], model.joint_dof_count],
            inputs=[
                body_q,
                joint_S_s,
                self.body_mass,
                self.body_com,
                self.affects_body,
                self.total_mass,
                self.frame_weights,
                self.weight,
                self.support_body_indices[0],
                self.support_body_indices[1],
                start_idx,
            ],
            outputs=[jacobian],
            device=self.device,
        )


@wp.func
def _wp_smooth_joint_filter_func(
    x            : wp.float32,
    lower_limit  : wp.float32,
    upper_limit  : wp.float32,
    padding_limit: wp.float32,
    m            : wp.float32,
    p            : wp.float32
):
    c = (lower_limit + upper_limit) * 0.5
    lower_limit += (padding_limit - c)
    upper_limit -= (padding_limit + c)
    if lower_limit < x and x <= upper_limit:
        return 0.0

    diff = wp.where(x <= lower_limit, lower_limit-x, x-upper_limit) * m
    return 1.0 - wp.exp(-wp.pow(diff, p))


@wp.kernel
def _smooth_joint_filter_residuals(
    joint_q: wp.array2d(dtype=wp.float32),           # (n_batch, n_coords)
    dof_to_coord: wp.array1d(dtype=wp.int32),        # (n_dofs)
    joint_limit_lower: wp.array1d(dtype=wp.float32), # (n_dofs)
    joint_limit_upper: wp.array1d(dtype=wp.float32), # (n_dofs)
    coord_masks: wp.array1d(dtype=wp.float32),       # (n_coords)
    weight: wp.array1d(dtype=wp.float32),            # (1)
    start_idx: int,
    # outputs
    residuals: wp.array2d(dtype=wp.float32),     # (n_batch, n_residuals)
):
    problem, dof_idx = wp.tid()
    coord_idx = dof_to_coord[dof_idx]
    mask = coord_masks[coord_idx]

    if coord_idx < 0:
        return

    if mask > 0.0:
        lower = joint_limit_lower[dof_idx]
        upper = joint_limit_upper[dof_idx]
        c = (lower + upper) * 0.5

        q = joint_q[problem, coord_idx]
        error = (q - c)

        smoother = _wp_smooth_joint_filter_func(error, lower, upper, 1.02, 1.0, 6.5)
        residuals[problem, start_idx + dof_idx] = error * smoother * weight[0] * mask
    else:
        residuals[problem, start_idx + dof_idx] = 0.0


@wp.kernel
def _update_weight(
    in_value: wp.float32,
    out_weight: wp.array1d(dtype=wp.float32),  # (1)
):
    out_weight[0] = in_value


@wp.kernel
def _smooth_joint_filter_jac_analytic(
    dof_to_coord: wp.array1d(dtype=wp.int32),    # (n_dofs)
    coord_masks: wp.array1d(dtype=wp.float32),   # (n_coords)
    n_dofs: int,
    start_idx: int,
    weight: wp.array1d(dtype=wp.float32), # (1)
    # outputs
    jacobian: wp.array3d(dtype=wp.float32),      # (n_batch, n_residuals, n_dofs)
):
    problem, dof_idx = wp.tid()
    coord_idx = dof_to_coord[dof_idx]
    mask = coord_masks[coord_idx]

    if coord_idx < 0:
        return

    # Jacobian is diagonal: dr[dof]/dq[dof] = weight
    jacobian[problem, start_idx + dof_idx, dof_idx] = weight[0] * mask


class IKSmoothJointFilter(ik.IKObjective):
    """
    An IK objective that applies a smooth penalty to joint coordinates that approach or exceed specified limits
    using an inverse gaussian filter.

    Args:
        joint_limit_lower (wp.array1d): An array of shape (n_dofs,) containing the lower limits for each joint degree of freedom.
        joint_limit_upper (wp.array1d): An array of shape (n_dofs,) containing the upper limits for each joint degree of freedom.
        weight (float, optional): A scalar weight that controls the strength of the joint limit penalty. Defaults to 0.01.
        coord_masks (wp.array1d, optional): An array of shape (n_coords,) containing mask values for each joint coordinate.
            Mask values should be in the range [0, 1], where 0 means the coordinate is ignored by this objective and 1 means it is fully considered.
            All coords are used by default if no masks are specified.
    """
    def __init__(self, joint_limit_lower, joint_limit_upper, weight=0.01, coord_masks=None):
        super().__init__()
        self.joint_limit_lower = joint_limit_lower
        self.joint_limit_upper = joint_limit_upper
        self.n_dofs = len(joint_limit_lower)
        self.dof_to_coord = None
        self.e_array = None
        self._weight = wp.array([weight], dtype=wp.float32)

        self.coord_masks = None
        self.coord_masks_np = None
        if coord_masks is not None:
            if isinstance(coord_masks, np.ndarray):
                self.coord_masks_np = coord_masks.astype(np.float32)
                self.coord_masks = None
            elif isinstance(coord_masks, wp.array):
                self.coord_masks = coord_masks
                self.coord_masks_np = None

    def bind_device(self, device):
        super().bind_device(device)

    def init_buffers(self, model, jacobian_mode):
        self._require_batch_layout()

        if self.coord_masks_np is not None and len(self.coord_masks_np) == model.joint_coord_count:
            self.coord_masks = wp.array(self.coord_masks_np, dtype=wp.float32, device=self.device)

        # All coords are considered if no coord masks have been declared
        if self.coord_masks is None:
            self.coord_masks = wp.ones(shape=model.joint_coord_count, dtype=wp.float32, device=self.device)

        # Build DOF to coordinate mapping
        dof_to_coord_np = np.full(self.n_dofs, -1, dtype=np.int32)
        q_start_np = model.joint_q_start.numpy()
        qd_start_np = model.joint_qd_start.numpy()
        joint_dof_dim_np = model.joint_dof_dim.numpy()

        for j in range(model.joint_count):
            dof0 = qd_start_np[j]
            coord0 = q_start_np[j]
            lin, ang = joint_dof_dim_np[j]
            for k in range(lin + ang):
                if dof0 + k < self.n_dofs:
                    dof_to_coord_np[dof0 + k] = coord0 + k

        self.dof_to_coord = wp.array(dof_to_coord_np, dtype=wp.int32, device=self.device)

        # For autodiff mode
        if jacobian_mode == IKJacobianType.AUTODIFF:
            e = np.zeros((self.n_batch, self.total_residuals), dtype=np.float32)
            for prob_idx in range(self.n_batch):
                for dof_idx in range(self.n_dofs):
                    e[prob_idx, self.residual_offset + dof_idx] = 1.0
            self.e_array = wp.array(e.flatten(), dtype=wp.float32, device=self.device)

    def supports_analytic(self):
        return True

    def residual_dim(self):
        return self.n_dofs

    def set_weight(self, value):
        if self.coord_masks is None:
            return

        wp.launch(
            _update_weight,
            dim=1,
            inputs=[value],
            outputs=[self._weight],
            device=self.device)

    def compute_residuals(self, body_q, joint_q, model, residuals, start_idx, problem_idx):
        count = joint_q.shape[0]
        wp.launch(
            _smooth_joint_filter_residuals,
            dim=[count, self.n_dofs],
            inputs=[
                joint_q,
                self.dof_to_coord,
                self.joint_limit_lower,
                self.joint_limit_upper,
                self.coord_masks,
                self._weight,
                start_idx,
            ],
            outputs=[residuals],
            device=self.device,
        )

    def compute_jacobian_autodiff(self, tape, model, jacobian, start_idx, dq_dof):
        self._require_batch_layout()
        tape.backward(grads={tape.outputs[0]: self.e_array})

        q_grad = tape.gradients[dq_dof]

        # Use the analytic Jacobian fill since it's simple
        wp.launch(
            _smooth_joint_filter_jac_analytic,
            dim=[self.n_batch, self.n_dofs],
            inputs=[
                self.dof_to_coord,
                self.coord_masks,
                self.n_dofs,
                start_idx,
                self._weight,
            ],
            outputs=[jacobian],
            device=self.device,
        )

    def compute_jacobian_analytic(self, body_q, joint_q, model, jacobian, joint_S_s, start_idx):
        count = joint_q.shape[0]
        wp.launch(
            _smooth_joint_filter_jac_analytic,
            dim=[count, self.n_dofs],
            inputs=[
                self.dof_to_coord,
                self.coord_masks,
                self.n_dofs,
                start_idx,
                self._weight,
            ],
            outputs=[jacobian],
            device=self.device,
        )
