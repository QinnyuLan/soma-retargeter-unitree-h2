import unittest
from pathlib import Path

import numpy as np
import warp as wp

from soma_retargeter.pipelines.feet_stabilizer import FeetStabilizer


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = REPO_ROOT / "soma_retargeter/configs/unitree_h2/h2_feet_stabilizer_config.json"
GROUND_ROTATION_CONFIG = (
    REPO_ROOT
    / "soma_retargeter/configs/unitree_h2/h2_feet_stabilizer_low_rotation_repair.json"
)


class FeetStabilizerTest(unittest.TestCase):
    def test_ground_contact_targets_are_flattened_with_smooth_release(self):
        stabilizer = FeetStabilizer(str(GROUND_ROTATION_CONFIG))
        targets = np.zeros((41, 2, 7), dtype=np.float32)
        targets[:, :, 6] = 1.0
        targets[:, :, 2] = np.linspace(0.0, 0.12, len(targets))[:, None]
        roll = np.deg2rad(30.0)
        targets[:, :, 3] = np.sin(0.5 * roll)
        targets[:, :, 6] = np.cos(0.5 * roll)

        prepared, weights = stabilizer.prepare_foot_targets(targets)

        np.testing.assert_allclose(prepared[0, :, 3:5], 0.0, atol=1e-4)
        np.testing.assert_allclose(
            prepared[-1, :, 3], targets[-1, :, 3], atol=1e-4
        )
        self.assertGreater(float(prepared[20, 0, 3]), 0.0)
        self.assertLess(float(prepared[20, 0, 3]), float(targets[20, 0, 3]))
        np.testing.assert_allclose(prepared[0, :, 2], 0.001, atol=1e-7)
        self.assertGreater(float(weights[0, 0]), 0.99)
        self.assertGreater(float(weights[20, 0]), 0.0)
        self.assertLess(float(weights[20, 0]), 1.0)
        self.assertLess(float(weights[-1, 0]), 1.0e-4)
        self.assertLess(float(np.max(np.abs(np.diff(weights[:, 0])))), 0.12)

    def test_reset_state_evaluates_fk_from_current_batched_joint_q(self):
        with wp.ScopedDevice("cpu"):
            stabilizer = FeetStabilizer(str(CONFIG))
            stabilizer.setup_num_envs(2)

            expected_root_positions = np.asarray(
                [[1.25, -0.5, 0.8], [-2.0, 0.75, 1.1]], dtype=np.float32
            )
            joint_q = stabilizer.joint_q.numpy()
            joint_q[:, :3] = expected_root_positions
            stabilizer.reset_state(wp.array(joint_q, dtype=wp.float32, device="cpu"))
            wp.synchronize()

            body_q = stabilizer.state.body_q.numpy().reshape(
                stabilizer.num_envs, stabilizer.num_body_count, 7
            )
            np.testing.assert_allclose(
                body_q[:, stabilizer.pelvis_idx, :3],
                expected_root_positions,
                rtol=0.0,
                atol=1e-6,
            )

    def test_moving_near_ground_foot_gets_minimum_swing_clearance(self):
        stabilizer = FeetStabilizer(str(GROUND_ROTATION_CONFIG))
        targets = np.zeros((5, 2, 7), dtype=np.float32)
        targets[:, :, 2] = 0.02
        targets[:, :, 6] = 1.0
        targets[:, 0, 0] = np.arange(5, dtype=np.float32) * 0.01

        prepared, weights = stabilizer.prepare_foot_targets(
            targets, sample_rate=120.0
        )

        np.testing.assert_allclose(weights[:, 0], 0.0, atol=1e-7)
        np.testing.assert_allclose(prepared[:, 0, 2], 0.091, atol=1e-7)
        self.assertTrue(np.all(weights[:, 1] > 0.99))
        np.testing.assert_allclose(prepared[:, 1, 2], 0.001, atol=1e-7)

    def test_edge_contact_preserves_rotation_and_clears_support_polygon(self):
        stabilizer = FeetStabilizer(str(CONFIG))
        targets = np.zeros((5, 2, 7), dtype=np.float32)
        targets[:, :, 2] = 0.001
        half_pitch = np.deg2rad(45.0)
        targets[:, :, 4] = np.sin(half_pitch)
        targets[:, :, 6] = np.cos(half_pitch)

        prepared, weights = stabilizer.prepare_foot_targets(targets)

        self.assertTrue(np.all(weights > 0.99))
        np.testing.assert_allclose(prepared[:, :, 3:7], targets[:, :, 3:7])
        expected_origin_height = 0.001 + 0.1441
        np.testing.assert_allclose(
            prepared[:, :, 2], expected_origin_height, atol=1.0e-6
        )

    def test_bilateral_deep_knee_flexion_lifts_pelvis_target(self):
        with wp.ScopedDevice("cpu"):
            stabilizer = FeetStabilizer(str(GROUND_ROTATION_CONFIG))
            stabilizer.setup_num_envs(1)
            joint_q = stabilizer.joint_q.numpy()
            joint_q[0, list(stabilizer.knee_coord_indices)] = np.deg2rad(150.0)
            deep_pose = joint_q.copy()
            stabilizer.reset_state(wp.array(deep_pose, dtype=wp.float32, device="cpu"))
            body_q = stabilizer.state.body_q.numpy().reshape(
                stabilizer.num_envs, stabilizer.num_body_count, 7
            )
            initial_pelvis_z = float(body_q[0, stabilizer.pelvis_idx, 2])
            foot_indices = [
                stabilizer.effector_mapped_indices[limb[1][-1]]
                for limb in stabilizer.ik_limb_data
            ]
            foot_targets = wp.array2d(
                body_q[:, foot_indices], dtype=wp.transform, device="cpu"
            )

            stabilizer.solve_warp(foot_targets)
            first_pelvis_target_z = float(stabilizer.out_effectors.numpy()[0, 0, 2])

            for _ in range(80):
                stabilizer.reset_state(
                    wp.array(deep_pose, dtype=wp.float32, device="cpu")
                )
                stabilizer.solve_warp(foot_targets)
            pelvis_target_z = float(stabilizer.out_effectors.numpy()[0, 0, 2])

            self.assertGreater(first_pelvis_target_z, initial_pelvis_z)
            self.assertLess(
                first_pelvis_target_z - initial_pelvis_z,
                stabilizer.deep_knee_root_lift_max_m,
            )

            self.assertAlmostEqual(
                pelvis_target_z - initial_pelvis_z,
                stabilizer.deep_knee_root_lift_max_m,
                places=5,
            )


if __name__ == "__main__":
    unittest.main()
