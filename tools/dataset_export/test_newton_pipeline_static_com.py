import unittest

import numpy as np

from soma_retargeter.pipelines.newton_pipeline import NewtonPipeline


class StaticCenterOfMassWeightTest(unittest.TestCase):
    def _pipeline(self, *, transition: float = 0.025, smoothing_s: float = 0.25):
        pipeline = NewtonPipeline.__new__(NewtonPipeline)
        pipeline.static_com_weight = 0.0
        pipeline.static_com_enabled = True
        pipeline.static_com_velocity_threshold = 0.05
        pipeline.static_com_velocity_transition = transition
        pipeline.static_com_smoothing_window_s = smoothing_s
        pipeline.static_com_stationary_hold_s = 0.25
        pipeline.static_com_moving_hold_s = 0.05
        pipeline.root_effector_index = 0
        pipeline.feet_effector_indices = [1, 2]
        return pipeline

    @staticmethod
    def _targets(root_x: np.ndarray) -> np.ndarray:
        targets = np.zeros((len(root_x), 3, 7), dtype=np.float32)
        targets[:, :, 6] = 1.0
        targets[:, :, 0] = root_x[:, None]
        return targets

    def test_stationary_motion_keeps_full_weight(self):
        weights = self._pipeline()._compute_static_com_weights(
            self._targets(np.zeros(120, dtype=np.float32)), 120.0
        )
        np.testing.assert_array_equal(weights, np.ones(120, dtype=np.float32))

    def test_motion_boundary_is_continuous_and_smoothed(self):
        root_x = np.zeros(180, dtype=np.float32)
        root_x[70:110] = np.arange(40, dtype=np.float32) * 0.001
        root_x[110:] = root_x[109]
        weights = self._pipeline()._compute_static_com_weights(
            self._targets(root_x), 120.0
        )

        self.assertLess(float(np.min(weights)), 0.05)
        self.assertGreater(float(weights[40]), 0.99)
        self.assertGreater(float(weights[160]), 0.99)
        self.assertLess(float(np.max(np.abs(np.diff(weights)))), 0.15)
        self.assertGreater(np.count_nonzero((weights > 0.0) & (weights < 1.0)), 20)

    def test_hysteresis_rejects_short_threshold_crossings(self):
        root_x = np.zeros(180, dtype=np.float32)
        root_x[60:63] = np.arange(3, dtype=np.float32) * 0.001
        root_x[63:] = root_x[62]
        weights = self._pipeline(smoothing_s=0.0)._compute_static_com_weights(
            self._targets(root_x), 120.0
        )
        np.testing.assert_array_equal(weights, np.ones(180, dtype=np.float32))

    def test_sustained_motion_requires_stationary_hold_before_reenable(self):
        root_x = np.zeros(240, dtype=np.float32)
        root_x[60:100] = np.arange(40, dtype=np.float32) * 0.001
        root_x[100:] = root_x[99]
        pipeline = self._pipeline(smoothing_s=0.0)
        weights = pipeline._compute_static_com_weights(self._targets(root_x), 120.0)
        self.assertEqual(weights[70], 0.0)
        self.assertEqual(weights[110], 0.0)
        self.assertEqual(weights[140], 1.0)


if __name__ == "__main__":
    unittest.main()
