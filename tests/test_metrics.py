import unittest

import torch

from humanoidverse.agents.evaluations import humanoidverse_isaac as evaluation


class MetricsTests(unittest.TestCase):
    def test_joint_derivatives_use_time_axis_and_seconds(self):
        time = torch.arange(4, dtype=torch.float64) * 0.25
        actual = time[:, None].square() * torch.tensor([1.0, 2.0])
        target = torch.zeros_like(actual)
        metrics = evaluation.compute_joint_pos_metrics(actual, target, dt=0.25)
        self.assertAlmostEqual(metrics["joint_angle_l2_rad"].item(), 0.21875 * 5**0.5)
        self.assertAlmostEqual(metrics["joint_velocity_l2_rad_s"].item(), 0.75 * 5**0.5)
        self.assertAlmostEqual(metrics["joint_acceleration_l2_rad_s2"].item(), 2 * 5**0.5)
        slower = evaluation.compute_joint_pos_metrics(actual, target, dt=0.5)
        self.assertAlmostEqual(slower["joint_velocity_l2_rad_s"].item(), metrics["joint_velocity_l2_rad_s"].item() / 2)
        self.assertAlmostEqual(slower["joint_acceleration_l2_rad_s2"].item(), metrics["joint_acceleration_l2_rad_s2"].item() / 4)
        constant = torch.tensor([[1.0, 3.0]], dtype=target.dtype).repeat(4, 1)
        metrics = evaluation.compute_joint_pos_metrics(constant, target, dt=0.25)
        self.assertEqual(metrics["joint_velocity_l2_rad_s"].item(), 0.0)
        self.assertEqual(metrics["joint_acceleration_l2_rad_s2"].item(), 0.0)
        self.assertTrue({"mpjpe_l", "vel_dist", "accel_dist"}.isdisjoint(metrics))

    def test_observation_emd_includes_all_joints_and_main_emd_is_unchanged(self):
        for joints in (23, 24, 29):
            with self.subTest(joints=joints):
                state = torch.zeros(4, joints * 2 + 6)
                state[:, joints - 1] = 1.0
                state[:, joints:] = 100.0  # Velocities/gravity must not enter pose EMD.
                actual = torch.zeros(4, joints)
                actual[:, -1] = 2.0
                metrics = evaluation._calc_metrics({
                    "observation": {"state": state},
                    "tracking_target": {"state": torch.zeros_like(state)},
                    "joint_pos": actual,
                    "target_joint_pos": torch.zeros_like(actual),
                    "dt": 0.02,
                    "motion_id": 0,
                    "motion_file": "test",
                })["test"]
                self.assertAlmostEqual(metrics["obs_state_emd"], 1.0)
                self.assertAlmostEqual(metrics["emd"], 2.0)

    def test_short_clips_and_invalid_metric_inputs(self):
        for frames in (1, 2):
            pose = torch.zeros(frames, 29)
            metrics = evaluation.compute_joint_pos_metrics(pose, pose, dt=0.02)
            self.assertEqual("joint_velocity_l2_rad_s" in metrics, frames >= 2)
            self.assertNotIn("joint_acceleration_l2_rad_s2", metrics)
            self.assertTrue(all(torch.isfinite(torch.as_tensor(value)).all() for value in metrics.values()))
        pose = torch.zeros(3, 29)
        for dt in (0.0, -0.02, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                evaluation.compute_joint_pos_metrics(pose, pose, dt=dt)
        for actual, target in ((pose[0], pose[0]), (pose, pose[:2]), (pose[:0], pose[:0])):
            with self.assertRaises(ValueError):
                evaluation.compute_joint_pos_metrics(actual, target, dt=0.02)


if __name__ == "__main__":
    unittest.main()
