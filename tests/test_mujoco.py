import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

import mujoco
import numpy as np
import torch

from humanoidverse.simulator.mujoco.mujoco import MuJoCo

XML = """<mujoco><option gravity="0 0 0"/><worldbody>
<body name="torso_link"><freejoint/>
<geom type="sphere" size="0.1" mass="1" pos="0.1 -0.2 0.3"/>
<body name="child" pos="1 0 0"><joint name="hinge" type="hinge"/>
<geom type="sphere" size="0.1" mass="1" pos="0.2 0.1 0"/>
</body></body></worldbody></mujoco>"""


def simulator(xml=XML):
    config = NS(
        simulator=NS(config=NS(sim=NS(substeps=1, fps=100))),
        robot=NS(asset=NS(asset_root="", xml_file="inline.xml"), dof_names=["hinge"], body_names=["torso_link", "child"]),
        domain_rand={},
        save_rendering_dir=None,
    )
    sim = MuJoCo(config, "cpu")
    with patch.object(mujoco.MjModel, "from_xml_path", return_value=mujoco.MjModel.from_xml_string(xml)):
        sim.setup()
    sim.load_assets()
    mujoco.mj_forward(sim.model, sim.data)
    return sim


class MuJoCoTests(unittest.TestCase):
    def test_root_reset_uses_target_orientation_and_preserves_input(self):
        for as_numpy in (False, True):
            with self.subTest(as_numpy=as_numpy):
                sim = simulator()
                root = torch.tensor([[3.0, 4.0, 5.0, 0.0, 0.0, 2**-0.5, 2**-0.5, 0.1, 0.2, 0.3, 1.0, 2.0, 3.0]])
                supplied = root.numpy().copy() if as_numpy else root.clone()
                sim.set_actor_root_state_tensor([0], supplied)
                np.testing.assert_array_equal(np.asarray(supplied), root.numpy())
                np.testing.assert_allclose(sim.data.qvel[3:6], [2.0, -1.0, 3.0], atol=1e-6)
                np.testing.assert_allclose(sim.data.xpos[1], [3.0, 4.0, 5.0])
                torch.testing.assert_close(sim.robot_root_states, root, atol=1e-6, rtol=1e-6)

    def test_joint_reset_refreshes_kinematics_and_empty_resets_do_nothing(self):
        sim = simulator()
        sim.set_dof_state_tensor([0], torch.tensor([[[0.4, 0.2]]]))
        self.assertAlmostEqual(sim.data.qpos[7], 0.4, places=6)
        np.testing.assert_allclose(sim.data.xmat[2].reshape(3, 3)[:, 0], [np.cos(0.4), np.sin(0.4), 0], atol=1e-6)
        np.testing.assert_allclose(sim._rigid_body_ang_vel[0, 1], [0, 0, 0.2], atol=1e-6)
        qpos, qvel = sim.data.qpos.copy(), sim.data.qvel.copy()
        sim.set_actor_root_state_tensor([], torch.zeros(1, 13))
        sim.set_dof_state_tensor([], torch.zeros(1, 1, 2))
        np.testing.assert_array_equal(sim.data.qpos, qpos)
        np.testing.assert_array_equal(sim.data.qvel, qvel)

    def test_body_velocity_matches_position_finite_difference(self):
        sim = simulator()
        sim.data.qpos[3:7] = [np.cos(0.35), 0, 0, np.sin(0.35)]
        sim.data.qpos[7] = 0.3
        sim.data.qvel[:] = [0.2, -0.3, 0.1, 0.4, 0.5, 0.8, -0.2]
        mujoco.mj_forward(sim.model, sim.data)
        velocity = sim._rigid_body_vel[0].numpy()
        pose = sim.data.qpos.copy()
        positions = []
        epsilon = 1e-6
        for delta in (-epsilon, epsilon):
            sim.data.qpos[:] = pose
            mujoco.mj_integratePos(sim.model, sim.data.qpos, sim.data.qvel, delta)
            mujoco.mj_forward(sim.model, sim.data)
            positions.append(sim.data.xpos[sim.body_id].copy())
        expected = (positions[1] - positions[0]) / (2 * epsilon)
        np.testing.assert_allclose(velocity, expected, atol=1e-6)

    def test_com_randomization_moves_com_without_changing_mass_or_live_state(self):
        sim = simulator()
        original_ipos = sim.model.body_ipos.copy()
        original_mass = sim.model.body_mass.copy()
        sim.domain_rand_config = {"randomize_base_com": True, "base_com_range": {"x": [0.01, 0.01], "y": [-0.02, -0.02], "z": [0.03, 0.03]}}
        sim.data.qpos[0] = 3.0
        sim.data.qpos[7] = 0.4
        sim.data.qvel[:] = np.arange(sim.model.nv) * 0.1
        sim.data.time = 2.5
        qpos, qvel = sim.data.qpos.copy(), sim.data.qvel.copy()
        for _ in range(2):
            sim.episodic_domain_randomization(None)
            np.testing.assert_allclose(sim.model.body_ipos[1] - original_ipos[1], [0.01, -0.02, 0.03])
            np.testing.assert_array_equal(sim.model.body_mass, original_mass)
            np.testing.assert_array_equal(sim.data.qpos, qpos)
            np.testing.assert_array_equal(sim.data.qvel, qvel)
            self.assertEqual(sim.data.time, 2.5)
            np.testing.assert_allclose(sim.data.xipos[1], sim.data.xpos[1] + sim.model.body_ipos[1])
        sim.domain_rand_config["randomize_base_com"] = False
        sim.episodic_domain_randomization(None)
        np.testing.assert_array_equal(sim.model.body_ipos, original_ipos)

    def test_refresh_after_substeps_updates_pose_without_advancing_time(self):
        sim = simulator()
        sim.data.qvel[0] = 1.0
        for _ in range(4):
            sim.simulate_at_each_physics_step()
        qpos, qvel = sim.data.qpos.copy(), sim.data.qvel.copy()
        sim.refresh_sim_tensors()
        self.assertAlmostEqual(sim.data.time, 0.04)
        np.testing.assert_array_equal(sim.data.qpos, qpos)
        np.testing.assert_array_equal(sim.data.qvel, qvel)
        np.testing.assert_allclose(sim._rigid_body_pos[0, 0], [0.04, 0, 0], atol=1e-6)

    def test_com_randomization_when_inertial_and_body_frames_start_aligned(self):
        sim = simulator(XML.replace('pos="0.1 -0.2 0.3"', 'pos="0 0 0"'))
        sim.data.qpos[3:7] = [2**-0.5, 2**-0.5, 0, 0]
        sim.refresh_sim_tensors()
        original_mass = sim.model.body_mass.copy()
        sim.domain_rand_config = {"randomize_base_com": True, "base_com_range": {"x": [0.01, 0.01], "y": [0.02, 0.02], "z": [0.03, 0.03]}}
        for _ in range(2):
            sim.episodic_domain_randomization(None)
            np.testing.assert_allclose(sim.data.xipos[1] - sim.data.xpos[1], [0.01, -0.03, 0.02], atol=1e-9)
            np.testing.assert_array_equal(sim.model.body_mass, original_mass)
        sim.domain_rand_config["randomize_base_com"] = False
        sim.episodic_domain_randomization(None)
        np.testing.assert_allclose(sim.data.xipos[1], sim.data.xpos[1], atol=1e-9)


if __name__ == "__main__":
    unittest.main()
