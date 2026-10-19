import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import gymnasium as gym
import numpy as np
import torch
from omegaconf import OmegaConf

from humanoidverse.agents.envs import humanoidverse_isaac as hv
from humanoidverse.envs.legged_base_task.legged_robot_base import LeggedRobotBase
from humanoidverse.envs.legged_robot_motions.legged_robot_motions import LeggedRobotMotions
from humanoidverse.utils import helpers


class FakeEnv:
    """Simulator boundary only; exercise the real wrapper and reset assignment."""

    def __init__(self, num_envs=2, simulator="mujoco"):
        self.num_envs = num_envs
        self.device = "cpu"
        self.spec = None
        self.unwrapped = self
        self.config = OmegaConf.create({"simulator": {"config": {"name": simulator}}})
        self.default_dof_pos = torch.linspace(-0.1, 0.3, 29).unsqueeze(0)
        self.base_init_state = torch.tensor([0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.target_robot_dof_state = torch.zeros(num_envs, 29, 2)
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long)
        root = self.base_init_state.repeat(num_envs, 1)
        self.simulator = NS(robot_root_states=root, dof_pos=torch.zeros(num_envs, 29), dof_vel=torch.zeros(num_envs, 29))
        self.single_action_space = gym.spaces.Box(-1.0, 1.0, (29,), dtype=np.float32)
        self.obs_buf_dict_raw = {
            "actor_obs": {
                "dof_pos": self.simulator.dof_pos,
                "dof_vel": self.simulator.dof_vel,
                "base_ang_vel": torch.zeros(num_envs, 3),
                "projected_gravity": torch.zeros(num_envs, 3),
                "max_local_self": torch.zeros(num_envs, 12),
                "actions": torch.zeros(num_envs, 29),
            }
        }

    def get_wrapper_attr(self, key):
        return getattr(self, key)

    def _get_obs_base_ang_vel(self):
        return self.obs_buf_dict_raw["actor_obs"]["base_ang_vel"]

    def reset_all(self, target_states=None):
        if target_states is not None:
            LeggedRobotMotions._reset_dofs(self, torch.arange(self.num_envs), target_states["dof_states"])
        return None, {}

    def close(self):
        pass


def wrapper(env):
    return hv.HumanoidVerseVectorEnv(env, include_history_actor=False, included_dr_obs_names=[])


class InferenceTests(unittest.TestCase):
    def test_reset_observation_and_history_match_written_state(self):
        for manual_reset, episode_reset in ((True, False), (False, True), (False, False)):
            with self.subTest(manual_reset=manual_reset, episode_reset=episode_reset):
                root = torch.zeros(2, 13)
                root[:, 6] = 1.0
                target = root.clone()
                target[0, 3:7] = torch.tensor([2**-0.5, 0.0, 0.0, 2**-0.5])
                target[0, 10:13] = torch.tensor([0.0, 2.0, 0.0])
                history = []
                env = NS(
                    simulator=NS(
                        robot_root_states=root,
                        base_quat=root[:, 3:7],
                        set_actor_root_state_tensor=lambda ids, states: root.__setitem__(ids, states[ids]),
                        set_dof_state_tensor=Mock(),
                    ),
                    target_robot_root_states=target,
                    target_robot_dof_state=torch.zeros(2, 1, 2),
                    base_quat=root[:, 3:7].clone(),
                    rpy=torch.zeros(2, 3),
                    base_lin_vel=torch.zeros(2, 3),
                    base_ang_vel=torch.zeros(2, 3),
                    projected_gravity=torch.zeros(2, 3),
                    gravity_vec=torch.tensor([[0.0, 0.0, -1.0]]).repeat(2, 1),
                    episode_length_buf=torch.zeros(2, dtype=torch.long),
                    reset_buf=torch.tensor([episode_reset, False]),
                    need_to_refresh_envs=torch.tensor([manual_reset, False]),
                    config=NS(normalization=NS(clip_observations=100.0)),
                    history_handler=NS(history={"state": None}, add=lambda key, value: history.append(value.clone())),
                    extras={}, log_dict={}, viewer=None,
                )
                for name in ("_refresh_sim_tensors", "_update_counters_each_step", "_update_tasks_callback",
                             "_check_termination", "_compute_reward", "_post_compute_observations_callback"):
                    setattr(env, name, lambda: None)
                env.reset_envs_idx = lambda ids: env.need_to_refresh_envs.__setitem__(ids, True)
                env._pre_compute_observations_callback = lambda: LeggedRobotBase._pre_compute_observations_callback(env)

                def observe():
                    state = torch.cat([env.projected_gravity, env.base_ang_vel], dim=-1).clone()
                    env.obs_buf_dict = {"actor_obs": state}
                    env.hist_obs_dict = {"state": state}

                env._compute_observations = observe
                LeggedRobotBase._post_physics_step(env)
                expected = torch.tensor([[0.0, 0.0, -1.0, 0.0, 0.0, 0.0]]).repeat(2, 1)
                if manual_reset or episode_reset:
                    expected[0] = torch.tensor([0.0, -1.0, 0.0, 0.0, 0.0, -2.0])
                torch.testing.assert_close(env.obs_buf_dict["actor_obs"], expected, atol=1e-6, rtol=1e-6)
                torch.testing.assert_close(history[0], expected, atol=1e-6, rtol=1e-6)
                self.assertFalse(env.need_to_refresh_envs.any())

    def test_default_joint_positions_and_zero_velocities(self):
        for num_envs in (1, 3):
            env = FakeEnv(num_envs)
            wrapped = wrapper(env)
            wrapped.reset(to_numpy=False, reset_to_default_pose=True)
            self.assertEqual(tuple(env.target_robot_dof_state.shape), (num_envs, 29, 2))
            torch.testing.assert_close(env.target_robot_dof_state[..., 0], env.default_dof_pos.expand(num_envs, -1))
            self.assertEqual(env.target_robot_dof_state[..., 1].count_nonzero(), 0)

    def test_default_root_quaternion_matches_backend(self):
        for simulator in ("mujoco", "isaacsim"):
            for quaternion in (torch.tensor([0.0, 0.0, 0.0, 1.0]), torch.tensor([0.0, 0.0, 2.0**-0.5, 2.0**-0.5])):
                env = FakeEnv(simulator=simulator)
                env.base_init_state[3:7] = quaternion
                wrapped = wrapper(env)
                expected = quaternion[[3, 0, 1, 2]] if simulator == "isaacsim" else quaternion
                torch.testing.assert_close(wrapped._default_pose_target_reset["root_states"][3:7], expected)
                torch.testing.assert_close(env.base_init_state[3:7], quaternion)


    def test_disable_dr_switch_controls_actual_push_flag(self):
        seen = []

        class StopBeforeSimulation(Exception):
            pass

        def capture(config, device):
            seen.append(config.domain_rand.push_robots)
            raise StopBeforeSimulation

        with (
            patch.object(hv, "_humanoidverse_env_singleton", None),
            patch.object(hv, "instantiate_isaac_sim"),
            patch.object(hv, "LeggedRobotMotions", capture),
        ):
            for disabled in (False, True):
                from humanoidverse import train

                with patch.object(train.TrainConfig, "build", autospec=True, return_value=Mock()) as build:
                    train.train_bfm_zero()
                cfg = build.call_args.args[0].env.model_copy(update={"disable_domain_randomization": disabled})
                with self.assertRaises(StopBeforeSimulation):
                    cfg.build(num_envs=2)
        self.assertEqual(seen, [True, False])

    def test_reference_matches_online_local_frame_and_scales(self):
        scales = {"dof_pos": 0.5, "dof_vel": 0.1, "projected_gravity": 2.0, "base_ang_vel": 0.25}
        env = NS(
            default_dof_pos=torch.tensor([[0.1, -0.2]]), gravity_vec=torch.tensor([[0.0, 0.0, -1.0]]), config=NS(obs=NS(obs_scales=scales))
        )
        q = torch.tensor([[0.0, 0.0, 2.0**-0.5, 2.0**-0.5]])
        data = helpers.get_reference_proprioception(
            env, torch.tensor([[0.3, 0.4]]), torch.tensor([[1.0, 2.0]]), q, torch.tensor([[1.0, 0.0, 0.0]])
        )
        online = NS(
            _get_obs_dof_pos=lambda: torch.tensor([[0.2, 0.6]]),
            _get_obs_dof_vel=lambda: torch.tensor([[1.0, 2.0]]),
            _get_obs_projected_gravity=lambda: torch.tensor([[0.0, 0.0, -1.0]]),
            _get_obs_base_ang_vel=lambda: torch.tensor([[0.0, -1.0, 0.0]]),
        )
        expected = {}
        helpers.parse_observation(online, scales, expected, scales, {k: 0.0 for k in scales}, 1.0, use_noise=False)
        for key in data:
            torch.testing.assert_close(data[key], expected[key], atol=1e-6, rtol=1e-6)

    def test_expert_training_and_both_inference_builders_agree(self):
        from humanoidverse.agents.evaluations import humanoidverse_isaac as evaluation

        count, joints = 4, 29
        env = FakeEnv(1)
        env.dt = 1.0
        env.gravity_vec = torch.tensor([[0.0, 0.0, -1.0]])
        env.use_contact_in_obs_max = False
        dims = {"actions": joints, "dof_pos": joints, "dof_vel": joints, "base_ang_vel": 3, "projected_gravity": 3}
        env.config.obs = {
            "use_obs_filter": True,
            "root_height_obs": False,
            "obs_scales": {"dof_pos": 0.5, "dof_vel": 0.1, "base_ang_vel": 0.25, "projected_gravity": 1.0},
            "obs_auxiliary": {"history_actor": {k: 1 for k in dims}},
            "obs_dims": dims,
        }
        motion = {
            "rg_pos_t": torch.zeros(count, 2, 3),
            "rg_rot_t": torch.tensor([0.0, 0.0, 2.0**-0.5, 2.0**-0.5]).repeat(count, 2, 1),
            "body_vel_t": torch.zeros(count, 2, 3),
            "body_ang_vel_t": torch.tensor([1.0, 0.0, 0.0]).repeat(count, 2, 1),
            "dof_pos": env.default_dof_pos.repeat(count, 1) + 1.0,
            "dof_vel": torch.ones(count, joints),
        }
        for key, value in [
            ("root_pos", "rg_pos_t"),
            ("root_rot", "rg_rot_t"),
            ("root_vel", "body_vel_t"),
            ("root_ang_vel", "body_ang_vel_t"),
        ]:
            motion[key] = motion[value][:, 0]
        env._motion_lib = NS(
            _num_unique_motions=1,
            _motion_lengths=torch.tensor([float(count)]),
            _motion_data_keys=["test"],
            load_motions_for_training=lambda: None,
            get_motion_state=lambda *args: motion,
        )
        expert = hv.load_expert_trajectories_from_motion_lib(env, NS(model=NS(seq_length=1)), add_history_noaction=True)
        inference, _ = helpers.get_backward_observation(env, 0)
        evaluated, _ = evaluation.get_backward_observation(env, 0, include_last_action=True)
        expected = torch.cat(
            [
                torch.full((count, joints), 0.5),
                torch.full((count, joints), 0.1),
                torch.tensor([0.0, 0.0, -1.0]).repeat(count, 1),
                torch.tensor([0.0, -0.25, 0.0]).repeat(count, 1),
            ],
            dim=-1,
        )
        for state in (expert.storage["observation"]["state"], inference["state"], evaluated["state"]):
            torch.testing.assert_close(state, expected, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(expert.storage["observation"]["history_noaction"][1, :3], expected[0, -3:], atol=1e-6, rtol=1e-6)

        # Goal inference requests zero reference velocities; both builders must respect it.
        for builder in (helpers.get_backward_observation, evaluation.get_backward_observation):
            kwargs = {"include_last_action": True} if builder is evaluation.get_backward_observation else {}
            stationary, _ = builder(env, 0, velocity_multiplier=0.0, **kwargs)
            self.assertEqual(stationary["state"][:, joints:2 * joints].count_nonzero(), 0)
            self.assertEqual(stationary["state"][:, -3:].count_nonzero(), 0)
            torch.testing.assert_close(stationary["state"][:, :joints], expected[:, :joints])


    def test_tracking_runs_every_motion_to_its_full_length(self):
        from humanoidverse import tracking_inference as tracking

        steps = []
        env = FakeEnv(1)
        env.simulator.dof_state = torch.zeros(1, 29, 2)
        env.set_is_evaluating = lambda motion_id: steps.append([motion_id, 0])
        env.reset_envs_idx = lambda *args, **kwargs: ({}, {})
        wrapped = wrapper(env)

        def step(action, **kwargs):
            steps[-1][1] += 1
            return {}, 0, False, False, {}

        wrapped.step = step
        model = NS(
            _actor=NS(input_filter=NS(output_space=NS(shape=[64]))),
            cfg=NS(archi=NS(z_dim=2, actor=NS(input_filter=NS(key=[])))),
            to=lambda *args: None,
            eval=lambda: None,
            backward_map=lambda obs: obs["state"],
            project_z=lambda z: z,
            act=lambda *args, **kwargs: torch.zeros(1, 29),
        )

        def reference(*args, **kwargs):
            length = {0: 104, 1: 6}[steps[-1][0]]
            return {"state": torch.zeros(length, 2)}, {
                "ref_body_pos": torch.zeros(length, 1, 3),
                "ref_body_rots": torch.tensor([0.0, 0.0, 0.0, 1.0]).repeat(length, 1, 1),
                "ref_body_vels": torch.zeros(length, 1, 3),
                "ref_body_angular_vels": torch.zeros(length, 1, 3),
                "dof_pos": torch.zeros(length, 29),
                "ref_dof_vel": torch.zeros(length, 29),
            }

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path / "config.json").write_text(json.dumps({"env": {"lafan_tail_path": folder, "hydra_overrides": []}}))
            with (
                patch.object(tracking, "load_model_from_checkpoint_dir", return_value=model),
                patch.object(tracking, "export_meta_policy_as_onnx"),
                patch.object(tracking, "get_backward_observation", side_effect=reference),
                patch.object(tracking.HumanoidVerseIsaacConfig, "build", return_value=(wrapped, {})),
                patch.object(
                    tracking,
                    "IsaacRendererWithMuJoco",
                    return_value=NS(
                        from_qpos=lambda poses: [np.zeros((1, 1, 3), dtype=np.uint8) for _ in poses],
                        render=lambda *args: [np.zeros((1, 1, 3), dtype=np.uint8)],
                    ),
                ),
                patch.object(tracking.media, "write_video") as write_video,
                patch("builtins.print"),
            ):
                tracking.main(path, device="cpu", simulator="mujoco", motion_list=[0, 1], save_mp4=True)
                # One initialization step plus all (length - 1) target latents.
                self.assertEqual(steps, [[0, 104], [1, 6]])
                self.assertEqual([Path(call.args[0]).name for call in write_video.call_args_list], ["tracking_0.mp4", "tracking_1.mp4"])
                self.assertEqual([len(call.args[1]) for call in write_video.call_args_list], [104, 6])

                steps.clear()
                write_video.reset_mock()
                tracking.main(path, device="cpu", simulator="mujoco", motion_list=[0], save_mp4=True)
                self.assertEqual(steps, [[0, 104]])
                write_video.assert_called_once()
                self.assertEqual(Path(write_video.call_args.args[0]).name, "tracking.mp4")
                self.assertEqual(len(write_video.call_args.args[1]), 104)
            self.assertTrue((path / "tracking_inference/zs_0.pkl").exists())
            self.assertTrue((path / "tracking_inference/zs_1.pkl").exists())


if __name__ == "__main__":
    unittest.main()
