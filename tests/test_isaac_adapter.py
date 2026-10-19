"""Exercise the actual adapter methods on CPU without starting Omniverse."""
import ast
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import torch
from omegaconf import OmegaConf
from humanoidverse.envs.legged_base_task.legged_robot_base import LeggedRobotBase
from humanoidverse.utils.torch_utils import my_quat_rotate


def adapter_class():
    path = Path(__file__).parents[1] / 'humanoidverse/simulator/isaacsim/isaacsim.py'
    tree = ast.parse(path.read_text())
    names = {'robot_root_states', 'all_root_states', '_rigid_body_vel', 'set_actor_root_state_tensor',
             'set_pd_gains', 'apply_torques_at_dof'}
    methods = [node for cls in tree.body if isinstance(cls, ast.ClassDef)
               for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {'torch': torch, 'my_quat_rotate': my_quat_rotate}
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(path), 'exec'), namespace)
    return type('Adapter', (), {node.name: namespace[node.name] for node in methods})


class IsaacAdapterTests(unittest.TestCase):
    def test_velocity_uses_link_origin_and_keeps_simulator_com_state(self):
        # A link rotating about its fixed origin has a moving COM but zero origin velocity.
        for quat, com_velocity in [([1., 0., 0., 0.], [0., .1, 0.]),
                                   ([2**-.5, 0., 0., 2**-.5], [-.1, 0., 0.])]:
            root = torch.tensor([[1., 2., 3., *quat, *com_velocity, 0., 0., 1.]])
            original = root.clone()
            sim = adapter_class()()
            sim._robot = NS(data=NS(root_state_w=root, com_pos_b=torch.tensor([[[.1, 0., 0.]]]),
                                   body_lin_vel_w=torch.tensor([[com_velocity]]), body_link_lin_vel_w=torch.zeros(1, 1, 3)))
            sim.body_ids = [0]
            torch.testing.assert_close(sim.robot_root_states[:, 7:10], torch.zeros(1, 3), atol=1e-7, rtol=1e-7)
            torch.testing.assert_close(sim.robot_root_states[:, 3:7], root[:, [4, 5, 6, 3]])
            torch.testing.assert_close(sim.all_root_states, sim.robot_root_states)
            torch.testing.assert_close(sim._rigid_body_vel, torch.zeros(1, 1, 3))
            torch.testing.assert_close(root, original)

    def test_root_reset_uses_link_velocity_api_after_pose_and_preserves_input(self):
        sim = adapter_class()()
        calls = []
        sim._robot = NS(write_root_link_pose_to_sim=lambda pose, ids: calls.append(('pose', pose.clone(), ids)),
                        write_root_link_velocity_to_sim=lambda vel, ids: calls.append(('velocity', vel.clone(), ids)))
        state = torch.arange(26.).reshape(2, 13)
        original = state.clone()
        sim.set_actor_root_state_tensor(torch.tensor([1]), state)
        self.assertEqual([call[0] for call in calls], ['pose', 'velocity'])
        torch.testing.assert_close(calls[0][1], state[1:2, :7])
        torch.testing.assert_close(calls[1][1], state[1:2, 7:])
        torch.testing.assert_close(state, original)
        sim.set_actor_root_state_tensor(torch.tensor([], dtype=torch.long), state)
        self.assertEqual(len(calls), 2)

    def test_pd_and_torque_randomization_reaches_actuator(self):
        sim = adapter_class()()
        sim.dof_ids = [2, 0]
        data = NS(joint_stiffness=torch.full((2, 3), 100.), joint_damping=torch.ones(2, 3))
        actuator = NS(joint_indices=slice(None), stiffness=data.joint_stiffness.clone(), damping=data.joint_damping.clone())
        def write(field, value, joint_ids, env_ids):
            getattr(data, field)[env_ids[:, None], joint_ids] = value
        efforts, targets = [], []
        sim._robot = NS(data=data, actuators={'all': actuator},
            write_joint_stiffness_to_sim=lambda value, **kw: write('joint_stiffness', value, **kw),
            write_joint_damping_to_sim=lambda value, **kw: write('joint_damping', value, **kw),
            set_joint_effort_target=lambda value, **kw: efforts.append(value.clone()),
            set_joint_position_target=lambda value, **kw: targets.append(value.clone()))
        env = NS(simulator=sim, num_dofs=2, device='cpu', _kp_scale=torch.ones(2, 2), _kd_scale=torch.ones(2, 2),
                 p_gains=torch.full((2,), 100.), d_gains=torch.ones(2), _rfi_lim_scale=torch.ones(2, 2),
                 torque_limits=torch.full((2,), 100.), _torque_noise=torch.zeros(2, 2), torques=torch.zeros(2, 2),
                 actions_after_delay=torch.ones(2, 2), default_dof_pos=torch.zeros(1, 2), default_dof_pos_offset=torch.zeros(2, 2),
                 config=OmegaConf.create({'simulator': {'config': {'name': 'isaacsim'}},
                    'robot': {'control': {'action_scale': .25, 'action_rescale': False, 'control_type': 'P', 'clip_torques': True}},
                    'domain_rand': {'randomize_pd_gain': True, 'kp_range': [.5, .5], 'kd_range': [2., 2.],
                        'randomize_rfi_lim': False, 'randomize_ctrl_delay': False, 'randomize_default_dof_pos': False,
                        'randomize_torque_rfi': True, 'rfi_lim': .1}}))
        sim.dof_pos = torch.zeros(2, 2); sim.dof_vel = torch.zeros(2, 2)
        LeggedRobotBase._episodic_domain_randomization(env, torch.tensor([1]))
        torch.testing.assert_close(data.joint_stiffness, torch.tensor([[100., 100., 100.], [50., 100., 50.]]))
        torch.testing.assert_close(data.joint_damping, torch.tensor([[1., 1., 1.], [2., 1., 2.]]))
        torch.testing.assert_close(actuator.stiffness, data.joint_stiffness)
        torch.testing.assert_close(actuator.damping, data.joint_damping)
        env._compute_torques = lambda actions: LeggedRobotBase._compute_torques(env, actions)
        with patch('torch.rand_like', side_effect=lambda value: torch.full_like(value, .75)):
            LeggedRobotBase._apply_force_in_physics_step(env)
        torch.testing.assert_close(efforts[-1], torch.full((2, 2), 5.))
        torch.testing.assert_close(env.torques, torch.tensor([[30., 30.], [17.5, 17.5]]))
        env.config.domain_rand.randomize_torque_rfi = False
        LeggedRobotBase._apply_force_in_physics_step(env)
        torch.testing.assert_close(efforts[-1], torch.zeros(2, 2))
        torch.testing.assert_close(targets[0], targets[1])

    def test_push_is_visible_in_same_step_observation_and_history(self):
        root = torch.zeros(2, 13); root[:, 6] = 1.; root[1, 10] = .2
        history = []
        env = NS(simulator=NS(robot_root_states=root, base_quat=root[:, 3:7]),
                 config=OmegaConf.create({'simulator': {'config': {'name': 'isaacsim'}}, 'normalization': {'clip_observations': 100.}}),
                 base_quat=root[:, 3:7].clone(), rpy=torch.zeros(2, 3), base_lin_vel=torch.zeros(2, 3),
                 base_ang_vel=torch.zeros(2, 3), projected_gravity=torch.zeros(2, 3), gravity_vec=torch.tensor([[0., 0., -1.]]).repeat(2, 1),
                 episode_length_buf=torch.zeros(2,dtype=torch.long), reset_buf=torch.zeros(2,dtype=torch.bool),
                 need_to_refresh_envs=torch.zeros(2,dtype=torch.bool), extras={}, log_dict={}, viewer=None,
                 history_handler=NS(history={'state': None}, add=lambda key, value: history.append(value.clone())))
        env.simulator.trigger_on_push_by_setting_velocity_events = lambda ids: root.__setitem__((ids, 10), .5)
        for name in ('_refresh_sim_tensors', '_update_counters_each_step', '_check_termination', '_compute_reward', '_post_compute_observations_callback'):
            setattr(env, name, lambda: None)
        env.reset_envs_idx = lambda ids: None
        env._pre_compute_observations_callback = lambda: LeggedRobotBase._pre_compute_observations_callback(env)
        env._update_tasks_callback = lambda: LeggedRobotBase._push_robots(env, torch.tensor([0]))
        def observe():
            env.obs_buf_dict = {'actor_obs': env.base_ang_vel.clone()}
            env.hist_obs_dict = {'state': env.base_ang_vel.clone()}
        env._compute_observations = observe
        LeggedRobotBase._post_physics_step(env)
        torch.testing.assert_close(env.obs_buf_dict['actor_obs'], root[:, 10:13])
        torch.testing.assert_close(history[0], root[:, 10:13])
