import tempfile
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

import numpy as np
import torch
from test_correctness import FakeEnv, wrapper
from torch.utils._pytree import tree_map

from humanoidverse import train
from humanoidverse.agents.buffers.trajectory import TrajectoryDictBufferMultiDim, get_idxs
from humanoidverse.envs.legged_base_task.legged_robot_base import LeggedRobotBase


def buffer(capacity=4):
    replay = TrajectoryDictBufferMultiDim(
        capacity=capacity, n_dim=2, end_key="done", output_key_t=["observation", "action", "next"], output_key_tp1=[]
    )
    # These tests exercise indexing eagerly; a separate check exercises compilation.
    replay._get_idxs = get_idxs
    return replay


class ReplayTests(unittest.TestCase):
    def test_reward_inference_preserves_explicit_successors_after_reload(self):
        from humanoidverse.envs.g1_env_helper.bench import reward_eval_hv as reward

        replay = buffer()
        data = train.make_transition(
            {"state": torch.zeros(1, 2)}, torch.ones(1, 1),
            {"state": torch.ones(1, 2)}, torch.ones(1),
            torch.tensor([False]), torch.tensor([True]),
            {"qpos": torch.zeros(1, 8), "qvel": torch.zeros(1, 7)},
            {"qpos": torch.ones(1, 8), "qvel": torch.ones(1, 7)}, torch.zeros(1),
        )
        replay.extend(tree_map(lambda value: value[None], data))
        replay.extend(tree_map(lambda value: value[None], data))
        with tempfile.TemporaryDirectory() as folder:
            replay.save(folder)
            loaded = TrajectoryDictBufferMultiDim.load(folder)
            loaded._get_idxs = get_idxs
            for count in (1, 2):
                seen = []
                def infer(**kwargs):
                    seen.append(kwargs)
                    return torch.ones(2)
                fake = NS(env_model=object(), inference_dataset=loaded,
                          num_samples_per_inference=count, max_workers=1,
                          process_executor=False, device="cpu",
                          model=NS(infer=infer), inference_function="infer")
                with patch.object(reward, "relabel", return_value=np.ones((1, 1))) as relabel, patch.object(reward, "make_from_name"):
                    reward.RewardWrapperHV.reward_inference(fake, "test")
                np.testing.assert_array_equal(relabel.call_args.args[1], np.ones((count, 8)))
                torch.testing.assert_close(seen[0]["next_obs"]["state"], torch.ones(count, 2))
                self.assertEqual(loaded.output_key_tp1, [])

    def test_partial_autoreset_keeps_physical_successor_and_history(self):
        env = FakeEnv(2)
        env.episode_length_buf[:] = 5
        env.simulator.dof_pos[:] = torch.tensor([[6.0], [16.0]])
        env.base_ang_vel = torch.zeros(2, 3)
        env.reset_buf = torch.tensor([1, 0])
        env.need_to_refresh_envs = torch.zeros(2, dtype=torch.bool)
        env.target_robot_root_states = env.simulator.robot_root_states.clone()
        env.target_robot_dof_state = torch.zeros(2, 29, 2)
        env.extras = {"time_outs": torch.tensor([True, False])}
        env.config.normalization = {"clip_observations": 1000.0}
        env.log_dict = {}
        env.viewer = None
        env.history_handler = NS(history={})
        env._refresh_sim_tensors = lambda: None
        env._update_counters_each_step = lambda: None
        env._pre_compute_observations_callback = lambda: None
        env._update_tasks_callback = lambda: None
        env._check_termination = lambda: None
        env._compute_reward = lambda: None
        env._post_compute_observations_callback = lambda: None

        def compute():
            env.obs_buf_dict_raw["actor_obs"]["dof_pos"] = env.simulator.dof_pos.clone()
            env.obs_buf_dict_raw["actor_obs"]["history_actor"] = env.episode_length_buf[:, None].float().repeat(1, 3)
            env.obs_buf_dict = {"actor_obs": env.simulator.dof_pos.clone()}

        env._compute_observations = compute

        def reset(ids):
            env.simulator.dof_pos[ids] = -100.0
            env.episode_length_buf[ids] = 0

        env.reset_envs_idx = reset

        def step(actions):
            LeggedRobotBase._post_physics_step(env)
            return {}, torch.ones(2), env.reset_buf, env.extras

        env.step = step
        wrapped = wrapper(env)
        wrapped.include_history_actor = True
        obs, reward, terminated, truncated, info = wrapped.step(torch.zeros(2, 29))
        self.assertEqual(obs["state"][0, 0], -100.0)
        self.assertEqual(info["final_observation"]["state"][0, 0], 6.0)
        self.assertEqual(info["final_observation"]["history_actor"][0, 0], 6.0)
        self.assertEqual(obs["history_actor"][0, 0], 0.0)
        data = train.make_transition(
            {"state": np.zeros((2, 64)), "history_actor": np.zeros((2, 3))},
            np.zeros((2, 29)),
            obs,
            reward,
            terminated,
            truncated,
            {},
            info,
            np.zeros(2),
        )
        self.assertEqual(data["next"]["observation"]["state"][0, 0], 6.0)
        self.assertEqual(data["next"]["observation"]["state"][1, 0], 16.0)
        self.assertFalse(data["next"]["terminated"][0])
        self.assertTrue(data["next"]["truncated"][0])
        env.reset_buf[:] = 0
        _, _, _, _, next_info = wrapped.step(torch.zeros(2, 29))
        self.assertNotIn("final_observation", next_info)

    def test_explicit_successors_survive_short_episodes_wrap_and_eval_cut(self):
        replay = buffer()
        for t in range(7):
            current = {"state": np.array([[t * 10.0], [t * 10.0 + 2.0]])}
            # Environment 0 times out every other step; env 1 truly terminates at t=2.
            terminated = np.array([False, t == 2])
            truncated = np.array([t % 2 == 1, t == 4])  # t=4 also represents an evaluation cut.
            ended = terminated | truncated
            physical = {"state": current["state"] + 1}
            returned = {"state": np.where(ended[:, None], -100.0, physical["state"])}
            info = {"_final_observation": ended, "final_observation": physical}
            data = train.make_transition(current, np.ones((2, 1)), returned, np.ones(2), terminated, truncated, {}, info, np.full(2, t))
            replay.extend(tree_map(lambda x: x[None], data))
        sample = replay.sample(4096)
        torch.testing.assert_close(sample["next"]["observation"]["state"], sample["observation"]["state"] + 1)
        self.assertEqual(set(sample["observation"]["state"].flatten().tolist()), {30.0, 32.0, 40.0, 42.0, 50.0, 52.0, 60.0, 62.0})
        full = replay.get_full_buffer()
        self.assertEqual(full["observation"]["state"].shape[0], 8)
        torch.testing.assert_close(full["next"]["observation"]["state"], full["observation"]["state"] + 1)

    def test_terminal_flags_and_current_step_rewards(self):
        data = train.make_transition(
            {"state": np.zeros((2, 1))},
            np.zeros((2, 1)),
            {"state": np.ones((2, 1))},
            np.array([3.0, 7.0]),
            np.array([True, False]),
            np.array([False, True]),
            {},
            {},
            np.zeros(2),
        )
        torch.testing.assert_close(data["reward"].flatten(), torch.tensor([3.0, 7.0], dtype=torch.float64))
        torch.testing.assert_close((~data["next"]["terminated"]).flatten(), torch.tensor([False, True]))

    def test_last_legal_start_is_sampled_and_no_episode_is_crossed(self):
        for offset, expected in [(0, {0, 1, 2}), (1, {0, 1})]:
            idx = get_idxs(2, 2000, torch.tensor([4]), torch.tensor([[0, 0]]), 4, offset=offset)
            starts = idx.view(2000, 2, 2)[:, 0, 0]
            self.assertEqual(set(starts.tolist()), expected)
        replay = buffer()
        single = {
            "observation": {"state": torch.tensor([[[1.0]]])},
            "action": torch.zeros(1, 1, 1),
            "next": {"observation": {"state": torch.tensor([[[2.0]]])}},
            "done": torch.ones(1, 1, 1, dtype=torch.bool),
        }
        replay.extend(single)
        self.assertEqual(replay.sample(1)["next"]["observation"]["state"].item(), 2.0)


if __name__ == "__main__":
    unittest.main()
