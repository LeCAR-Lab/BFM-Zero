import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import numpy as np
import safetensors.torch
import torch
from test_correctness import FakeEnv, wrapper

from humanoidverse import train
from humanoidverse.agents import base_model
from humanoidverse.agents.fb.agent import FBAgent, FBAgentTrainConfig


def default_config():
    return train.default_train_config()


class TrainingTests(unittest.TestCase):
    def test_legacy_replay_rejected_before_environment_or_files_change(self):
        cfg = default_config()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "checkpoint/buffers/train").mkdir(parents=True)
            (root / "checkpoint/train_status.json").write_text('{"time": 100}')
            original = cfg.model_dump_json()
            (root / "config.json").write_text(original)
            with patch.object(type(cfg.env), "build") as build:
                with self.assertRaisesRegex(ValueError, "Legacy replay"):
                    train.Workspace(cfg.model_copy(update={"work_dir": folder}))
                build.assert_not_called()
            self.assertEqual((root / "config.json").read_text(), original)

    def test_expert_rollouts_use_exact_number_of_distinct_environments(self):
        agent = FBAgent.__new__(FBAgent)
        agent.cfg = NS(
            train=FBAgentTrainConfig(rollout_expert_trajectories=True, rollout_expert_trajectories_percentage=0.5),
            model=NS(archi=NS(z_dim=2)),
        )
        agent._model = NS(device="cpu", sample_z=lambda count, device: torch.zeros(count, 2))
        agent._sample_tracking_z = lambda replay, count, length: torch.ones(count, length, 2)
        z = None
        for step in (0, 1, 250):
            z = agent.maybe_update_rollout_context(z, torch.full((1024, 1), step))
            self.assertEqual(agent.env_idx_with_expert_rollout.unique().numel(), 512)
            torch.testing.assert_close(z[agent.env_idx_with_expert_rollout], torch.ones(512, 2))
        agent._sample_tracking_z = FBAgent._sample_tracking_z.__get__(agent)
        self.assertEqual(tuple(agent._sample_tracking_z({}, 0, 250).shape), (0, 250, 2))
        for fraction in (-0.1, 1.1):
            with self.assertRaises(ValueError):
                FBAgentTrainConfig(rollout_expert_trajectories_percentage=fraction)

    def test_resume_rejects_config_drift_but_accepts_new_budget_and_device(self):
        cfg = default_config()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path / "checkpoint").mkdir()
            (path / "config.json").write_text(cfg.model_dump_json())
            (path / "checkpoint/config.json").write_text(cfg.agent.model_dump_json())
            train.validate_resume_config(path, cfg)
            train.validate_resume_config(path, cfg.model_copy(update={"num_env_steps": 999999999, "use_wandb": True}))
            moved_agent = cfg.agent.model_copy(update={"model": cfg.agent.model.model_copy(update={"device": "cpu"})})
            train.validate_resume_config(path, cfg.model_copy(update={"agent": moved_agent}))
            for field, value in [
                ("env", cfg.env.model_copy(update={"root_height_obs": not cfg.env.root_height_obs})),
                ("agent", cfg.agent.model_copy(update={"train": cfg.agent.train.model_copy(update={"lr_actor": 0.1})})),
                ("seed", cfg.seed + 1),
            ]:
                with self.assertRaisesRegex(ValueError, field):
                    train.validate_resume_config(path, cfg.model_copy(update={field: value}))
            # Even a mislabeled top-level config cannot override the actual saved agent.
            bad = cfg.model_dump(mode="json")
            bad["agent"]["train"]["lr_actor"] = 0.1
            (path / "config.json").write_text(json.dumps(bad))
            changed_agent = cfg.agent.model_copy(update={"train": cfg.agent.train.model_copy(update={"lr_actor": 0.1})})
            with self.assertRaisesRegex(ValueError, "agent"):
                train.validate_resume_config(path, cfg.model_copy(update={"agent": changed_agent}))

    def test_seed_is_set_before_environment_startup(self):
        cfg = default_config()
        samples = []

        class StopAtBuild(Exception):
            pass

        def capture(*args, **kwargs):
            samples.append((np.random.rand(), torch.rand(1).item()))
            raise StopAtBuild

        with tempfile.TemporaryDirectory() as folder, patch.object(type(cfg.env), "build", side_effect=capture):
            cfg = cfg.model_copy(update={"work_dir": folder})
            for seed in (13, 77):
                np.random.seed(seed)
                torch.manual_seed(seed)
                with self.assertRaises(StopAtBuild):
                    train.Workspace(cfg)
        self.assertEqual(samples[0], samples[1])


    def test_fb_cpr_aux_update_and_checkpoint_roundtrip(self):
        import gymnasium as gym
        import safetensors.torch
        from torch.utils._pytree import tree_map

        from humanoidverse.agents.buffers.trajectory import TrajectoryDictBufferMultiDim, get_idxs
        from humanoidverse.agents.buffers.transition import DictBuffer

        torch.manual_seed(4728)
        cfg = default_config().agent
        arch = cfg.model.archi
        networks = {
            key: getattr(arch, key).model_copy(update={"hidden_dim": 32, "hidden_layers": 2})
            for key in ("f", "b", "actor", "critic", "aux_critic", "discriminator")
        }
        networks["aux_critic"] = networks["aux_critic"].model_copy(update={"hidden_dim": 16, "num_parallel": 3})
        arch = arch.model_copy(update={**networks, "z_dim": 8})
        model_cfg = cfg.model.model_copy(update={"archi": arch, "device": "cpu", "seq_length": 2})
        cfg = cfg.model_copy(
            update={"model": model_cfg, "train": cfg.train.model_copy(update={"batch_size": 8}), "compile": False, "cudagraphs": False}
        )
        spaces = gym.spaces.Dict(
            {
                key: gym.spaces.Box(-np.inf, np.inf, (dim,), dtype=np.float32)
                for key, dim in [("state", 4), ("privileged_state", 6), ("last_action", 2), ("history_actor", 8)]
            }
        )
        agent = cfg.build(spaces, 2)
        self.assertEqual(agent._model._aux_critic.hidden_dim, 16)
        self.assertEqual(agent._model._aux_critic.num_parallel, 3)
        replay = TrajectoryDictBufferMultiDim(
            capacity=8, n_dim=2, end_key="done", output_key_tp1=[], output_key_t=["observation", "action", "next", "z", "aux_rewards"]
        )
        replay._get_idxs = get_idxs
        expert = DictBuffer(capacity=32)
        for step in range(4):
            obs = {key: torch.randn(4, *space.shape) for key, space in spaces.items()}
            next_obs = {key: torch.randn_like(value) for key, value in obs.items()}
            info = {"aux_rewards": {key: torch.randn(4) for key in cfg.aux_rewards}}
            data = train.make_transition(
                obs,
                torch.randn(4, 2),
                next_obs,
                torch.zeros(4),
                torch.tensor([False, False, step == 3, False]),
                torch.tensor([step == 1, False, False, False]),
                {},
                info,
                torch.full((4,), step),
            )
            data["z"] = agent._model.sample_z(4)
            replay.extend(tree_map(lambda x: x[None], data))
            expert.extend({"observation": obs, "next": {"observation": next_obs}, "done": torch.zeros(4, 1, dtype=torch.bool)})
        before = next(agent._model._actor.parameters()).detach().clone()
        for step in range(2):
            metrics = agent.update({"train": replay, "expert_slicer": expert}, step)
            for key, value in metrics.items():
                self.assertTrue(torch.isfinite(value).all(), key)
        self.assertFalse(torch.equal(before, next(agent._model._actor.parameters())))
        with tempfile.TemporaryDirectory() as folder:
            agent.save(folder)
            loaded = type(agent).load(folder, device="cpu")
            for key, value in agent._model.state_dict().items():
                torch.testing.assert_close(loaded._model.state_dict()[key], value)
            inference = type(agent._model).load(Path(folder) / "model", device="cpu")
            for key, value in agent._model.state_dict().items():
                torch.testing.assert_close(inference.state_dict()[key], value)
            # A real agent checkpoint missing a non-aliased actor weight must fail too.
            path = Path(folder) / "model/model.safetensors"
            state = safetensors.torch.load_file(path)
            key = next(key for key in state if key.startswith("_actor.") and key.endswith("weight"))
            del state[key]
            safetensors.torch.save_file(state, path)
            with self.assertRaises(RuntimeError):
                type(agent).load(folder, device="cpu")

        # Legacy metadata described an aux architecture that was never actually built.
        legacy_arch = arch.model_copy(update={"aux_critic": arch.critic.model_copy(deep=True)})
        legacy_cfg = cfg.model_copy(update={"model": cfg.model.model_copy(update={"archi": legacy_arch})})
        legacy = legacy_cfg.build(spaces, 2)
        with tempfile.TemporaryDirectory() as folder:
            legacy.save(folder)
            for relative in ("config.json", "model/config.json"):
                path = Path(folder) / relative
                saved = json.loads(path.read_text())
                model_config = saved["model"] if relative == "config.json" else saved
                del model_config["aux_critic_config_version"]
                model_config["archi"]["aux_critic"] = arch.aux_critic.model_dump()
                path.write_text(json.dumps(saved))
            loaded_agent = type(legacy).load(folder, device="cpu")
            loaded_model = type(legacy._model).load(Path(folder) / "model", device="cpu")
            for key, value in legacy._model.state_dict().items():
                torch.testing.assert_close(loaded_agent._model.state_dict()[key], value)
                torch.testing.assert_close(loaded_model.state_dict()[key], value)


    def test_reset_seed_includes_action_space(self):
        from humanoidverse.envs.gymnasium_wrapper import HumanoidVerseVectorEnv

        for wrapped in (wrapper(FakeEnv()), HumanoidVerseVectorEnv(FakeEnv())):
            wrapped.reset(seed=4728)
            first = wrapped.action_space.sample()
            wrapped.reset(seed=4728)
            np.testing.assert_array_equal(wrapped.action_space.sample(), first)


    def test_strict_load_rejects_missing_actor_and_preserves_aliases(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.actor = torch.nn.Linear(2, 2)
                self.actor_alias = self.actor

        class Config:
            def __init__(self, **kwargs):
                pass

            def build(self, **kwargs):
                return Model()

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path / "config.json").write_text("{}")
            model = Model()
            safetensors.torch.save_model(model, path / "model.safetensors")
            loaded = base_model.load_model(folder, "cpu", True, Config, {})
            torch.testing.assert_close(loaded.actor.weight, model.actor.weight)
            self.assertIs(loaded.actor, loaded.actor_alias)
            safetensors.torch.save_file({"actor.bias": model.actor.bias.detach()}, path / "model.safetensors")
            with self.assertRaises(RuntimeError):
                base_model.load_model(folder, "cpu", True, Config, {})
            base_model.load_model(folder, "cpu", False, Config, {})


if __name__ == "__main__":
    unittest.main()
