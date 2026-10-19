# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import os

from humanoidverse.agents.evaluations.humanoidverse_isaac import (
    HumanoidVerseIsaacTrackingEvaluation,
    HumanoidVerseIsaacTrackingEvaluationConfig,
)
from humanoidverse.agents.envs.humanoidverse_isaac import load_expert_trajectories_from_motion_lib, HumanoidVerseIsaacConfig

os.environ["OMP_NUM_THREADS"] = "1"

import torch

torch.set_float32_matmul_precision("high")

import json
import shutil
import tempfile
import time
import typing as tp
import warnings
from pathlib import Path
from typing import Dict, List
from torch.utils._pytree import tree_map

import exca as xk
import gymnasium
import numpy as np
import pydantic
import torch  # better to use scoped import if we use processes
import tyro
import wandb
from torch.utils._pytree import tree_map
from tqdm import tqdm


from humanoidverse.agents.base import BaseConfig
from humanoidverse.agents.base_model import migrate_legacy_aux_critic_config
from humanoidverse.agents.buffers.load_data import load_expert_trajectories
from humanoidverse.agents.buffers.trajectory import TrajectoryDictBufferMultiDim
from humanoidverse.agents.buffers.transition import DictBuffer, dtype_numpytotorch_lower_precision
from humanoidverse.agents.fb_cpr.agent import FBcprAgentConfig
from humanoidverse.agents.fb_cpr_aux.agent import FBcprAuxAgentConfig
from humanoidverse.agents.misc.loggers import CSVLogger
from humanoidverse.agents.utils import EveryNStepsChecker, get_local_workdir, set_seed_everywhere

TRAIN_LOG_FILENAME = "train_log.txt"
REWARD_EVAL_LOG_FILENAME = "reward_eval_log.csv"
TRACKING_EVAL_LOG_FILENAME = "tracking_eval_log.csv"

CHECKPOINT_DIR_NAME = "checkpoint"

_ENC_CONFIG_TO_EXPERT_DATA_OBS_MAPPER = {
    HumanoidVerseIsaacConfig: None,
}



Evaluation = tp.Annotated[
    tp.Union[
        HumanoidVerseIsaacTrackingEvaluationConfig,
    ],
    pydantic.Field(discriminator="name"),
]

Agent = FBcprAgentConfig | FBcprAuxAgentConfig


class TrainConfig(BaseConfig):
    # The "pydantic.Field" field is used to explicitely tell which field is the discriminative
    # feature
    agent: Agent = pydantic.Field(discriminator="name")
    motions: str | None = None
    motions_root: str | None = None

    env: HumanoidVerseIsaacConfig = pydantic.Field(discriminator="name")

    work_dir: str = pydantic.Field(default_factory=lambda: get_local_workdir("g1mujoco_train"))

    seed: int = 0
    online_parallel_envs: int = 50
    # Note: this is in env steps (multiples of online_parallel_envs)
    log_every_updates: int = 100_000
    num_env_steps: int = 30_000_000
    # Note: this is in env steps (multiples of online_parallel_envs)
    update_agent_every: int = 500
    # Note: this is in env steps (multiples of online_parallel_envs)
    num_seed_steps: int = 50_000
    num_agent_updates: int = 50
    # Note: this is in env steps (multiples of online_parallel_envs)
    checkpoint_every_steps: int = 5_000_000
    checkpoint_buffer: bool = True
    prioritization: bool = False
    prioritization_min_val: float = 0.5
    prioritization_max_val: float = 5
    prioritization_scale: float = 2
    prioritization_mode: str = "bin"  # ["bin", "exp", "lin"]
    padding_beginning: int = 0
    padding_end: int = 0

    # Buffer
    use_trajectory_buffer: bool = False
    buffer_size: int = 5_000_000

    # WANDB
    use_wandb: bool = False
    wandb_ename: str | None = None
    wandb_gname: str | None = None
    wandb_pname: str | None = None

    # misc
    load_isaac_expert_data: bool = True
    buffer_device: str = "cpu"
    # Default to True; otherwise you will spam the console with tqdm
    disable_tqdm: bool = True

    # If you want to add more available evaluations, Update "Evaluations" type above
    evaluations: Dict[str, Evaluation] | List[Evaluation] = pydantic.Field(default_factory=lambda: [])
    # Note: this is in env steps (multiples of online_parallel_envs)
    eval_every_steps: int = 1_000_000

    tags: dict = pydantic.Field(default_factory=lambda: {})

    # exca
    infra: xk.TaskInfra = xk.TaskInfra(version="1")

    def model_post_init(self, context):
        # TODO prioritization needs tracking eval to work, but this is bit hacky to check for it
        if self.load_isaac_expert_data and not isinstance(self.env, HumanoidVerseIsaacConfig):
            raise ValueError("Loading expert isaac data is only supported for HumanoidVerseIsaacConfig")

        if self.prioritization:
            has_prioritization_eval = False
            for eval_type in self.evaluations:
                if isinstance(eval_type, (HumanoidVerseIsaacTrackingEvaluationConfig)):
                    has_prioritization_eval = True
                    break
            if not has_prioritization_eval:
                raise ValueError("Prioritization requires tracking evaluation to be enabled")


        if self.motions is None or self.motions_root is None:
            if self.prioritization:
                raise ValueError("Prioritization requires expert data to be provided (motions and motions_root)")
            elif self.agent == FBcprAgentConfig:
                # TODO how to do checks like these in pydantic or more systematically?
                raise ValueError("FBcprAgent requires expert data to be provided (motions and motions_root)")

        # Ensure all evaluations have unique log names
        if isinstance(self.evaluations, list):
            log_names = set()
            for eval_cfg in self.evaluations:
                if eval_cfg.name_in_logs in log_names:
                    raise ValueError(
                        f"Duplicate evaluation name_in_logs found: {eval_cfg.name}. These should be unique so we do not overwrite any logs"
                    )
                log_names.add(eval_cfg.name_in_logs)

    def build(self):
        """In case of cluster run, use exca and process instead of explivit build"""
        return Workspace(self)


def checkpoint_path(work_dir: Path) -> Path:
    path = work_dir / CHECKPOINT_DIR_NAME
    previous = work_dir / (CHECKPOINT_DIR_NAME + ".previous")
    # A process can stop between the two directory renames during publication.
    return previous if not path.exists() and previous.exists() else path


def validate_resume_config(work_dir: Path, cfg: TrainConfig):
    checkpoint_dir = checkpoint_path(work_dir)
    if not checkpoint_dir.exists():
        return
    if (checkpoint_dir / "buffers/train").exists():
        with (checkpoint_dir / "train_status.json").open() as f:
            status = json.load(f)
        if status.get("replay_format") != "explicit_successor_v1":
            raise ValueError("Legacy replay has no verified physical successors. Start a new work_dir; model weights remain loadable for inference.")
    with (work_dir / "config.json").open() as f:
        previous = json.load(f)
    current = cfg.model_dump(mode="json")
    # Logging, device placement and the remaining budget may change on resume.
    operational = {
        "work_dir", "num_env_steps", "log_every_updates", "checkpoint_every_steps", "checkpoint_buffer",
        "buffer_device", "disable_tqdm", "use_wandb", "wandb_ename", "wandb_gname", "wandb_pname", "tags", "infra",
    }
    with (checkpoint_dir / "config.json").open() as f:
        previous["agent"] = json.load(f)
    migrate_legacy_aux_critic_config(previous["agent"]["model"])
    for config in (previous, current):
        config["agent"]["model"].pop("device", None)
        config["env"].pop("device", None)
    changed = sorted(key for key in previous.keys() | current.keys() if key not in operational and previous.get(key) != current.get(key))
    if changed:
        raise ValueError(f"Resume configuration differs in {changed}. Use the saved configuration or a new work_dir.")


def create_agent_or_load_checkpoint(work_dir: Path, cfg: TrainConfig, agent_build_kwargs: dict[str, tp.Any]):
    checkpoint_dir = checkpoint_path(work_dir)
    checkpoint_time = 0
    if checkpoint_dir.exists():
        with (checkpoint_dir / "train_status.json").open("r") as f:
            train_status = json.load(f)
        checkpoint_time = train_status["time"]

        print(f"Loading the agent at time {checkpoint_time}")
        agent = cfg.agent.object_class.load(checkpoint_dir, device=cfg.agent.model.device)
    else:
        agent = cfg.agent.build(**agent_build_kwargs)
    return agent, cfg, checkpoint_time


def make_transition(observation, action, next_observation, reward, terminated, truncated, info, next_info, step_count):
    """Store same-step autoreset successors explicitly, including timeout bootstraps."""
    def tensor(value):
        return torch.as_tensor(value).detach().clone()

    current = {key: tensor(value) for key, value in observation.items() if key not in ("time", "history")}
    successor = {key: tensor(value) for key, value in next_observation.items() if key not in ("time", "history")}
    final_mask = next_info.get("_final_observation")
    if final_mask is not None:
        for key, value in successor.items():
            mask = torch.as_tensor(final_mask, device=value.device, dtype=torch.bool)
            value[mask] = torch.as_tensor(next_info["final_observation"][key], device=value.device)[mask]
    data = {
        "observation": current,
        "action": tensor(action),
        "step_count": tensor(step_count).reshape(-1, 1),
        "reward": tensor(reward).reshape(-1, 1),
        "next": {
            "observation": successor,
            "terminated": tensor(terminated).reshape(-1, 1),
            "truncated": tensor(truncated).reshape(-1, 1),
        },
    }
    data["done"] = data["next"]["terminated"] | data["next"]["truncated"]
    for key in ("qpos", "qvel"):
        if key in info:
            data[key] = tensor(info[key])
            data["next"][key] = tensor(next_info[key])
            if final_mask is not None:
                value = data["next"][key]
                mask = torch.as_tensor(final_mask, device=value.device, dtype=torch.bool)
                value[mask] = torch.as_tensor(next_info["final_info"][key], device=value.device)[mask]
    if "aux_rewards" in next_info:
        data["aux_rewards"] = {key: tensor(value).reshape(-1, 1) for key, value in next_info["aux_rewards"].items() if not key.startswith("_")}
    return data


def init_wandb(cfg: TrainConfig):
    exp_name = "BFM-Zero"
    wandb_name = exp_name
    wandb_config = cfg.model_dump()
    wandb.init(entity=cfg.wandb_ename, project=cfg.wandb_pname, group=cfg.wandb_gname, name=wandb_name, config=wandb_config, dir="./_wandb")


class Workspace:
    def __init__(self, cfg: TrainConfig) -> None:
        self.cfg = cfg
        validate_resume_config(Path(cfg.work_dir), cfg)
        set_seed_everywhere(cfg.seed)

        # HACK with Isaac, we can not recreate environments with current code, so we need to
        #      create the environment with desired number of envs here
        if isinstance(cfg.env, HumanoidVerseIsaacConfig):
            from omegaconf import OmegaConf

            self.train_env, self.train_env_info = cfg.env.build(num_envs=cfg.online_parallel_envs)
            self.train_env.action_space.seed(cfg.seed)
            self.train_env.single_action_space.seed(cfg.seed)
            self.obs_space = self.train_env.single_observation_space
            self.action_space = self.train_env.single_action_space
        else:
            sample_env, _ = cfg.env.build(num_envs=1)
            self.obs_space = sample_env.observation_space
            self.action_space = sample_env.action_space

        assert "time" in self.obs_space.keys(), "Observation space must contain 'obs' and 'time' (TimeAwareObservation wrapper)"
        assert len(self.action_space.shape) == 1, "Only 1D action space is supported (first dim should be vector env)"
        # TODO for backwards consistency, we do not pass "time" to the agent, so we remove it from the obs_space we pass to the agent/model
        #      but would we need it at some point?
        del self.obs_space.spaces["time"]

        self.action_dim = self.action_space.shape[0]

        print(f"Workdir: {self.cfg.work_dir}")
        self.work_dir = Path(self.cfg.work_dir)
        self.work_dir.mkdir(exist_ok=True, parents=True)

        if isinstance(cfg.env, HumanoidVerseIsaacConfig):
            with open(self.work_dir / "config.yaml", "w") as file:
                OmegaConf.save(self.train_env_info["unresolved_conf"], file)

        self.train_logger = CSVLogger(filename=self.work_dir / TRAIN_LOG_FILENAME)

        self.agent, self.cfg, self._checkpoint_time = create_agent_or_load_checkpoint(
            self.work_dir, self.cfg, agent_build_kwargs=dict(obs_space=self.obs_space, action_dim=self.action_dim)
        )
        self.agent._model.train()

        if isinstance(self.cfg.evaluations, list):
            self.evaluations = {eval_cfg.name_in_logs: eval_cfg.build() for eval_cfg in self.cfg.evaluations}
        else:
            self.evaluations = {eval_cfg: eval_cfg.build() for name, eval_cfg in self.cfg.evaluations.items()}
        self.evaluate = len(self.evaluations) > 0

        self.eval_loggers = {name: CSVLogger(filename=self.work_dir / f"{name}.csv") for name in self.evaluations.keys()}

        if self.cfg.use_wandb:
            init_wandb(self.cfg)

        with (self.work_dir / "config.json").open("w") as f:
            f.write(self.cfg.model_dump_json(indent=4))

        self.priorization_eval_name = None
        if self.cfg.prioritization:
            for name, evaluation in self.evaluations.items():
                if isinstance(evaluation.cfg, HumanoidVerseIsaacTrackingEvaluationConfig):
                    self.priorization_eval_name = name
                    break
            if self.priorization_eval_name is None:
                raise ValueError("Prioritization requires tracking evaluation to be enabled")

        self.training_with_expert_data = True

        self.manager = None

    def train(self):
        self.start_time = time.time()
        self.train_online()

    def train_online(self) -> None:
        if self.training_with_expert_data:
            if self.cfg.load_isaac_expert_data:
                expert_buffer = load_expert_trajectories_from_motion_lib(self.train_env._env, self.cfg.agent, device=self.cfg.buffer_device)
            else:
                print("Loading expert trajectories")
                expert_buffer = load_expert_trajectories(
                    self.cfg.motions,
                    self.cfg.motions_root,
                    seq_length=self.agent.cfg.model.seq_length,
                    device=self.cfg.buffer_device,
                    # TODO data stored in disk does not have dictionary obs, so we need to manually
                    #      define what obs key the data on disk corresponds to
                    obs_dict_mapper=_ENC_CONFIG_TO_EXPERT_DATA_OBS_MAPPER[self.cfg.env.__class__],
                )
        print("Creating the training environment")

        if isinstance(self.cfg.env, HumanoidVerseIsaacConfig):
            train_env = self.train_env
            train_env_info = self.train_env_info
        else:
            train_env, train_env_info = self.cfg.env.build(num_envs=self.cfg.online_parallel_envs)

        print("Allocating buffers")
        replay_buffer = {}
        checkpoint_dir = checkpoint_path(self.work_dir)
        if (checkpoint_dir / "buffers/train").exists():
            print("Loading checkpointed buffer")
            if self.cfg.use_trajectory_buffer:
                replay_buffer["train"] = TrajectoryDictBufferMultiDim.load(checkpoint_dir / "buffers/train", device=self.cfg.buffer_device)
            else:
                replay_buffer["train"] = DictBuffer.load(checkpoint_dir / "buffers/train", device=self.cfg.buffer_device)
            buffer = replay_buffer["train"]
            if self.cfg.use_trajectory_buffer and buffer.output_key_tp1:
                raise ValueError("Legacy trajectory replay has no explicit successors. Start a new work_dir; old buffer boundaries cannot be repaired.")
            if self.cfg.use_trajectory_buffer and len(buffer):
                buffer.storage["done"][buffer._idx - 1] = True
                buffer.storage["next"]["truncated"][buffer._idx - 1] = True
                buffer._recompute_start_stop = True
            print(f"Loaded buffer of size {len(buffer)}")
        else:
            if self.cfg.use_trajectory_buffer:
                output_key_t = ["observation", "action", "z", "next", "step_count", "reward"]
                # TODO this interface should be more elegant (how to inform buffer what keys are coming in / need to be sampled?)
                if isinstance(self.cfg.agent, (FBcprAuxAgentConfig)):
                    output_key_t.append("aux_rewards")

                replay_buffer["train"] = TrajectoryDictBufferMultiDim(
                    capacity=self.cfg.buffer_size // self.cfg.online_parallel_envs,  # make sure to divide by num_envs
                    device=self.cfg.buffer_device,
                    n_dim=2,
                    end_key="done",
                    output_key_t=output_key_t,  # TODO(team): fix this. in principle we could avoid to sample qpos, qvel for training but we need them for reward evaluation
                    output_key_tp1=[],
                )
            else:
                replay_buffer["train"] = DictBuffer(capacity=self.cfg.buffer_size, device=self.cfg.buffer_device)
        if self.training_with_expert_data:
            replay_buffer["expert_slicer"] = expert_buffer

        print("Starting training")
        progb = tqdm(total=self.cfg.num_env_steps, disable=self.cfg.disable_tqdm)
        td, info = train_env.reset()
        total_metrics, context = None, None
        start_time = time.time()
        fps_start_time = time.time()
        checkpoint_time_checker = EveryNStepsChecker(self._checkpoint_time, self.cfg.checkpoint_every_steps)
        eval_time_checker = EveryNStepsChecker(self._checkpoint_time, self.cfg.eval_every_steps)
        update_agent_time_checker = EveryNStepsChecker(self._checkpoint_time, self.cfg.update_agent_every)
        log_time_checker = EveryNStepsChecker(self._checkpoint_time, self.cfg.log_every_updates)

        eval_instances = []
        for evaluation_name in self.evaluations.keys():
            evaluation = self.evaluations[evaluation_name]
            eval_instances.append(isinstance(evaluation, HumanoidVerseIsaacTrackingEvaluation))
        uses_humanoidverse_eval = True if any(eval_instances) else False

        completed_steps = self._checkpoint_time
        for t in range(self._checkpoint_time, self.cfg.num_env_steps, self.cfg.online_parallel_envs):
            if (t != self._checkpoint_time) and checkpoint_time_checker.check(t):
                checkpoint_time_checker.update_last_step(t)
                self.save(t, replay_buffer)

            if (self.evaluate and eval_time_checker.check(t)) or (self.evaluate and t == self._checkpoint_time):
                eval_metrics = self.eval(t, replay_buffer=replay_buffer)
                eval_time_checker.update_last_step(t)
                if uses_humanoidverse_eval:
                    # reset if there is a humanoidverse evaluation
                    td, info = train_env.reset()

                if self.cfg.prioritization:
                    assert len(eval_metrics[self.priorization_eval_name]) == len(replay_buffer["expert_slicer"].motion_ids), (
                        "Mismatch in number of motions returned by the eval"
                    )
                    # priorities
                    index_in_buffer, name_in_buffer = {}, {}
                    for i, motion_id in enumerate(replay_buffer["expert_slicer"].motion_ids):
                        index_in_buffer[motion_id] = i
                        if hasattr(replay_buffer["expert_slicer"], "file_names"):
                            name_in_buffer[motion_id] = replay_buffer["expert_slicer"].file_names[i]
                    motions_id, priorities, idxs = [], [], []
                    for _, metr in eval_metrics[self.priorization_eval_name].items():
                        motions_id.append(metr["motion_id"])
                        priorities.append(metr["emd"])
                        idxs.append(index_in_buffer[metr["motion_id"]])
                    priorities = (
                        torch.clamp(
                            torch.tensor(priorities, dtype=torch.float32, device=self.agent.device),
                            min=self.cfg.prioritization_min_val,
                            max=self.cfg.prioritization_max_val,
                        )
                        * self.cfg.prioritization_scale
                    )

                    if self.cfg.prioritization_mode == "lin":
                        pass
                    elif self.cfg.prioritization_mode == "exp":
                        priorities = 2**priorities
                    elif self.cfg.prioritization_mode == "bin":
                        bins = torch.floor(priorities)
                        for i in range(int(bins.min().item()), int(bins.max().item()) + 1):
                            mask = bins == i
                            n = mask.sum().item()
                            if n > 0:
                                priorities[mask] = 1 / n
                    else:
                        raise ValueError(f"Unsupported prioritization mode {self.cfg.prioritization_mode}")

                    train_env._env._motion_lib.update_sampling_weight_by_id(
                        priorities=list(priorities), motions_id=idxs, file_name=name_in_buffer
                    )

                    replay_buffer["expert_slicer"].update_priorities(
                        priorities=priorities.to(self.cfg.buffer_device), idxs=torch.tensor(np.array(idxs), device=self.cfg.buffer_device)
                    )

            with torch.no_grad():
                obs = tree_map(lambda x: torch.tensor(x, dtype=dtype_numpytotorch_lower_precision(x.dtype), device=self.agent.device), td)
                # TODO consistency with obs_space: remove time assigned by TimeAwareObservationWrapper
                step_count = obs.pop("time")

                history_context = None
                if "history" in obs:
                    # this works in inference mode
                    if len(obs["history"]["action"]) == 0:
                        history_context = self.agent._model._context_encoder.get_initial_context(self.cfg.online_parallel_envs)
                    else:
                        history_context = self.agent.history_inference(obs=obs["history"]["observation"], action=obs["history"]["action"])[
                            :, -1
                        ].clone()

                context = self.agent.maybe_update_rollout_context(z=context, step_count=step_count, replay_buffer=replay_buffer)
                if t < self.cfg.num_seed_steps:
                    action = train_env.action_space.sample().astype(np.float32)
                else:
                    # this works in inference mode
                    if history_context is not None:
                        action = self.agent.act(obs=obs, z=context, context=history_context, mean=False)
                    else:
                        action = self.agent.act(obs=obs, z=context, mean=False)
                    # TODO a bit hard-coded -- just to avoid moving stuff from cpu to cuda
                    if not isinstance(self.cfg.env, HumanoidVerseIsaacConfig):
                        action = action.cpu().detach().numpy()
            new_td, new_reward, new_terminated, new_truncated, new_info = train_env.step(action)

            # we check if at the next iteration we will evaluate
            next_t = t + self.cfg.online_parallel_envs
            if (self.evaluate and eval_time_checker.check(next_t)) or (self.evaluate and next_t == self._checkpoint_time):
                if isinstance(self.cfg.env, HumanoidVerseIsaacConfig) and uses_humanoidverse_eval:
                    # make sure we set truncated since at the next iteration we are forced to reset the environment
                    # after the evaluation. This is because we share the environment with the evaluation
                    new_truncated = np.ones_like(new_truncated, dtype=bool)

            data = make_transition(obs, action, new_td, new_reward, new_terminated, new_truncated, info, new_info, step_count)
            if context is not None:
                data["z"] = context
            if history_context is not None:
                data["history_context"] = history_context
            if self.cfg.use_trajectory_buffer:
                data = tree_map(lambda value: value[None, ...], data)
            replay_buffer["train"].extend(data)

            if len(replay_buffer["train"]) > 0 and t > self.cfg.num_seed_steps and update_agent_time_checker.check(t):
                update_agent_time_checker.update_last_step(t)
                for _ in range(self.cfg.num_agent_updates):
                    metrics = self.agent.update(replay_buffer, t)
                    if total_metrics is None:
                        num_metrics_updates = 1
                        total_metrics = {k: metrics[k].float().clone() for k in metrics.keys()}
                    else:
                        num_metrics_updates += 1
                        total_metrics = {k: total_metrics[k] + metrics[k].float() for k in metrics.keys()}

            if log_time_checker.check(t) and total_metrics is not None:
                log_time_checker.update_last_step(t)
                m_dict = {}
                for k in sorted(list(total_metrics.keys())):
                    tmp = total_metrics[k] / num_metrics_updates
                    m_dict[k] = np.round(tmp.mean().item(), 6)
                m_dict["duration [minutes]"] = (time.time() - start_time) / 60
                m_dict["FPS"] = (1 if t == 0 else self.cfg.log_every_updates) / (time.time() - fps_start_time)
                if self.cfg.use_wandb:
                    wandb.log(
                        {f"train/{k}": v for k, v in m_dict.items()},
                        step=t,
                    )
                print(m_dict)
                total_metrics = None
                fps_start_time = time.time()
                m_dict["timestep"] = t
                self.train_logger.log(m_dict)

            progb.update(self.cfg.online_parallel_envs)
            completed_steps = t + self.cfg.online_parallel_envs
            td = new_td
            info = new_info
        if completed_steps > self._checkpoint_time:
            self.save(completed_steps, replay_buffer)
            if self.evaluate and eval_time_checker.check(completed_steps):
                self.eval(completed_steps, replay_buffer=replay_buffer)
        train_env.close()

    def eval(self, t, replay_buffer):
        print(f"Starting evaluation at time {t}")
        evaluation_results = {}

        # This will contain the results, mapping evaluation.cfg.name --> dict of metrics
        evaluation_results = {}
        for evaluation_name in self.evaluations.keys():
            logger = self.eval_loggers[evaluation_name]
            evaluation = self.evaluations[evaluation_name]

            # NOTE we have this inside the loop so that the agent is not moved to cpu if we don't evaluate
            if not isinstance(self.cfg.env, HumanoidVerseIsaacConfig):
                self.agent._model.to("cpu")
            self.agent._model.train(False)

            if isinstance(self.cfg.env, HumanoidVerseIsaacConfig):
                # Pass train env
                evaluation_metrics, wandb_dict = evaluation.run(
                    timestep=t, agent_or_model=self.agent, replay_buffer=replay_buffer, logger=logger, env=self.train_env
                )
            else:
                evaluation_metrics, wandb_dict = evaluation.run(
                    timestep=t,
                    agent_or_model=self.agent,
                    replay_buffer=replay_buffer,
                    logger=logger,
                )
            # For wandb dict, put it on wandb
            if self.cfg.use_wandb and wandb_dict is not None:
                wandb.log(
                    {f"eval/{evaluation_name}/{k}": v for k, v in wandb_dict.items()},
                    step=t,
                )

            evaluation_results[evaluation_name] = evaluation_metrics

        # ---------------------------------------------------------------
        # this is important, move back the agent to cuda and
        # restart the training
        if not isinstance(self.cfg.env, HumanoidVerseIsaacConfig):
            self.agent._model.to(self.cfg.agent.model.device)
        self.agent._model.train()

        return evaluation_results

    def save(self, time: int, replay_buffer: Dict[str, tp.Any]) -> None:
        print(f"Checkpointing at time {time}")
        checkpoint_dir = self.work_dir / CHECKPOINT_DIR_NAME
        previous = self.work_dir / (CHECKPOINT_DIR_NAME + ".previous")
        if not checkpoint_dir.exists() and previous.exists():
            previous.replace(checkpoint_dir)
        with tempfile.TemporaryDirectory(prefix=".checkpoint-", dir=self.work_dir) as folder:
            staging = Path(folder)
            self.agent.save(str(staging))
            if self.cfg.checkpoint_buffer:
                replay_buffer["train"].save(staging / "buffers" / "train")
            with (staging / "train_status.json").open("w") as f:
                json.dump({"time": time, "replay_format": "explicit_successor_v1"}, f, indent=4)
            if previous.exists():
                shutil.rmtree(previous)
            if checkpoint_dir.exists():
                checkpoint_dir.replace(previous)
            try:
                staging.replace(checkpoint_dir)
            except BaseException:
                if previous.exists():
                    previous.replace(checkpoint_dir)
                raise
        if previous.exists():
            shutil.rmtree(previous)


def default_train_config() -> TrainConfig:
    from humanoidverse.agents.fb_cpr_aux.model import FBcprAuxModelArchiConfig, FBcprAuxModelConfig
    from humanoidverse.agents.fb_cpr_aux.agent import FBcprAuxAgentTrainConfig
    from humanoidverse.agents.nn_models import ForwardArchiConfig, BackwardArchiConfig, ActorArchiConfig, DiscriminatorArchiConfig, RewardNormalizerConfig
    from humanoidverse.agents.normalizers import ObsNormalizerConfig, BatchNormNormalizerConfig
    from humanoidverse.agents.nn_filters import DictInputFilterConfig

    cfg = TrainConfig(
        name='TrainConfig',
        agent=FBcprAuxAgentConfig(
            name='FBcprAuxAgent',
            model=FBcprAuxModelConfig(
                name='FBcprAuxModel',
                device='cuda',
                archi=FBcprAuxModelArchiConfig(
                    name='FBcprAuxModelArchiConfig',
                    z_dim=256,
                    norm_z=True,
                    f=ForwardArchiConfig(name='ForwardArchi', hidden_dim=2048, model='residual', hidden_layers=6, embedding_layers=2, num_parallel=2, ensemble_mode='batch', input_filter=DictInputFilterConfig(name='DictInputFilterConfig', key=['state', 'privileged_state', 'last_action', 'history_actor'])),
                    b=BackwardArchiConfig(name='BackwardArchi', hidden_dim=256, hidden_layers=1, norm=True, input_filter=DictInputFilterConfig(name='DictInputFilterConfig', key=['state', 'privileged_state'])),
                    actor=ActorArchiConfig(name='actor', model='residual', hidden_dim=2048, hidden_layers=6, embedding_layers=2, input_filter=DictInputFilterConfig(name='DictInputFilterConfig', key=['state', 'last_action', 'history_actor'])),
                    critic=ForwardArchiConfig(name='ForwardArchi', hidden_dim=2048, model='residual', hidden_layers=6, embedding_layers=2, num_parallel=2, ensemble_mode='batch', input_filter=DictInputFilterConfig(name='DictInputFilterConfig', key=['state', 'privileged_state', 'last_action', 'history_actor'])),
                    discriminator=DiscriminatorArchiConfig(name='DiscriminatorArchi', hidden_dim=1024, hidden_layers=3, input_filter=DictInputFilterConfig(name='DictInputFilterConfig', key=['state', 'privileged_state'])),
                    aux_critic=ForwardArchiConfig(name='ForwardArchi', hidden_dim=2048, model='residual', hidden_layers=6, embedding_layers=2, num_parallel=2, ensemble_mode='batch', input_filter=DictInputFilterConfig(name='DictInputFilterConfig', key=['state', 'privileged_state', 'last_action', 'history_actor']))
                ),
                obs_normalizer=ObsNormalizerConfig(
                    name='ObsNormalizerConfig',
                    normalizers={
                        'state': BatchNormNormalizerConfig(name='BatchNormNormalizerConfig', momentum=0.01),
                        'privileged_state': BatchNormNormalizerConfig(name='BatchNormNormalizerConfig', momentum=0.01),
                        'last_action': BatchNormNormalizerConfig(name='BatchNormNormalizerConfig', momentum=0.01),
                        'history_actor': BatchNormNormalizerConfig(name='BatchNormNormalizerConfig', momentum=0.01)
                    },
                    allow_mismatching_keys=True
                ),
                inference_batch_size=500000,
                seq_length=8,
                actor_std=0.05,
                amp=False,
                norm_aux_reward=RewardNormalizerConfig(name='RewardNormalizer', translate=False, scale=True)
            ),
            train=FBcprAuxAgentTrainConfig(
                name='FBcprAuxAgentTrainConfig',
                lr_f=0.0003,
                lr_b=1e-05,
                lr_actor=0.0003,
                weight_decay=0.0,
                clip_grad_norm=0.0,
                fb_target_tau=0.01,
                ortho_coef=100.0,
                train_goal_ratio=0.2,
                fb_pessimism_penalty=0.0,
                actor_pessimism_penalty=0.5,
                stddev_clip=0.3,
                q_loss_coef=0.0,
                batch_size=1024,
                discount=0.98,
                use_mix_rollout=True,
                update_z_every_step=100,
                z_buffer_size=8192,
                rollout_expert_trajectories=True,
                rollout_expert_trajectories_length=250,
                rollout_expert_trajectories_percentage=0.5,
                lr_discriminator=1e-05,
                lr_critic=0.0003,
                critic_target_tau=0.005,
                critic_pessimism_penalty=0.5,
                reg_coeff=0.05,
                scale_reg=True,
                expert_asm_ratio=0.6,
                relabel_ratio=0.8,
                grad_penalty_discriminator=10.0,
                weight_decay_discriminator=0.0,
                lr_aux_critic=0.0003,
                reg_coeff_aux=0.02,
                aux_critic_pessimism_penalty=0.5
            ),
            aux_rewards=['penalty_torques', 'penalty_action_rate', 'limits_dof_pos', 'limits_torque', 'penalty_undesired_contact', 'penalty_feet_ori', 'penalty_ankle_roll', 'penalty_slippage'],
            aux_rewards_scaling={'penalty_action_rate': -0.1, 'penalty_feet_ori': -0.4, 'penalty_ankle_roll': -4.0, 'limits_dof_pos': -10.0, 'penalty_slippage': -2.0, 'penalty_undesired_contact': -1.0, 'penalty_torques': 0.0, 'limits_torque': 0.0},
            cudagraphs=False,
            compile=True
        ),
        motions='',
        motions_root='',
        env=HumanoidVerseIsaacConfig(
            name='humanoidverse_isaac',
            device='cuda:0',
            # TODO this needs to be updated to point to a path with lafan dataset chunked into 10s clips
            lafan_tail_path='humanoidverse/data/lafan_29dof_10s-clipped.pkl',
            enable_cameras=False,
            camera_render_save_dir='isaac_videos',
            max_episode_length_s=None,
            disable_obs_noise=False,
            disable_domain_randomization=False,
            relative_config_path='exp/bfm_zero/bfm_zero',
            include_last_action=True,
            hydra_overrides=['robot=g1/g1_29dof_hard_waist', 'robot.control.action_scale=0.25', 'robot.control.action_clip_value=5.0', 'robot.control.normalize_action_to=5.0', 'env.config.lie_down_init=True', 'env.config.lie_down_init_prob=0.3'],
            context_length=None,
            include_dr_info=False,
            included_dr_obs_names=None,
            include_history_actor=True,
            include_history_noaction=False,
            make_config_g1env_compatible=False,
            root_height_obs=True
        ),
        work_dir='results/bfmzero-isaac',
        seed=4728,
        online_parallel_envs=1024,
        log_every_updates=384000,
        num_env_steps=384000000,
        update_agent_every=1024,
        num_seed_steps=10240,
        num_agent_updates=16,
        checkpoint_every_steps=9600000,
        checkpoint_buffer=True,
        prioritization=True,
        prioritization_min_val=0.5,
        prioritization_max_val=2.0,
        prioritization_scale=2.0,
        prioritization_mode='exp',
        use_trajectory_buffer=True,
        buffer_size=5120000,
        use_wandb=False,
        wandb_ename='yitangl',  # your wandb entity (username/team), empty = default from wandb login
        wandb_gname='bfmzero-isaac',  # run group
        wandb_pname='bfmzero-isaac',  # your wandb project name
        load_isaac_expert_data=True,
        buffer_device='cuda',
        disable_tqdm=True,
        evaluations=[HumanoidVerseIsaacTrackingEvaluationConfig(name='HumanoidVerseIsaacTrackingEvaluationConfig', generate_videos=False, videos_dir='videos', video_name_prefix='unknown_agent', name_in_logs='humanoidverse_tracking_eval', env=None, num_envs=1024, n_episodes_per_motion=1)],
        eval_every_steps=9600000,
        tags={},
    )
    return cfg


def train_bfm_zero():
    workspace = default_train_config().build()
    workspace.train()


if __name__ == "__main__":
    # This is the bare minimum CLI interface to launch experiments, but ideally you should
    # launch your experiments from Python code (e.g., see under "scripts")
    train_bfm_zero()

# uv run --no-cache -m humanoidverse.meta_online_entry_point
