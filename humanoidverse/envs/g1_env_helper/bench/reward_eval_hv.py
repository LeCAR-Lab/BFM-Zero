# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
from typing import Any, Callable, Dict, List, Sequence

import gymnasium
import torch

import mujoco
import numpy as np
from humenv.rewards import RewardFunction
from humanoidverse.envs.g1_env_helper.robot import make_from_name
from humanoidverse.agents.buffers.trajectory import TrajectoryDictBufferMultiDim
from humanoidverse.utils.g1_env_config import G1EnvConfigsType

@dataclasses.dataclass(kw_only=True)
class RewardEvaluationHV:
    tasks: List[str]
    num_episodes: int = 10
    env_config: G1EnvConfigsType

    def run(self, agent: Any, disable_tqdm: bool = True) -> Dict[str, Any]:
        # TODO make parallel
        env, mp_info = self.env_config.build(1)

        def reset_task(task):
            if not isinstance(env, (gymnasium.vector.AsyncVectorEnv, gymnasium.vector.SyncVectorEnv)):
                env.unwrapped.set_task(task)
            else:
                env.call("set_task", task)

        task_to_rewards = {task: {"reward": []} for task in self.tasks}
        for task in self.tasks:
            ctx = agent.reward_inference(task=task)
            reset_task(task)

            for _ in range(self.num_episodes):
                observation, info = env.reset()
                cumulative_reward = 0.0
                truncated = False
                terminated = False
                while not truncated and not terminated:
                    observation = torch.tensor(observation, dtype=torch.float32)[None]
                    action = agent.act(observation, ctx).ravel()
                    observation, reward, terminated, truncated, info = env.step(action)
                    cumulative_reward += reward

            task_to_rewards[task]["reward"].append(cumulative_reward)

        env.close()
        if mp_info is not None:
            mp_info["manager"].shutdown()

        return task_to_rewards

from humanoidverse.agents.wrappers.humenvbench import BaseHumEnvBenchWrapper, get_next
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import functools

@dataclasses.dataclass(kw_only=True)
class RewardWrapperHV(BaseHumEnvBenchWrapper):
    inference_dataset: Any
    num_samples_per_inference: int
    inference_function: str
    max_workers: int
    process_executor: bool = False
    process_context: str = "spawn"
    # env: Any = None
    # env_model: str = "humanoidverse/data/robots/g1/g1_29dof_anneal_23dof.xml"
    env_model: str = "humanoidverse/data/robots/g1/scene_flat_terrain_playground.xml"

    def reward_inference(self, task: str) -> torch.Tensor:
        if isinstance(self.env_model, str):
            self.env_model = mujoco.MjModel.from_xml_path(self.env_model)

        if isinstance(self.inference_dataset, TrajectoryDictBufferMultiDim) and "next" not in self.inference_dataset.output_key_t:
            if "qpos" not in self.inference_dataset.output_key_tp1:
                self.inference_dataset.output_key_tp1.append("qpos")
            if "qvel" not in self.inference_dataset.output_key_tp1:
                self.inference_dataset.output_key_tp1.append("qvel")

        # env, _ = self.make_env_fn(task=task, **kwargs)
        if self.num_samples_per_inference >= self.inference_dataset.size() and hasattr(self.inference_dataset, "get_full_buffer"):
            # TODO change from "hasattr" to some more sensible thing
            data = self.inference_dataset.get_full_buffer()
        else:
            data = self.inference_dataset.sample(self.num_samples_per_inference)
        
        def xyzw_to_wxyz(quats_xyzw):
            return torch.cat([quats_xyzw[:, 3:4], quats_xyzw[:, 0:3]], dim=-1)
        # import ipdb; ipdb.set_trace()
        qpos = get_next("qpos", data)
        qvel = get_next("qvel", data)
        # from humanoidverse.isaac_utils.rotations import my_quat_rotate
        # qvel[:, 3:6] = my_quat_rotate(qpos[:, 3:7], qvel[:, 3:6])  # convert xyzw to wxyz
        # qpos[:, 3:7] = xyzw_to_wxyz(qpos[:, 3:7])  # convert xyzw to wxyz
        
        action = data["action"]
        if isinstance(qpos, torch.Tensor):
            qpos = qpos.cpu().detach().numpy()
            qvel = qvel.cpu().detach().numpy()
            action = action.cpu().detach().numpy()
        rewards = relabel(
            # env,
            self.env_model,
            qpos,
            qvel,
            action,
            # env.unwrapped.task,
            make_from_name(task),
            max_workers=self.max_workers,
            process_executor=self.process_executor,
        )
        # env.close()

        td = {
            "reward": torch.tensor(rewards, dtype=torch.float32, device=self.device),
        }
        if "B" in data:
            td["B_vect"] = data["B"]
        else:
            td["next_obs"] = get_next("observation", data)
        inference_fn = getattr(self.model, self.inference_function, None)
        ctxs = inference_fn(**td).reshape(1, -1)
        return ctxs

    def __deepcopy__(self, memo):
        # Create a new instance of the same type as self
        import copy
        return type(self)(
            model=copy.deepcopy(self.model, memo),
            numpy_output=self.numpy_output,
            _dtype=copy.deepcopy(self._dtype),
            inference_dataset=copy.deepcopy(self.inference_dataset),
            num_samples_per_inference=self.num_samples_per_inference,
            inference_function=self.inference_function,
            max_workers=self.max_workers,
            process_executor=self.process_executor,
            process_context=self.process_context,
        )

    def __getstate__(self):
        # Return a dictionary containing the state of the object
        return {
            "model": self.model,
            "numpy_output": self.numpy_output,
            "_dtype": self._dtype,
            "inference_dataset": self.inference_dataset,
            "num_samples_per_inference": self.num_samples_per_inference,
            "inference_function": self.inference_function,
            "max_workers": self.max_workers,
            "process_executor": self.process_executor,
            "process_context": self.process_context,
        }

    def __setstate__(self, state):
        # Restore the state of the object from the given dictionary
        self.model = state["model"]
        self.numpy_output = state["numpy_output"]
        self._dtype = state["_dtype"]
        self.inference_dataset = state["inference_dataset"]
        self.num_samples_per_inference = state["num_samples_per_inference"]
        self.inference_function = state["inference_function"]
        self.max_workers = state["max_workers"]
        self.process_executor = state["process_executor"]
        self.process_context = state["process_context"]
        

def _relabel_worker(
    x,
    model: mujoco.MjModel,
    reward_fn: RewardFunction,
):
    qpos, qvel, action = x
    assert len(qpos.shape) > 1
    assert qvel.shape[0] == qpos.shape[0]
    assert qvel.shape[0] == action.shape[0]
    rewards = np.zeros((qpos.shape[0], 1))
    for i in range(qpos.shape[0]):
        rewards[i] = reward_fn(model, qpos[i], qvel[i], action[i])
    return rewards

def relabel(
    # env: Any,
    model: Any,
    qpos: np.ndarray,
    qvel: np.ndarray,
    action: np.ndarray,
    reward_fn: RewardFunction,
    max_workers: int = 5,
    process_executor: bool = False,
    process_context: str = "spawn",
):
    chunk_size = int(np.ceil(qpos.shape[0] / max_workers))
    args = [(qpos[i : i + chunk_size], qvel[i : i + chunk_size], action[i : i + chunk_size]) for i in range(0, qpos.shape[0], chunk_size)]
    if max_workers == 1:
        result = [_relabel_worker(args[0], model=model, reward_fn=reward_fn)]
    else:
        if process_executor:
            import multiprocessing

            with ProcessPoolExecutor(
                max_workers=max_workers,
                mp_context=multiprocessing.get_context(process_context),
            ) as exe:
                f = functools.partial(_relabel_worker, model=model, reward_fn=reward_fn)
                result = exe.map(f, args)
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as exe:
                f = functools.partial(_relabel_worker, model=model, reward_fn=reward_fn)
                result = exe.map(f, args)

    tmp = [r for r in result]
    return np.concatenate(tmp)
