# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import functools
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List

import gymnasium
import numpy as np
import ot
import torch
from packaging.version import Version
from torch.utils._pytree import tree_map
from tqdm import tqdm

from humenv import CustomManager
from humenv.bench.gym_utils.episodes import Episode
from humenv.misc.motionlib import MotionBuffer

from humanoidverse.utils.g1_env_config import G1EnvConfigsType
from humanoidverse.agents.evaluations.humanoidverse_isaac import compute_joint_pos_metrics

if Version("0.26") <= Version(gymnasium.__version__) < Version("1.0"):

    def cast_obs_wrapper(env) -> gymnasium.Wrapper:
        return gymnasium.wrappers.TransformObservation(env, lambda obs: obs.astype(np.float32))
else:

    def cast_obs_wrapper(env) -> gymnasium.Wrapper:
        return gymnasium.wrappers.TransformObservation(env, lambda obs: obs.astype(np.float32), env.observation_space)

def check_framestack_and_get_frames(env: gymnasium.Env):
    current_env = env
    while isinstance(current_env, gymnasium.Wrapper):
        if isinstance(current_env, gymnasium.wrappers.FrameStackObservation):
            # FrameStackObservation stores the number of frames in its 'num_stack' attribute
            return current_env.stack_size
        current_env = current_env.env  # Go to the wrapped environment

    return None

xpos_bodies = [
    "pelvis",
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
    "waist_yaw_link",
    "waist_roll_link",
    "torso_link",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    # "left_wrist_roll_link",
    # "left_wrist_pitch_link",
    # "left_wrist_yaw_link",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    # "right_wrist_roll_link",
    # "right_wrist_pitch_link",
    # "right_wrist_yaw_link",
]


@dataclasses.dataclass(kw_only=True)
class TrackingEvaluation:
    motions: str | List[str]
    motion_base_path: str | None = None
    # environment parameters
    num_envs: int = 1
    env_config: G1EnvConfigsType
    mp_context: str = "forkserver"

    def __post_init__(self) -> None:
        if self.num_envs > 1:
            self.mp_manager = CustomManager()
            self.mp_manager.start()
            self.motion_buffer = self.mp_manager.MotionBuffer(
                files=self.motions,
                base_path=self.motion_base_path,
                keys=["qpos", "qvel", "observation"],
            )
        else:
            self.mp_manager = None
            self.motion_buffer = MotionBuffer(
                files=self.motions,
                base_path=self.motion_base_path,
                keys=["qpos", "qvel", "observation"],
            )

    def run(self, agent: Any, disable_tqdm: bool = False) -> Dict[str, Any]:
        ids = self.motion_buffer.get_motion_ids()
        np.random.shuffle(ids)  # shuffle the ids to evenly distribute the motions, as different datasets have different motion length
        num_workers = min(self.num_envs, len(ids))
        motions_per_worker = np.array_split(ids, num_workers)
        f = functools.partial(
            _async_tracking_worker,
            env_config=self.env_config,
            motion_buffer=self.motion_buffer,
            disable_tqdm=disable_tqdm,
        )
        if num_workers == 1:
            metrics = f((motions_per_worker[0], 0, agent))
        else:
            prev_omp_num_th = os.environ.get("OMP_NUM_THREADS", None)
            os.environ["OMP_NUM_THREADS"] = "1"
            with ProcessPoolExecutor(
                max_workers=num_workers,
                mp_context=multiprocessing.get_context(self.mp_context),
            ) as pool:
                inputs = [(x, y, agent) for x, y in zip(motions_per_worker, range(len(motions_per_worker)))]
                list_res = pool.map(f, inputs)
                metrics = {}
                for el in list_res:
                    metrics.update(el)
            if prev_omp_num_th is None:
                del os.environ["OMP_NUM_THREADS"]
            else:
                os.environ["OMP_NUM_THREADS"] = prev_omp_num_th
        return metrics

    def close(self) -> None:
        if self.mp_manager is not None:
            self.mp_manager.shutdown()


def _async_tracking_worker(inputs, env_config: G1EnvConfigsType, motion_buffer: MotionBuffer, disable_tqdm: bool = False):
    motion_ids, pos, agent = inputs
    env, _ = env_config.build(num_envs=1)
    frame_stack = check_framestack_and_get_frames(env)

    metrics = {}
    for m_id in tqdm(motion_ids, position=pos, leave=False, disable=disable_tqdm):
        ep_ = motion_buffer.get(m_id)

        # we ignore the first state since we need to pass the next observation
        if frame_stack is not None:
            tracking_target = tree_map(lambda x: x[1:][..., None], ep_["observation"])
        else:
            tracking_target = tree_map(lambda x: x[1:], ep_["observation"])
        ctx = agent.tracking_inference(next_obs=tracking_target)
        ctx = [None] * (ep_["qpos"].shape[0] - 1) if ctx is None else ctx
        observation, info = env.reset(options={"qpos": ep_["qpos"][0], "qvel": ep_["qvel"][0]})
        xpos = [np.stack([env.unwrapped._mj_data.body(i).xpos for i in xpos_bodies]).ravel()]
        joint_pos = [info["qpos"].ravel()[7:]]
        _episode = Episode()
        # For calculating the metrics later, we need the same dim obs as expert data is in
        # (i.e., we need to remove "last_act" and "phase" data)
        # TODO this logic should be more consistent around code (separatio between what agent sees vs. what are meaningful data)
        _episode.initialise(observation, info)
        for i in range(len(ctx)):
            # be sure that there is a batch dimension for the observation
            obs = tree_map(lambda x: x[None, ...], observation)
            action = agent.act(obs, ctx[i]).ravel()
            observation, reward, terminated, truncated, info = env.step(action)
            xpos.append(np.stack([env.unwrapped._mj_data.body(i).xpos for i in xpos_bodies]).ravel())
            joint_pos.append(info["qpos"].ravel()[7:])
            # Same cutting of the obs here
            _episode.add(
                observation,
                reward,
                action,
                terminated,
                truncated,
                info,
            )
        tmp = _episode.get()
        # re-execute motion
        target_xpos = []
        for i in range(len(ctx)):
            env.reset(options={"qpos": ep_["qpos"][i], "qvel": ep_["qvel"][i]})
            target_xpos.append(np.stack([env.unwrapped._mj_data.body(i).xpos for i in xpos_bodies]).ravel())
        tmp["xpos"] = np.stack(xpos)[:-1]
        tmp["target_xpos"] = np.stack(target_xpos)
        tmp["tracking_target"] = tracking_target
        tmp["motion_id"] = m_id
        tmp["motion_file"] = motion_buffer.get_name(m_id)
        tmp["target_joint_pos"] = ep_["qpos"][1:, 7:]
        tmp["joint_pos"] = np.stack(joint_pos)[1:]
        tmp["dt"] = env.unwrapped.dt
        metrics.update(_calc_metrics(tmp, frame_stack))
    env.close()
    return metrics


def distance_matrix(X: torch.Tensor, Y: torch.Tensor):
    X_norm = X.pow(2).sum(1).reshape(-1, 1)
    Y_norm = Y.pow(2).sum(1).reshape(1, -1)
    val = X_norm + Y_norm - 2 * torch.matmul(X, Y.T)
    return torch.sqrt(torch.clamp(val, min=0))


def emd_numpy(next_obs: torch.Tensor, tracking_target: torch.Tensor):
    # keep only pose part of the observations
    agent_obs = next_obs.to("cpu")
    tracked_obs = tracking_target.to("cpu")
    # compute optimal transport cost
    cost_matrix = distance_matrix(agent_obs, tracked_obs).cpu().detach().numpy()
    X_pot = np.ones(agent_obs.shape[0]) / agent_obs.shape[0]
    Y_pot = np.ones(tracked_obs.shape[0]) / tracked_obs.shape[0]
    transport_cost = ot.emd2(X_pot, Y_pot, cost_matrix, numItermax=100000)
    return {"emd": transport_cost}


def distance_proximity(next_obs: torch.Tensor, tracking_target: torch.Tensor, bound: float = 2.0, margin: float = 2):
    stats = {}
    dist = torch.norm(next_obs - tracking_target, dim=-1)
    in_bounds_mask = dist <= bound
    out_bounds_mask = dist > bound + margin
    stats["proximity"] = (in_bounds_mask + ((bound + margin - dist) / margin) * (~in_bounds_mask) * (~out_bounds_mask)).mean()
    stats["distance"] = dist.mean()
    stats["success"] = in_bounds_mask.min().float()
    return stats


def _calc_metrics(ep, frame_stack: int | None):
    metr = {}
    # keep only qpos states
    # TODO hardcoded observation used for the tracking ("state")
    if isinstance(ep["observation"], dict):
        # Assume that interesting data is in "state" key
        next_obs = ep["observation"]["state"][1:]
        tracking_target = ep["tracking_target"]["state"]
    else:
        # If flat, assume this is the state data
        next_obs = ep["observation"][1:]
        tracking_target = ep["tracking_target"]
    if frame_stack is not None:
        # if frame stack is used, we take only the last frame of each observation
        next_obs = next_obs[..., -1]
        assert tracking_target.shape[-1] == 1, "Tracking target should have only one frame in the last dimension"
        tracking_target = tracking_target.squeeze(-1)

    joint_count = ep["joint_pos"].shape[-1]
    next_obs = torch.tensor(next_obs[:, :joint_count], dtype=torch.float32)
    tracking_target = torch.tensor(tracking_target[:, :joint_count], dtype=torch.float32)

    dist_prox_res = distance_proximity(next_obs=next_obs, tracking_target=tracking_target)
    metr.update({f"obs_state_{key}": value for key, value in dist_prox_res.items()})
    emd_res = emd_numpy(next_obs=next_obs, tracking_target=tracking_target)
    metr.update({f"obs_state_{key}": value for key, value in emd_res.items()})
    # add distance wrt xpos
    xpos = torch.tensor(ep["xpos"], dtype=torch.float32)
    target_xpos = torch.tensor(ep["target_xpos"], dtype=torch.float32)
    dist_prox_res = distance_proximity(next_obs=xpos, tracking_target=target_xpos)
    for k, v in dist_prox_res.items():
        metr[f"{k}_xpos"] = v
    metr["emd_xpos"] = emd_numpy(next_obs=xpos, tracking_target=target_xpos)["emd"]
    jpos_pred = xpos.reshape(xpos.shape[0], -1, 3)  # length x 23 x 3
    jpos_gt = target_xpos.reshape(target_xpos.shape[0], -1, 3)  # length x 23 x 3
    metr["success_phc_linf_xpos"] = torch.all(torch.norm(jpos_pred - jpos_gt, dim=-1) <= 0.5).float()
    metr["success_phc_mean_xpos"] = torch.all(torch.norm(jpos_pred - jpos_gt, dim=-1).mean(dim=-1) <= 0.5).float()
    metr.update(compute_joint_pos_metrics(
        torch.as_tensor(ep["joint_pos"], dtype=torch.float32),
        torch.as_tensor(ep["target_joint_pos"], dtype=torch.float32),
        dt=ep["dt"],
    ))
    for k, v in metr.items():
        if isinstance(v, torch.Tensor):
            metr[k] = v.tolist()
    metr["motion_id"] = ep["motion_id"]
    # metr["motion_file"] = ep["motion_file"]
    return {ep["motion_file"]: metr}
