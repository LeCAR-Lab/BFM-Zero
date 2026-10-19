
import collections
import functools
import numbers
import random
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any, Dict, Mapping
import dataclasses
from collections import defaultdict
import copy

import mujoco
import numpy as np
import ot
import torch
from torch.utils._pytree import tree_map
from tqdm import tqdm
from humanoidverse.utils.helpers import get_reference_proprioception

from ..buffers.trajectory import TrajectoryDictBufferMultiDim
from ..envs.humanoidverse_isaac import HumanoidVerseIsaacConfig, HumanoidVerseVectorEnv, IsaacRendererWithMuJoco
from .base import BaseEvalConfig, extract_model

def get_next(field: str, data: Any):
    if "next" in data and field in data["next"]:
        return data["next"][field]
    elif f"next_{field}" in data:
        return data[f"next_{field}"]
    else:
        raise ValueError(f"No next of {field} found in data.")


@dataclasses.dataclass
class Episode:
    storage: Dict | None = None

    def initialise(self, observation: np.ndarray | Dict[str, np.ndarray], info: Dict) -> None:
        if self.storage is None:
            self.storage = defaultdict(list)
        if isinstance(observation, Mapping):
            self.storage["observation"] = {k: [copy.deepcopy(v)] for k, v in observation.items()}
        else:
            self.storage["observation"].append(copy.deepcopy(observation))
        self.storage["info"] = {
            k: [copy.deepcopy(v)] for k, v in info.items() if not k.startswith("_") and k not in ["final_observation", "final_info"]
        }

    def add(
        self,
        observation: np.ndarray | Dict[str, np.ndarray],
        reward: np.ndarray,
        action: np.ndarray,
        terminated: np.ndarray,
        truncated: np.ndarray,
        info: Dict,
    ) -> None:
        if isinstance(observation, Mapping):
            for k, v in observation.items():
                self.storage["observation"][k].append(copy.deepcopy(v))
        else:
            self.storage["observation"].append(copy.deepcopy(observation))
        for k, v in info.items():
            if not k.startswith("_") and k not in ["final_observation", "final_info"]:
                if k in self.storage["info"]:
                    # this is a quick fix but not robust fix to the problem that certain environments may return keys that are different from the one in the first step
                    # We may want to find a different solution to deal with this problem
                    self.storage["info"][k].append(copy.deepcopy(v))
                else:
                    self.storage["info"][k] = [copy.deepcopy(v)]
        self.storage["reward"].append(copy.deepcopy(reward))
        self.storage["action"].append(copy.deepcopy(action))
        self.storage["terminated"].append(copy.deepcopy(terminated))
        self.storage["truncated"].append(copy.deepcopy(truncated))

    def get(self) -> Dict[str, np.ndarray]:
        output = {}
        for k, v in self.storage.items():
            if k in ["observation", "info"]:
                if isinstance(v, Mapping):
                    output[k] = {}
                    for k2, v2 in v.items():
                        output[k][k2] = np.array(v2)
                else:
                    output[k] = np.array(v)
            else:
                output[k] = np.array(v)
        return output

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

def get_backward_observation(env, motion_id, include_last_action, velocity_multiplier: float = 1.0) -> torch.Tensor:
    import numpy as np
    from humanoidverse.envs.legged_robot_motions.legged_robot_motions import (
        compute_humanoid_observations_max,
        compute_humanoid_observations_max_with_contact,
    )
    from humanoidverse.utils.torch_utils import quat_rotate_inverse

    motion_times = torch.arange(int(np.ceil((env._motion_lib._motion_lengths[motion_id] / env.dt).cpu()))).to(env.device) * env.dt
    # motion_times = torch.arange(0, env._motion_lib._motion_num_frames.item() * frame_interval, frame_interval, device=env.device)

    # get blend motion state
    motion_state = env._motion_lib.get_motion_state(motion_id, motion_times)

    ref_body_pos = motion_state["rg_pos_t"]
    ref_body_rots = motion_state["rg_rot_t"]
    ref_body_vels = motion_state["body_vel_t"] * velocity_multiplier
    ref_body_angular_vels = motion_state["body_ang_vel_t"] * velocity_multiplier
    ref_dof_pos = motion_state["dof_pos"] - env.default_dof_pos[0]
    ref_dof_vel = motion_state["dof_vel"] * velocity_multiplier

    # npz_path = "dataprocess/push_door-3steps-no_door_amass-robot_only-processed/motion.npz"
    # data_npz = np.load(npz_path)
    # data_dict = {key: data_npz[key] for key in data_npz.files}

    def process_body_data(data_dict, device="cuda"):
        def extract_and_concat_vec3(arr: torch.Tensor) -> torch.Tensor:
            arr = arr.float()
            # arr: (T, N, 3)
            return torch.cat(
                [
                    arr[:, 0:13],
                    torch.zeros_like(arr[:, 14:16]),  # 两个补 0
                    arr[:, 13:18],
                    arr[:, 21:25],
                    arr[:, 20:21],
                    arr[:, 27:28],
                    arr[:, 12:13],
                ],
                dim=1,
            )

        def extract_and_concat_quat_torch(arr: torch.Tensor) -> torch.Tensor:
            arr = arr.float()
            # arr: (T, N, 4), wxyz
            seg1 = arr[:, 0:13]
            seg2 = torch.tensor([1, 0, 0, 0], dtype=arr.dtype, device=arr.device).view(1, 1, 4).repeat(arr.shape[0], 2, 1)
            seg3 = arr[:, 13:18]
            seg4 = arr[:, 21:25]
            seg7 = arr[:, 12:13]
            seg5 = arr[:, 20:21]
            seg6 = arr[:, 27:28]
            quat_wxyz = torch.cat([seg1, seg2, seg3, seg4, seg5, seg6, seg7], dim=1)  # (T, 24, 4)
            quat_xyzw = torch.cat([quat_wxyz[..., 1:], quat_wxyz[..., :1]], dim=-1)  # (T, 24, 4)
            return quat_xyzw

        # 转为 torch tensor 并送到 GPU
        body_pos = torch.from_numpy(data_dict["body_pos_w"]).to(device)
        body_quat = torch.from_numpy(data_dict["body_quat_w"]).to(device)
        body_lin_vel = torch.from_numpy(data_dict["body_lin_vel_w"]).to(device)
        body_ang_vel = torch.from_numpy(data_dict["body_ang_vel_w"]).to(device)
        joint_pos = torch.from_numpy(data_dict["joint_pos"]).to(device)[:, 1:]
        joint_vel = torch.from_numpy(data_dict["joint_vel"]).to(device)[:, 1:]

        # 处理
        ref_body_pos = extract_and_concat_vec3(body_pos)
        ref_body_vels = extract_and_concat_vec3(body_lin_vel)
        ref_body_angular_vels = extract_and_concat_vec3(body_ang_vel)
        ref_body_rots = extract_and_concat_quat_torch(body_quat)
        ref_dof_pos = joint_pos.float() - env.default_dof_pos[0]
        ref_dof_vel = joint_vel.float()

        return ref_body_pos, ref_body_rots, ref_body_vels, ref_body_angular_vels, ref_dof_pos, ref_dof_vel

    # import ipdb; ipdb.set_trace()
    # ref_body_pos, ref_body_rots, ref_body_vels, ref_body_angular_vels, ref_dof_pos, ref_dof_vel = process_body_data(data_dict)

    # construct observation
    if env.use_contact_in_obs_max:
        contact_binary = env.foot_contact_detect(ref_body_pos, ref_body_vels)
        obs_dict = compute_humanoid_observations_max_with_contact(
            ref_body_pos,
            ref_body_rots,
            ref_body_vels,
            ref_body_angular_vels,
            local_root_obs=True,
            root_height_obs=env.config.obs.root_height_obs,
            contact_binary=contact_binary,
        )
    else:
        obs_dict = compute_humanoid_observations_max(
            ref_body_pos,
            ref_body_rots,
            ref_body_vels,
            ref_body_angular_vels,
            local_root_obs=True,
            root_height_obs=env.config.obs.root_height_obs,
        )
    max_local_self_obs = torch.cat([v for v in obs_dict.values()], dim=-1)

    if env.config.obs.use_obs_filter:
        base_quat = ref_body_rots[:, 0]  # root orientation
        # ref_dof_pos = motion_state["dof_pos"] - env.default_dof_pos[0]
        # ref_dof_vel = motion_state["dof_vel"]
        ref_ang_vel = ref_body_angular_vels[:, 0]
        projected_gravity = quat_rotate_inverse(base_quat, env.gravity_vec[0:1].repeat(max_local_self_obs.shape[0], 1), w_last=True)
        bogus_actions = ref_dof_pos

        bogus_history_actor = torch.cat([bogus_actions, ref_ang_vel, ref_dof_pos, ref_dof_vel, projected_gravity], dim=-1).repeat(1, 4)
        # obs = torch.cat([bogus_actions, ref_ang_vel, ref_dof_pos, ref_dof_vel, bogus_history_actor, max_local_self_obs, projected_gravity], dim=-1)
        ref_dict = {
            "actions": bogus_actions,
            "ref_ang_vel": ref_ang_vel,
            "ref_dof_pos": ref_dof_pos,
            "dof_pos": motion_state["dof_pos"],
            "ref_dof_vel": ref_dof_vel,
            "dof_vel": motion_state["dof_vel"],
            "fake_history": bogus_history_actor,
            "max_local_self_obs": max_local_self_obs,
            "projected_gravity": projected_gravity,
            "ref_body_pos": ref_body_pos,
            "ref_body_rots": ref_body_rots,
            "ref_body_vels": ref_body_vels,
            "ref_body_angular_vels": ref_body_angular_vels,
            "root_pos": motion_state["root_pos"],
            "root_rot": motion_state["root_rot"],
            "root_vel": motion_state["root_vel"],
            "root_ang_vel": motion_state["root_ang_vel"],
        }
    else:
        # obs = max_local_self_obs
        ref_dict = {
            "max_local_self_obs": max_local_self_obs,
        }

    proprioception = get_reference_proprioception(
        env, motion_state["dof_pos"], ref_dof_vel, ref_body_rots[:, 0], ref_body_angular_vels[:, 0]
    )
    g1env_state = torch.cat(list(proprioception.values()), dim=-1)
    bogus_actions = ref_dof_pos
    g1env_last_action = bogus_actions

    g1env_privileged_obs = ref_dict["max_local_self_obs"]

    g1env_obs = {
        "state": g1env_state,
        "privileged_state": g1env_privileged_obs,
    }

    if include_last_action:
        # NOTE we multiply by zero to align with mujoco data
        g1env_obs["last_action"] = g1env_last_action * 0

    return g1env_obs, ref_dict


class HumanoidVerseIsaacTrackingEvaluationConfig(BaseEvalConfig):
    name_in_logs: str = "humanoidverse_tracking_eval"
    env: HumanoidVerseIsaacConfig | None = None
    # Used if above env is provided
    num_envs: int = 1024

    n_episodes_per_motion: int = 1
    include_results_from_all_envs: bool = False  # If True, include results from all num_envs envs, not just first num_motions

    disable_tqdm: bool = True

    def build(self):
        return HumanoidVerseIsaacTrackingEvaluation(self)


class HumanoidVerseIsaacTrackingEvaluation:
    def __init__(self, config: HumanoidVerseIsaacTrackingEvaluationConfig):
        self.cfg = config

    def run(self, *, timestep, agent_or_model, logger, env: HumanoidVerseVectorEnv | None = None, **kwargs) -> Dict[str, Any]:
        if env is None:
            if self.cfg.env is None:
                raise ValueError("Either env or cfg.env must be provided")
            env, _ = self.cfg.env.build(num_envs=self.cfg.num_envs)
        else:
            if self.cfg.env is not None:
                raise ValueError("Both env and cfg.env are provided, please provide only one (which one you want to evaluate with?)")
        # NOTE: this is not used as it disables some domain randomization we want to keep for evaluation
        # env._env.set_is_evaluating()

        self.motion_ids = list(range(env._env._motion_lib._num_unique_motions))
        n_envs = env._env.num_envs

        # "Parallelization" is handled on the worker level.
        metrics = {}
        for repetition_i in range(self.cfg.n_episodes_per_motion):
            run_metrics = {}
            # If we have less envs than motions, we run iteratively over the motions
            for motion_id_chunk_start in range(0, len(self.motion_ids), n_envs):
                motion_id_chunk = self.motion_ids[motion_id_chunk_start : motion_id_chunk_start + n_envs]
                motion_chunk_results = _async_tracking_worker(
                    (motion_id_chunk, 0, agent_or_model),
                    env=env,
                    disable_tqdm=self.cfg.disable_tqdm,
                    include_results_from_all_envs=self.cfg.include_results_from_all_envs,
                )
                for k, v in motion_chunk_results.items():
                    assert k not in run_metrics, "Tried to override existing metric"
                    run_metrics[k] = v

            if self.cfg.n_episodes_per_motion == 1:
                # Just return metrics, do not change names
                metrics = run_metrics
            else:
                # Append run number to metric names
                for k, v in run_metrics.items():
                    metrics[f"{k}_repetition#{repetition_i}"] = v

        aggregate = collections.defaultdict(list)
        wandb_dict = {}
        for _, metr in metrics.items():
            for k, v in metr.items():
                if isinstance(v, numbers.Number):
                    aggregate[k].append(v)
        for k, v in aggregate.items():
            wandb_dict[k] = np.mean(v)
            wandb_dict[f"{k}#std"] = np.std(v)

        if logger is not None:
            for k, v in metrics.items():
                v["motion_name"] = k
                v["timestep"] = timestep
                logger.log(v)

        # Resume back to original state of the motion lib if we were using shared env
        if self.cfg.env is None:
            env._env._motion_lib.load_motions_for_training()
        env._env.set_is_training()

        return metrics, wandb_dict

    def close(self) -> None:
        if self.mp_manager is not None:
            self.mp_manager.shutdown()


def group_assign_motions_to_envs_with_map(motion_ids, num_envs, device=None):
    motion_ids = torch.tensor(motion_ids, device=device)
    num_motions = len(motion_ids)

    env_idxs = list(range(num_envs))
    # Shuffle the environment indices to assign random motion to each env.
    # This is to apply domain randomization on tracking at random: isaacsim has fixed mass/friction
    # randomization per env, so assigning same motion to same env would mean we always evaluate the motion
    # with same mass/friction settings.
    random.shuffle(env_idxs)
    shuffled_env_idxs = torch.tensor(env_idxs, device=device, dtype=torch.long)

    assigned = motion_ids[shuffled_env_idxs % num_motions]

    motion_to_envs = {}
    for env_id, motion_id in enumerate(assigned.tolist()):
        motion_to_envs.setdefault(motion_id, []).append(env_id)

    return assigned, motion_to_envs


def _async_tracking_worker(
    inputs, env: HumanoidVerseVectorEnv, disable_tqdm: bool = False, include_results_from_all_envs=False
):
    motion_ids, pos, agent = inputs
    model = extract_model(agent)
    # import ipdb; ipdb.set_trace()
    # motion_ids = torch.tensor(motion_ids, device=env.device)
    isaac_env = env._env

    if not isaac_env._motion_lib.all_motions_loaded:
        isaac_env._motion_lib.all_motions_loaded = True
        isaac_env._motion_lib.load_motions(random_sample=False, num_motions_to_load=isaac_env._motion_lib._num_unique_motions, start_idx=0)

    metrics = {}
    num_envs = isaac_env.num_envs

    # Assign motions to envs in grouped fashion
    assigned_motions, motion_to_envs = group_assign_motions_to_envs_with_map(motion_ids, num_envs)

    ctx_dict = {}
    tracking_targets = {}
    # target_xpos_dict = {}
    tracking_joint_pos = {}
    dof_states_list = [None] * num_envs
    root_states_list = [None] * num_envs

    # Precompute context and target states per motion
    for m_id in motion_ids:
        tracking_target, tracking_target_dict = get_backward_observation(env._env, m_id, include_last_action=env.include_last_action)
        # first z should try to reach the next state, ie 1:
        z = model.backward_map(tree_map(lambda x: x[1:], tracking_target)).clone()
        for step in range(z.shape[0]):
            end_idx = min(step + 1, z.shape[0])
            z[step] = z[step:end_idx].mean(dim=0)
        ctx = model.project_z(z)
        ctx_dict[m_id] = ctx
        tracking_targets[m_id] = tree_map(lambda x: x.cpu(), tracking_target)
        tracking_joint_pos[m_id] = tracking_target_dict["dof_pos"].clone()
        # import ipdb; ipdb.set_trace()

        ref_body_rots = tracking_target_dict["ref_body_rots"][0, 0]
        if env._env.simulator.__class__.__name__ == "IsaacSim":
            # Avoid importing isaacsim as that breaks unless isaaclab is initialized
            ref_body_rots = ref_body_rots[[3, 0, 1, 2]]

        ref_root_init_state = torch.cat(
            [
                tracking_target_dict["ref_body_pos"][0, 0],
                ref_body_rots,
                tracking_target_dict["ref_body_vels"][0, 0],
                tracking_target_dict["ref_body_angular_vels"][0, 0],
            ]
        )
        # import ipdb; ipdb.set_trace()
        for env_id in motion_to_envs[m_id]:
            dof_init_state = torch.zeros_like(isaac_env.simulator.dof_state.view(num_envs, -1, 2)[0])
            dof_init_state[..., 0] = tracking_target_dict["dof_pos"][0]
            dof_init_state[..., 1] = tracking_target_dict["ref_dof_vel"][0]
            dof_states_list[env_id] = dof_init_state
            root_states_list[env_id] = ref_root_init_state
            # target_xpos_dict[env_id] = tracking_target_dict["ref_body_pos"][:, : len(xpos_bodies)]

    # this is for environments that are not initialized
    for i in range(num_envs):
        if dof_states_list[i] is None:
            dof_states_list[i] = torch.zeros_like(isaac_env.simulator.dof_state.view(num_envs, -1, 2)[0])
        if root_states_list[i] is None:
            root_states_list[i] = torch.zeros_like(root_states_list[0])

    target_states = {"dof_states": torch.stack(dof_states_list), "root_states": torch.stack(root_states_list)}

    env_ids = list(range(num_envs))
    env_ids = torch.tensor(env_ids, dtype=torch.long, device=isaac_env.device)

    with torch.no_grad():
        observation, info = env.reset(target_states=target_states, to_numpy=False)
        assert torch.allclose(env._env.simulator.dof_pos.clone()[env_ids], target_states["dof_states"][..., 0])

        if env.context_length:
            env._update_history(torch.arange(num_envs, device=isaac_env.device), observation, None)
            observation["history"] = {"observation": [], "action": []}

        # info = {}
        obs_cpu = tree_map(lambda x: x.cpu() if torch.is_tensor(x) else x, observation)
        info_cpu = tree_map(lambda x: x.cpu() if torch.is_tensor(x) else x, info)

        _episode = Episode()
        ooo = {k: v for k, v in obs_cpu.items() if k != "history"}
        _episode.initialise(ooo, info_cpu)

        xpos_log = [isaac_env.simulator._rigid_body_pos.reshape(num_envs, -1, 3)]
        joint_pos = [isaac_env.simulator.dof_state[..., 0]]
        joint_vel = [isaac_env.simulator.dof_state[..., 1]]

        max_ctx_len = max([ctx.shape[0] for ctx in ctx_dict.values()])
        for step in tqdm(range(max_ctx_len), desc="Tracking Evaluation", disable=disable_tqdm):
            ctx_batch = []
            for env_id in range(num_envs):
                m_id = assigned_motions[env_id].item()
                ctx = ctx_dict[m_id]
                ctx_t = ctx[step % ctx.shape[0]]
                ctx_batch.append(ctx_t)
            ctx_batch = torch.stack(ctx_batch)
            action = agent.act(observation, ctx_batch, mean=True)
            observation, reward, terminated, truncated, info = env.step(action, to_numpy=False)
            joint_pos.append(isaac_env.simulator.dof_state[..., 0])
            joint_vel.append(isaac_env.simulator.dof_state[..., 1])
            xpos_log.append(isaac_env.simulator._rigid_body_pos.reshape(num_envs, -1, 3))

            ooo = {k: v for k, v in observation.items() if k != "history"}
            _episode.add(
                tree_map(lambda x: x.cpu(), ooo),
                reward.cpu(),
                action.cpu(),
                terminated.cpu(),
                truncated.cpu(),
                tree_map(lambda x: x.cpu(), info),
            )

        episode_data = _episode.get()
        episode_data["xpos"] = torch.stack(xpos_log)[:-1]
        joint_pos = torch.stack(joint_pos)
        joint_vel = torch.stack(joint_vel)

        for m_id, envs_with_current_motion in motion_to_envs.items():
            for motion_repetition, env_id in enumerate(envs_with_current_motion):
                # A motion was executed on multiple environments
                # we average the results
                _joint_pos = joint_pos[: ctx_dict[m_id].shape[0] + 1, env_id]
                _target_joint_pos = tracking_joint_pos[m_id]
                local_metrics = _calc_metrics(
                    {
                        # "xpos": episode_data["xpos"][0 : ctx_dict[m_id].shape[0], env_id],
                        # "target_xpos": target_xpos_dict[env_id],
                        "tracking_target": tracking_targets[m_id],
                        "motion_id": m_id,
                        "motion_file": isaac_env._motion_lib.curr_motion_keys[m_id],
                        "observation": tree_map(lambda x: x[0 : ctx_dict[m_id].shape[0] + 1, env_id], episode_data["observation"]),
                        "joint_pos": _joint_pos,
                        "target_joint_pos": _target_joint_pos,
                        "dt": isaac_env.dt,
                    },
                )

                # Rename the metrics so that repetitions do not overlap
                assert len(local_metrics) == 1
                metric_key = list(local_metrics.keys())[0]

                if include_results_from_all_envs:
                    metrics[f"{metric_key}_repetition#{motion_repetition}"] = local_metrics[metric_key]
                else:
                    # Retain the original motion name without "repetition#"
                    # Note that this means that motions with the same name get overriden, so you only get one result
                    metrics[metric_key] = local_metrics[metric_key]
                    break

    return metrics



def distance_matrix(X: torch.Tensor, Y: torch.Tensor):
    X_norm = X.pow(2).sum(1).reshape(-1, 1)
    Y_norm = Y.pow(2).sum(1).reshape(1, -1)
    val = X_norm + Y_norm - 2 * torch.matmul(X, Y.T)
    return torch.sqrt(torch.clamp(val, min=0))


def emd_numpy(next_obs: torch.Tensor, tracking_target: torch.Tensor, prefix=""):
    # keep only pose part of the observations
    agent_obs = next_obs.to("cpu")
    tracked_obs = tracking_target.to("cpu")
    # compute optimal transport cost
    cost_matrix = distance_matrix(agent_obs, tracked_obs).cpu().detach().numpy()
    X_pot = np.ones(agent_obs.shape[0]) / agent_obs.shape[0]
    Y_pot = np.ones(tracked_obs.shape[0]) / tracked_obs.shape[0]
    transport_cost = ot.emd2(X_pot, Y_pot, cost_matrix, numItermax=100000)
    return {f"{prefix}emd": transport_cost}


def distance_proximity(next_obs: torch.Tensor, tracking_target: torch.Tensor, bound: float = 2.0, margin: float = 2, prefix=""):
    stats = {}
    dist = torch.norm(next_obs - tracking_target, dim=-1)
    in_bounds_mask = dist <= bound
    out_bounds_mask = dist > bound + margin
    stats[f"{prefix}proximity"] = (in_bounds_mask + ((bound + margin - dist) / margin) * (~in_bounds_mask) * (~out_bounds_mask)).mean()
    stats[f"{prefix}distance"] = dist.mean()
    return stats


def _calc_metrics(ep):
    metr = {}
    num_joints = ep["joint_pos"].shape[-1]
    next_obs = torch.as_tensor(ep["observation"]["state"][:, :num_joints], dtype=torch.float32)
    tracking_target = torch.as_tensor(ep["tracking_target"]["state"][:, :num_joints], dtype=torch.float32)
    dist_prox_res = distance_proximity(next_obs=next_obs, tracking_target=tracking_target, prefix="obs_state_")
    metr.update(dist_prox_res)
    emd_res = emd_numpy(next_obs=next_obs, tracking_target=tracking_target, prefix="obs_state_")
    metr.update(emd_res)
    # # add distance wrt xpos
    # xpos = torch.tensor(ep["xpos"], dtype=torch.float32).view(ep["xpos"].shape[0], -1)  # [keyframes, 72]
    # target_xpos = torch.tensor(ep["target_xpos"], dtype=torch.float32).view(ep["target_xpos"].shape[0], -1)
    # target_xpos = target_xpos[:tracking_length]

    # dist_prox_res = distance_proximity(next_obs=xpos, tracking_target=target_xpos)
    # for k, v in dist_prox_res.items():
    #     metr[f"{k}_xpos"] = v
    # # import ipdb; ipdb.set_trace()
    # metr["emd_xpos"] = emd_numpy(next_obs=xpos, tracking_target=target_xpos)["emd"]
    # jpos_pred = xpos.reshape(xpos.shape[0], -1, 3)  # length x 23 x 3
    # jpos_gt = target_xpos.reshape(target_xpos.shape[0], -1, 3)  # length x 23 x 3
    # metr["success_phc_linf_xpos"] = torch.all(torch.norm(jpos_pred - jpos_gt, dim=-1) <= 0.5).float()
    # metr["success_phc_mean_xpos"] = torch.all(torch.norm(jpos_pred - jpos_gt, dim=-1).mean(dim=-1) <= 0.5).float()

    # phc metrics
    phc_metrics = compute_joint_pos_metrics(joint_pos=ep["joint_pos"], target_joint_pos=ep["target_joint_pos"], dt=ep["dt"])
    metr.update(phc_metrics)
    for k, v in metr.items():
        if isinstance(v, torch.Tensor):
            metr[k] = v.tolist()
    metr["motion_id"] = ep["motion_id"]
    metr["motion_file"] = ep["motion_file"]
    return {ep["motion_file"]: metr}


def compute_joint_pos_metrics(joint_pos, target_joint_pos, dt):
    """Joint-angle errors for [time, joint] arrays, with dt in seconds."""
    if not np.isfinite(dt) or dt <= 0 or joint_pos.ndim != 2 or joint_pos.shape != target_joint_pos.shape or not joint_pos.numel():
        raise ValueError("Expected nonempty matching [time, joint] arrays and a positive finite timestep.")
    error = joint_pos - target_joint_pos
    stats = {"joint_angle_l2_rad": error.norm(dim=-1).mean()}
    # A derivative is undefined when the clip has too few frames.
    if error.shape[0] >= 2:
        velocity_error = torch.diff(error, dim=0) / dt
        stats["joint_velocity_l2_rad_s"] = velocity_error.norm(dim=-1).mean()
    if error.shape[0] >= 3:
        acceleration_error = torch.diff(velocity_error, dim=0) / dt
        stats["joint_acceleration_l2_rad_s2"] = acceleration_error.norm(dim=-1).mean()

    stats.update(
        distance_proximity(next_obs=joint_pos, tracking_target=target_joint_pos, prefix="")
    )
    stats.update(
        emd_numpy(next_obs=joint_pos, tracking_target=target_joint_pos, prefix="")
    )
    return stats
