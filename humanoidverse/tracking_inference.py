import os

os.environ["MUJOCO_GL"] = "egl"  # Use EGL for rendering
os.environ["OMP_NUM_THREADS"] = "1"

from pathlib import Path
from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
import json
from humanoidverse.agents.envs.humanoidverse_isaac import HumanoidVerseIsaacConfig, IsaacRendererWithMuJoco
import torch
from humanoidverse.utils.helpers import export_meta_policy_as_onnx
from humanoidverse.utils.helpers import get_backward_observation
import joblib
import mediapy as media
import numpy as np
from torch.utils._pytree import tree_map

import humanoidverse
if getattr(humanoidverse, "__file__", None) is not None:
    HUMANOIDVERSE_DIR = Path(humanoidverse.__file__).parent
else:
    HUMANOIDVERSE_DIR = Path(__file__).resolve().parent


def main(model_folder: Path, data_path: Path | None = None, headless: bool = True, device="cuda", simulator: str = "isaacsim", save_mp4: bool=False, disable_dr: bool = False, disable_obs_noise: bool = False, motion_list: list[int] = [25]):
    # motion_list: motion ids to evaluate (default [25])
    
    model_folder = Path(model_folder)

    model = load_model_from_checkpoint_dir(model_folder / "checkpoint", device=device)
    model.to(device)
    model.eval()
    model_name = "model"
    model_name = model.__class__.__name__
    with open(model_folder / "config.json", "r") as f:
        config = json.load(f)
        
    use_root_height_obs = config["env"].get("root_height_obs", False)

    if data_path is not None:
        config["env"]["lafan_tail_path"] = str(Path(data_path).resolve())
    elif not Path(config["env"].get("lafan_tail_path", "")).exists():
        default_path = HUMANOIDVERSE_DIR / "data" / "lafan_29dof.pkl"
        if default_path.exists():
            config["env"]["lafan_tail_path"] = str(default_path)
        else:
            config["env"]["lafan_tail_path"] = "data/lafan_29dof.pkl"
    # import ipdb; ipdb.set_trace()
    config["env"]["hydra_overrides"].append("env.config.max_episode_length_s=10000")
    config["env"]["hydra_overrides"].append(f"env.config.headless={headless}")
    config["env"]["hydra_overrides"].append(f"simulator={simulator}")
    config["env"]["disable_domain_randomization"] = disable_dr
    config["env"]["disable_obs_noise"] = disable_obs_noise

    # Outputs under model_folder/tracking_inference (sibling of exported/)
    output_dir = model_folder / "exported"
    output_dir.mkdir(parents=True, exist_ok=True)
    export_meta_policy_as_onnx(
        model,
        output_dir,
        f"{model_name}.onnx",
        {"actor_obs": torch.randn(1, model._actor.input_filter.output_space.shape[0] + model.cfg.archi.z_dim)},
        z_dim=model.cfg.archi.z_dim,
        history=('history_actor' in model.cfg.archi.actor.input_filter.key),
        use_29dof=True,
    )
    print(f"Exported model to {output_dir}/{model_name}.onnx")

    def tracking_inference(obs) -> torch.Tensor:
        z = model.backward_map(obs)
        for step in range(z.shape[0]):
            end_idx = min(step + 1, z.shape[0])
            z[step] = z[step:end_idx].mean(dim=0)
        return model.project_z(z)

    # rgb_renderer = IsaacRendererWithMuJoco(render_size=256)
    env_cfg = HumanoidVerseIsaacConfig(**config["env"])
    num_envs = 1
    wrapped_env, _ = env_cfg.build(num_envs)
    env = wrapped_env._env
    print("="*80)
    print(env.config.simulator)
    print("-"*80)
    
    output_dir = model_folder / "tracking_inference"

    for MOTION_ID in motion_list:
        env.set_is_evaluating(MOTION_ID)
        # we visulize the first env
        obs, obs_dict = get_backward_observation(env, 0, use_root_height_obs=use_root_height_obs)

        expert_qpos = np.concatenate([
            obs_dict["ref_body_pos"][:,0].cpu().numpy(),
            np.roll(obs_dict["ref_body_rots"][:,0].cpu().numpy(),1,axis=-1),
            obs_dict["dof_pos"].cpu().numpy()
        ], axis=-1)

        # import ipdb; ipdb.set_trace()

        z = tracking_inference(tree_map(lambda x: x[1:], obs))
        output_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(z.cpu().numpy(), output_dir / f"zs_{MOTION_ID}.pkl")
        print(f"Saved zs_{MOTION_ID}.pkl")
        
    observation, info = wrapped_env.reset(to_numpy=False)

    # Root state: pos(3) + quat(4) + lin_vel(3) + ang_vel(3). Isaac expects quat as wxyz; motion lib uses xyzw.
    ref_body_rots = obs_dict["ref_body_rots"][0, 0].clone()
    if simulator == "isaacsim":
        ref_body_rots = ref_body_rots[[3, 0, 1, 2]]  # xyzw -> wxyz for correct humanoid facing in Isaac
    ref_root_init_state = torch.cat(
            [
                obs_dict["ref_body_pos"][0, 0],
                ref_body_rots,
                obs_dict["ref_body_vels"][0, 0],
                obs_dict["ref_body_angular_vels"][0, 0],
            ]
        )
    dof_init_state = torch.zeros_like(wrapped_env._env.simulator.dof_state.view(num_envs, -1, 2)[0])
    dof_init_state[..., 0] = obs_dict["dof_pos"][0]
    dof_init_state[..., 1] = obs_dict["ref_dof_vel"][0]
    target_states = {
        "dof_states": dof_init_state,
        "root_states": torch.stack([ref_root_init_state.clone() for i in range(num_envs)])
    }
    env_ids = torch.arange(num_envs, dtype=torch.long)
    observation, info = wrapped_env._env.reset_envs_idx(env_ids, target_states=target_states)
    # refresh_env_ids = wrapped_env._env.need_to_refresh_envs.nonzero(as_tuple=False).flatten()
    # wrapped_env._env.simulator.set_actor_root_state_tensor(refresh_env_ids, wrapped_env._env.target_robot_root_states)
    # wrapped_env._env.simulator.set_dof_state_tensor(refresh_env_ids, wrapped_env._env.target_robot_dof_state)
    # wrapped_env._env.need_to_refresh_envs[refresh_env_ids] = False
    observation_new, reward, terminated, truncated, info = wrapped_env.step(torch.zeros((num_envs, wrapped_env.action_space.shape[-1]), dtype=torch.float32), to_numpy=False)
    observation = wrapped_env._get_g1env_observation(to_numpy=False)
    qpos, qvel = wrapped_env._get_qpos_qvel(to_numpy=True)
    assert np.allclose(wrapped_env._env.simulator.dof_pos.clone().cpu(), expert_qpos[0,7:])
    joint_pos = [wrapped_env._env.simulator.dof_state[..., 0].clone().cpu().numpy()]

    # Visualization length: match inference length so expert and policy videos align
    episode_len = z.shape[0]
    episode_len = 3000
    print(f"Saving video for tracking ({episode_len} steps)")
    if save_mp4:
        rgb_renderer = IsaacRendererWithMuJoco(render_size=256)
        # Only render 1 + episode_len frames (same as frames list), not the full motion
        expert_video = rgb_renderer.from_qpos(expert_qpos[: 1 + episode_len])
        frames = [rgb_renderer.render(wrapped_env._env, 0)[0]]

    print(f"Running tracking inference for {episode_len} steps")
    for i in range(episode_len):
        print(f"Step {i} of {episode_len}")
        action = model.act(observation, z[i % len(z)].repeat(num_envs, 1), mean=True)
        observation, reward, terminated, truncated, info = wrapped_env.step(action, to_numpy=False)
        joint_pos.append(wrapped_env._env.simulator.dof_state[..., 0].clone().cpu().numpy())
        if save_mp4:
            frames.append(rgb_renderer.render(wrapped_env._env, 0)[0])

    joint_pos = np.stack(joint_pos, axis=0).squeeze(1)
    stats = {}
    
    # breakpoint()  # use PYTHONBREAKPOINT=0 to disable, or install ipdb for a nicer debugger

    if save_mp4:
        new_frames = []
        for a, b in zip(expert_video, frames):
            new_frames.append(np.concatenate([a, b], axis=1))
        video_path = output_dir / f"tracking_{MOTION_ID}.mp4"
        media.write_video(str(video_path), new_frames, fps=50)
        print(f"Saved video for tracking: {video_path}")


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
