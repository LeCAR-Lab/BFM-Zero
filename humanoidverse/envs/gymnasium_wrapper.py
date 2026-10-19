import gymnasium
from gymnasium.vector import VectorEnv
from gymnasium import Env
import torch
from typing import Any, Dict, Tuple, Union
from .legged_robot_motions.legged_robot_motions import LeggedRobotMotions
import numpy as np


class HumanoidVerseVectorEnv(VectorEnv):
    """Wrapper to make the humanoidverse env compatible with gymnasium's VectorEnv interface."""

    def __init__(
        self,
        env: LeggedRobotMotions,
        add_time_aware_observation: bool = True,
    ):
        super().__init__()
        self._env = env
        self.spec = self._env.spec
        self.num_envs = self._env.num_envs
        self.add_time_aware_observation = add_time_aware_observation

        # TODO does might not be ideal, but we need to reset things to get the observation space
        example_observation, _ = self.reset()
        observation_spaces = {}
        for key, value in example_observation.items():
            observation_spaces[key] = gymnasium.spaces.Box(
                low=-float("inf"),
                high=float("inf"),
                shape=value.shape,
                dtype=np.float32,
            )

        if add_time_aware_observation:
            # Update the observation spaces to include time-aware observation
            observation_spaces["time"] = gymnasium.spaces.Box(
                low=0.0,
                high=float("inf"),
                shape=(1,),
                dtype=np.int32,
            )
        self.observation_space = gymnasium.spaces.Dict(observation_spaces)
        self.single_action_space = self._env.get_wrapper_attr("single_action_space")
        # Repeat the action space to match the vectorized environment
        action_space_shape = (self.num_envs,) + self.single_action_space.shape
        self.action_space = gymnasium.spaces.Box(
            low=np.tile(self.single_action_space.low, (self.num_envs, 1)),
            high=np.tile(self.single_action_space.high, (self.num_envs, 1)),
            shape=action_space_shape,
            dtype=self.single_action_space.dtype,
        )

    @property
    def single_observation_space(self):
        """Return the observation space for a single environment."""
        single_obs_spaces = {}
        for key, space in self.observation_space.spaces.items():
            single_obs_spaces[key] = gymnasium.spaces.Box(
                low=space.low[0],
                high=space.high[0],
                shape=space.shape[1:],
                dtype=space.dtype,
            )
        return gymnasium.spaces.Dict(single_obs_spaces)

    @property
    def device(self):
        return self.base_env.device

    @property
    def base_env(self) -> Env:
        return self._env.unwrapped

    @property
    def unwrapped(self):
        return self.base_env

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if seed is not None:
            from humanoidverse.agents.utils import set_seed_everywhere

            set_seed_everywhere(seed)
            self.action_space.seed(seed)
            self.single_action_space.seed(seed)
        _, info = self.base_env.reset_all()
        observation = self._get_g1env_observation()
        return observation, info

    def _get_g1env_observation(self):
        """Turn current Isaac sim state into a G1Env-like observation."""
        raw_obs = self._env.obs_buf_dict_raw["actor_obs"]
        g1env_state = torch.cat(
            [
                raw_obs["dof_pos"],
                raw_obs["dof_vel"],
                raw_obs["projected_gravity"],
                raw_obs["base_ang_vel"],
            ],
            dim=-1,
        )
        last_action = raw_obs["actions"]
        privileged_state = raw_obs["max_local_self"]
        observation = {
            "state": g1env_state,
            "last_action": last_action,
            "privileged_state": privileged_state,
        }
        if self.add_time_aware_observation:
            observation["time"] = self._env.episode_length_buf.unsqueeze(-1)
        return observation

    def step(self, actions: Union[torch.Tensor, Dict]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        actions = torch.tensor(actions, device=self._env.device, dtype=torch.float32) if isinstance(actions, np.ndarray) else actions
        actions = {"actions": actions}
        _, reward, reset, new_info = self.base_env.step(actions)

        reset = reset.bool()
        time_outs = new_info["time_outs"].bool()
        terminated = torch.logical_and(reset, ~time_outs)
        truncated = time_outs

        observation = self._get_g1env_observation()
        return observation, reward, terminated, truncated, new_info

    def close(self):
        return self.base_env.close()


_ISAAC_SIM_INITIALIZED = False


def instantiate_isaac_sim(num_envs):
    global _ISAAC_SIM_INITIALIZED
    if _ISAAC_SIM_INITIALIZED:
        return
    # Import things lazily
    import argparse

    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="")
    AppLauncher.add_app_launcher_args(parser)

    args_cli, _ = parser.parse_known_args()
    args_cli.num_envs = num_envs
    args_cli.enable_cameras = False
    args_cli.headless = True

    app_launcher = AppLauncher(args_cli)
    _ = app_launcher.app
    _ISAAC_SIM_INITIALIZED = True


# class HumanoidVerseIsaacConfig(BaseConfig):
#     name: tp.Literal["humanoidverse_isaac"] = "humanoidverse_isaac"

#     lafan_tail_path: str

#     # Relative path inside the humanoidverse/config directory
#     relative_config_path: str = HYDRA_CONFIG_REL_PATH

#     def build(self, num_envs: int = 1) -> tp.Tuple[gymnasium.Env, tp.Any]:
#         # TODO num_envs does not work yet
#         assert num_envs >= 1

#         # Use config from humanoidverse to create the environment
#         # however, we need to make sure we use isaacsim instead of isaacgym
#         # --> create new file with that single line changed
#         original_config_path = os.path.join(HYDRA_CONFIG_DIR, self.relative_config_path + ".yaml")
#         config_contents = open(original_config_path, "r").read()
#         config_contents = config_contents.replace("/simulator: isaacgym", "/simulator: isaacsim")
#         new_config_rel_path = self.relative_config_path + "_isaacsim"
#         new_config_path = os.path.join(HYDRA_CONFIG_DIR, new_config_rel_path + ".yaml")
#         with open(new_config_path, "w") as f:
#             f.write(config_contents)

#         with hydra.initialize_config_dir(config_dir=HYDRA_CONFIG_DIR):
#             cfg = hydra.compose(config_name=new_config_rel_path)

#         # Add custom resolvers used in the configs
#         if not OmegaConf.has_resolver("eval"):
#             OmegaConf.register_new_resolver("eval", lambda x: eval(x))

#         # We need to manually fix some paths
#         cfg.num_envs = num_envs
#         cfg.exp_base = "__no_exp_base__"
#         cfg.env.config.headless = True
#         cfg.robot.asset.asset_root = cfg.robot.asset.asset_root.replace("humanoidverse", HUMANOIDVERSE_DIR)
#         cfg.robot.motion.asset.assetRoot = cfg.robot.motion.asset.assetRoot.replace("humanoidverse", HUMANOIDVERSE_DIR)
#         cfg.robot.motion.motion_file = self.lafan_tail_path

#         # This sets obs/action dims etc
#         pre_process_config(cfg)

#         instantiate_isaac_sim(num_envs)
#         OmegaConf.set_struct(cfg, False)
#         isaac_env = LeggedRobotMotions(cfg.env.config, device="cuda:0")

#         env = HumanoidVerseVectorEnv(isaac_env)

#         return env, {}
