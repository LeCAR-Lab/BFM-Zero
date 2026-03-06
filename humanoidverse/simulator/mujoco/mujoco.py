import os
import numpy as np
import torch
from loguru import logger
from pathlib import Path
import mujoco
import mujoco.viewer

from humanoidverse.utils.torch_utils import *
from humanoidverse.simulator.base_simulator.base_simulator import BaseSimulator

# Assume BaseSimulator is defined elsewhere.
class MuJoCo(BaseSimulator):
    def __init__(self, config, device):
        super().__init__(config, device)

        self.simulator_config = config.simulator.config
        self.robot_cfg = config.robot
        self.domain_rand_config = config.domain_rand
        self.device = device
        self.visualize_viewer = False
        self.renderer = None
        if config.save_rendering_dir is not None:
            self.save_rendering_dir = Path(config.save_rendering_dir)
        self.render_width=400
        self.render_height=400
    
    def setup(self):
        # Build the path to the MuJoCo model (MJCF/XML file)
        self.model_path = os.path.join(
            self.robot_cfg.asset.asset_root, 
            self.robot_cfg.asset.xml_file
        )
        hv_root = Path(__file__).parents[2]
        self.model_path = str(hv_root / "data/robots/g1/scene_29dof_freebase_mujoco.xml")
        self.freebase = True

        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)
        self.sim_substeps = self.simulator_config.sim.substeps
        self.sim_dt = 1 / self.simulator_config.sim.fps  # MuJoCo timestep from the model options.

        self.default_dof_frictionloss = self.model.dof_frictionloss.copy()
        self.default_body_mass = self.model.body_mass.copy()
        self.default_geom_friction = self.model.geom_friction.copy()

        self.model.opt.timestep = 1 / self.simulator_config.sim.fps
        # import ipdb;ipdb.set_trace()
        # self.model.opt.iterations = self.sim_substeps
        
        # MuJoCo does not support GPU acceleration, so we use CPU.
        
        # Optionally, set up a viewer for visualization.
        if self.visualize_viewer:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        else:
            self.viewer = None

        self.episodic_domain_randomization(None)


    def episodic_domain_randomization(self, env_ids):
        # First reset to defaults
        self.model.dof_frictionloss[:] = self.default_dof_frictionloss
        self.model.body_mass[:] = self.default_body_mass
        self.model.geom_friction[:] = self.default_geom_friction

        if self.domain_rand_config.get("randomize_link_mass", False):
            dmass = np.random.uniform(
                low=self.domain_rand_config["link_mass_range"][0],
                high=self.domain_rand_config["link_mass_range"][1],
                size=(self.model.nbody,),
            )
            self.model.body_mass[:] = self.model.body_mass * dmass

        # Scale static friction
        if self.domain_rand_config.get("randomize_friction", False):
            njnt = self.model.njnt
            # TODO isaacsim config sets the numbers straight to [-0.5, 1.25],
            #      but we do not know how these map to mujoco frictionloss.
            #      The change seems(?) drastic, so instead here we randomly sample a multiplier
            #      (with positive multipliers, as friction should probably not be negative)

            # NOTE: this changes friction of joints
            frictionloss = self.model.dof_frictionloss[6:] * np.random.uniform(
                low=abs(self.domain_rand_config["friction_range"][0]),
                high=abs(self.domain_rand_config["friction_range"][1]),
                size=(njnt - 1,),
            )

            self.model.dof_frictionloss[6:] = frictionloss

            # NOTE: This changes friction of the geoms (but does not seem to change the dynamic friction)
            friction = self.model.geom_friction * np.random.uniform(
                low=abs(self.domain_rand_config["friction_range"][0]),
                high=abs(self.domain_rand_config["friction_range"][1]),
                size=(self.model.ngeom, 1),
            )
            self.model.geom_friction[:] = friction

            # TODO: dynamic friction?

        if self.domain_rand_config.get("randomize_base_com", False):
            # get id of torso
            self.torso_id = self.model.body("torso_link").id
            assert self.torso_id > -1
            x_uniform_range = self.domain_rand_config["base_com_range"]["x"]
            assert self.domain_rand_config["base_com_range"]["y"] == x_uniform_range
            assert self.domain_rand_config["base_com_range"]["z"] == x_uniform_range
            #TODO: is this an appropriate way to randomize base com? We add weight to the torso
            dmass_torso = np.random.uniform(low=x_uniform_range[0], high=x_uniform_range[1])
            self.model.body_mass[self.torso_id] = self.model.body_mass[self.torso_id] + dmass_torso

    def setup_terrain(self, mesh_type):
        """Sets up the terrain based on the specified mesh type."""
        if mesh_type == 'plane':
            pass
            # self._create_ground_plane()
        elif mesh_type == 'heightfield':
            self._create_heightfield()
        elif mesh_type == 'trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognized. Allowed types are [None, plane, heightfield, trimesh]")

    def _create_ground_plane(self):
        """Creates a ground plane in MuJoCo by modifying the model's geom properties."""
        print("Creating plane terrain")

        # MuJoCo uses a geom of type "plane" for ground planes.
        plane_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "ground_plane")
        if plane_id == -1:
            # Add a new geom for the plane
            new_geom_id = self.model.ngeom
            self.model.geom_pos = np.vstack([self.model.geom_pos, [0, 0, 0]])
            self.model.geom_size = np.vstack([self.model.geom_size, [1, 1, 0.01]])  # Plane size
            self.model.geom_friction = np.vstack([
                self.model.geom_friction,
                [self.simulator_config.terrain.static_friction, 
                 self.simulator_config.terrain.dynamic_friction, 0]
            ])
            self.model.geom_type = np.append(self.model.geom_type, mujoco.mjtGeom.mjGEOM_PLANE)

            print("Created plane terrain")
        else:
            print("Plane terrain already exists.")

    def _create_heightfield(self):
        """Creates a heightfield terrain in MuJoCo."""
        print("Creating heightfield terrain")

        heightfield_size = (self.simulator_config.terrain.width, self.simulator_config.terrain.length)
        height_samples = self._genemodel_pathrate_heightfield_data(heightfield_size)

        # MuJoCo expects heightfields to be normalized between 0 and 1
        height_samples = (height_samples - np.min(height_samples)) / (np.max(height_samples) - np.min(height_samples))

        # Define the heightfield in the MuJoCo model
        hf_id = self.model.nhfield
        self.model.hfield_nrow[hf_id] = heightfield_size[0]
        self.model.hfield_ncol[hf_id] = heightfield_size[1]
        self.model.hfield_size[hf_id] = [self.simulator_config.terrain.horizontal_scale, 
                                         self.simulator_config.terrain.horizontal_scale, 
                                         self.simulator_config.terrain.vertical_scale]
        self.model.hfield_data[hf_id] = height_samples.flatten()

        print("Created heightfield terrain")

    def load_assets(self):
        # Extract degrees of freedom (DOFs) and bodies from the model.
        self.num_dof = self.model.nv - 6  # Number of generalized velocities.
        self.num_bodies = self.model.nbody -1
        
        # Retrieve joint and body names.
        self.dof_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(self.model.njnt)][1: ]
        self.body_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, b) for b in range(self.model.nbody)][1: ]

        self.body_id = np.arange(self.num_bodies, dtype=np.int32) + 1

        if "23" in self.model_path:
            for b in range(self.model.nbody):
                if "wrist" in mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, b) or "hand" in mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, b):
                    self.body_names.remove(mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, b))
                    self.num_bodies -= 1
                    self.body_id = np.delete(self.body_id, np.where(self.body_id == b))

        if "29" in self.model_path:
            for b in range(self.model.nbody):
                if "hand" in mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, b):
                    self.body_names.remove(mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, b))
                    self.num_bodies -= 1
                    self.body_id = np.delete(self.body_id, np.where(self.body_id == b))
        
        # Validate configuration consistency.
        assert self.num_dof == len(self.robot_cfg.dof_names), "Number of DOFs must match the config."
        assert self.num_bodies == len(self.robot_cfg.body_names), "Number of bodies must match the config."
        assert self.dof_names == self.robot_cfg.dof_names, "DOF names must match the config."
        assert self.body_names == self.robot_cfg.body_names, "Body names must match the config."
    
    def create_envs(self, num_envs, env_origins, base_init_state):
        # MuJoCo does not support multiple environments in a single simulation.
        # import ipdb; ipdb.set_trace()
        self.num_envs = 1
        self.env_config = self.config
        self.env_origins = env_origins
        self.envs = [self.data]
        self.robot_handles = []  # Not applicable in MuJoCo.
        self.base_init_state = base_init_state
        # Set initial state using provided base_init_state.
        # import ipdb; ipdb.set_trace()
        self.data.qpos[0:3] = base_init_state[0:3].cpu()
        self.data.qpos[3:7] = base_init_state[3:7][..., [3, 0, 1, 2]].cpu()
        self.data.qvel[0:6] = base_init_state[7:13].cpu()
        # mujoco.mj_forward(self.model, self.data)
        self._body_list = self.body_names
        # dof_props_asset = self.get_dof_properties(self.model)
        # dof_props = self._process_dof_props(dof_props_asset, 0)
        return self.envs, self.robot_handles
    
    def prepare_sim(self):
        # In MuJoCo, forward simulation updates the state.
        mujoco.mj_forward(self.model, self.data)
        mujoco.mj_step(self.model, self.data)
        # import ipdb; ipdb.set_trace()
        self.refresh_sim_tensors()
    
    def refresh_sim_tensors(self):
        pass  
    
    def apply_torques_at_dof(self, torques):
        # Convert torch tensor to numpy if needed.
        if isinstance(torques, torch.Tensor):
            torques = torques.cpu().numpy()
        if self.freebase:
            self.data.ctrl[6:] = torques
        else:   
            self.data.ctrl[:] = torques
        # mujoco.mj_step(self.model, self.data)
    
    def set_actor_root_state_tensor(self, set_env_ids, root_states):
        # In MuJoCo, the full state is given by qpos and qvel.
        if isinstance(root_states, torch.Tensor):
            root_states[:, 10:13] = quat_rotate_inverse(self.base_quat, root_states[:, 10:13], w_last=True)
            root_states = root_states.cpu().numpy()
        self.data.qpos[0:3] = root_states[0, 0:3]
        self.data.qpos[3:7] = root_states[0, 3:7][..., [3, 0, 1, 2]]
        self.data.qvel[0:3] = root_states[0, 7:10]
        self.data.qvel[3:6] = root_states[0, 10:13]
        # print(root_states.shape)
        # mujoco.mj_forward(self.model, self.data)
    
    def set_dof_state_tensor(self, set_env_ids, dof_states):
        # Update joint positions and velocities.
        if isinstance(dof_states, torch.Tensor):
            dof_states = dof_states.cpu().numpy()
        self.data.qpos[7:] = dof_states[0, :, 0]
        self.data.qvel[6:] = dof_states[0, :, 1]
        # mujoco.mj_forward(self.model, self.data)
    
    def apply_rigid_body_force_at_pos_tensor(self, force_tensor, force, pos):
        # In MuJoCo, external forces are applied via xfrc_applied.
        # xfrc_applied is a (nbody, 6) array: first three for force, last three for torque.
        self.data.xfrc_applied[:, 0:3] = force_tensor
        # mujoco.mj_step(self.model, self.data)
    
    def simulate_at_each_physics_step(self):
        # import ipdb;
        # ipdb.set_trace()
        mujoco.mj_step(self.model, self.data)
        # if self.viewer is not None:
        #     mujoco.viewer.sync(self.viewer, self.model, self.data)
        self.refresh_sim_tensors()
        if self.viewer is not None:
            self.viewer.sync()

    def get_dof_properties(self, model):
        """ Retrieves the DOF properties for a robot in a MuJoCo simulation.
            Retrieves joint position limits, velocity limits, and torque limits.

        Args:
            model (mujoco.MjModel): MuJoCo model containing the robot's assets.

        Returns:
            dict: A dictionary containing DOF properties like position limits, velocity limits, and torque limits.
        """
        dof_props = {}

        # Position limits (lower and upper)
        dof_props["lower"] = torch.tensor([model.jnt_range[1:][i, 0] for i in range(self.num_dof)])
        dof_props["upper"] = torch.tensor([model.jnt_range[1:][i, 1] for i in range(self.num_dof)])

        # Velocity limits (using model.dof_damping for approximation or another method)
        dof_props["velocity"] = torch.tensor([model.dof_damping[6:][i] for i in range(self.num_dof)])

        # Torque limits (from actuator control range)
        dof_props["effort"] = torch.tensor([model.actuator_ctrlrange[6:][i, 1] for i in range(self.num_dof)])

        return dof_props

    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id==0:
            self.hard_dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)

            self.dof_pos_limits_termination = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)

            for i in range(len(props)):
                
                self.hard_dof_pos_limits[i, 0] = props["lower"][i].item()
                self.hard_dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.env_config.rewards.reward_limit.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.env_config.rewards.reward_limit.soft_dof_pos_limit

                self.dof_pos_limits_termination[i, 0] = m - 0.5 * r * self.env_config.termination_scales.termination_close_to_dof_pos_limit
                self.dof_pos_limits_termination[i, 1] = m + 0.5 * r * self.env_config.termination_scales.termination_close_to_dof_pos_limit
        return props

    def get_dof_limits_properties(self):
        self.hard_dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.sim_device, requires_grad=False)
        self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.sim_device, requires_grad=False)
        self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.sim_device, requires_grad=False)
        self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.sim_device, requires_grad=False)
        for i in range(self.num_dof):
            self.hard_dof_pos_limits[i, 0] = self.robot_cfg.dof_pos_lower_limit_list[i]
            self.hard_dof_pos_limits[i, 1] = self.robot_cfg.dof_pos_upper_limit_list[i]
            self.dof_pos_limits[i, 0] = self.robot_cfg.dof_pos_lower_limit_list[i]
            self.dof_pos_limits[i, 1] = self.robot_cfg.dof_pos_upper_limit_list[i]
            self.dof_vel_limits[i] = self.robot_cfg.dof_vel_limit_list[i]
            self.torque_limits[i] = self.robot_cfg.dof_effort_limit_list[i]
            # soft limits
            m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
            r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
            self.dof_pos_limits[i, 0] = m - 0.5 * r * self.config.rewards.reward_limit.soft_dof_pos_limit
            self.dof_pos_limits[i, 1] = m + 0.5 * r * self.config.rewards.reward_limit.soft_dof_pos_limit
        return self.dof_pos_limits, self.dof_vel_limits, self.torque_limits



    def apply_commands(self, commands_description):
        if commands_description == "forward_command":
            self.commands[:, 4] = 1
            self.commands[:, 0] += 0.4
            logger.info(f"Current Command: {self.commands[:, ]}")
        elif commands_description == "backward_command":
            self.commands[:, 0] -= 0.1
            logger.info(f"Current Command: {self.commands[:, ]}")
        elif commands_description == "left_command":
            self.commands[:, 1] -= 0.1
            logger.info(f"Current Command: {self.commands[:, ]}")
        elif commands_description == "right_command":
            self.commands[:, 1] += 0.1
            logger.info(f"Current Command: {self.commands[:, ]}")
        elif commands_description == "heading_left_command":
            self.commands[:, 3] -= 0.1
            logger.info(f"Current Command: {self.commands[:, ]}")
        elif commands_description == "heading_right_command":
            self.commands[:, 3] += 0.1
            logger.info(f"Current Command: {self.commands[:, ]}")
        elif commands_description == "zero_command":
            self.commands[:, :4] = 0
            logger.info(f"Current Command: {self.commands[:, ]}")

    def find_rigid_body_indice(self, body_name):
        """
        Finds the index of a specified rigid body.

        Args:
            body_name (str): Name of the rigid body to locate.

        Returns:
            int: Index of the rigid body.
        """
        if body_name not in self.body_names:
            raise ValueError(f"Rigid body '{body_name}' not found in the model.")
        return self.body_names.index(body_name)

    def setup_viewer(self):
        """
        Sets up a viewer for visualizing the simulation, allowing keyboard interactions.
        """
        # self.viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data)
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
    
    def render(self, sync_frame_time=True):
        """
        Renders the simulation frame-by-frame, syncing frame time if required.

        Args:
            sync_frame_time (bool): Whether to synchronize the frame time.
        """
        if self.renderer is None:
            self.renderer = mujoco.Renderer(
                self.model,
                width=self.render_width,
                height=self.render_height,
            )
        self.renderer.update_scene(
            self.data,
            camera="track"
        )
        return self.renderer.render()
        if self.viewer is None:
            raise RuntimeError("Viewer is not initialized. Call 'setup_viewer' first.")
        return
        # mujoco.mj_step(self.model, self.data)
        # self.viewer.sync()

    @property
    def dof_state(self):
        return torch.cat([self.dof_pos[..., None], self.dof_vel[..., None]], dim=-1)
    
    @property
    def robot_root_states(self):
        base_quat = self.base_quat
        qvel_tensor = torch.tensor([self.data.qvel[0:6]], device=self.device, dtype=torch.float32)
        return torch.cat(
            [
                torch.tensor([self.data.qpos[0:3]], device=self.device, dtype=torch.float32),
                base_quat,
                qvel_tensor[:, 0:3],
                quat_rotate(base_quat, qvel_tensor[:, 3:6], w_last=True),
            ], dim=-1
        )
    
    @property
    def contact_forces(self):
        return torch.tensor([self.data.cfrc_ext[1:, 3:6]], device=self.device, dtype=torch.float32)
    
    @property
    def base_quat(self):
        return torch.tensor([self.data.qpos[3:7]], device=self.device, dtype=torch.float32)[..., [1, 2, 3, 0]]
    
    @property
    def all_root_states(self):
        base_quat = self.base_quat
        qvel_tensor = torch.tensor([self.data.qvel[0:6]], device=self.device, dtype=torch.float32)
        return torch.cat(
            [
                torch.tensor([self.data.qpos[0:3]], device=self.device, dtype=torch.float32),
                base_quat,
                qvel_tensor[:, 0:3],
                quat_rotate(base_quat, qvel_tensor[:, 3:6], w_last=True),
            ], dim=-1
        )
    
    @property
    def dof_pos(self):
        return torch.tensor([self.data.qpos[7:]], device=self.device, dtype=torch.float32)
    
    @property
    def dof_vel(self):
        return torch.tensor([self.data.qvel[6:]], device=self.device, dtype=torch.float32)
    
    @property
    def _rigid_body_pos(self):
        return torch.tensor([self.data.xpos[self.body_id, :]], device=self.device, dtype=torch.float32)[:]
    
    @property
    def _rigid_body_rot(self):
        return torch.tensor([self.data.xquat[self.body_id, :]], device=self.device, dtype=torch.float32)[..., [1, 2, 3, 0]]
    
    @property
    def _rigid_body_vel(self):
        return torch.tensor([self.data.cvel[self.body_id, 3:6]], device=self.device, dtype=torch.float32) # Using xvelp for global velocity
    
    @property
    def _rigid_body_ang_vel(self):
        return torch.tensor([self.data.cvel[self.body_id, 0:3]], device=self.device, dtype=torch.float32)
