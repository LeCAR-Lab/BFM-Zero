"""
MuJoCo environment wrapper for BFM-Zero policy.

Provides a clean gym-like interface for running the BFM-Zero policy in MuJoCo.
"""

import numpy as np
import torch
import mujoco
from scipy.spatial.transform import Rotation
from torch.utils._pytree import tree_map
from collections import OrderedDict


# Rotation utility functions (implemented without external dependencies)
def quat_mul(a, b, w_last=True):
    """Multiply two quaternions. Format: [x, y, z, w] if w_last=True, else [w, x, y, z]"""
    if isinstance(a, np.ndarray):
        a = torch.tensor(a)
    if isinstance(b, np.ndarray):
        b = torch.tensor(b)
    
    shape = a.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 4)
    
    if w_last:
        x1, y1, z1, w1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
        x2, y2, z2, w2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    else:
        w1, x1, y1, z1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
        w2, x2, y2, z2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)
    
    if w_last:
        quat = torch.stack([x, y, z, w], dim=-1).view(shape)
    else:
        quat = torch.stack([w, x, y, z], dim=-1).view(shape)
    return quat


def quat_rotate(q, v):
    """Rotate vector v by quaternion q. q format: [x, y, z, w]"""
    if isinstance(q, np.ndarray):
        q = torch.tensor(q)
    if isinstance(v, np.ndarray):
        v = torch.tensor(v)
    
    shape = q.shape
    q_w = q[:, -1]  # w component
    q_vec = q[:, :3]  # x, y, z components
    
    a = v * (2.0 * q_w**2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.bmm(q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)).squeeze(-1) * 2.0
    return a + b + c


def quat_to_tan_norm(q, w_last=True):
    """Convert quaternion to tangent-normal representation (6D). Returns [tan, norm]"""
    if isinstance(q, np.ndarray):
        q = torch.tensor(q)
    
    # Reference vectors
    ref_tan = torch.zeros_like(q[..., 0:3])
    ref_tan[..., 0] = 1  # [1, 0, 0]
    
    ref_norm = torch.zeros_like(q[..., 0:3])
    ref_norm[..., -1] = 1  # [0, 0, 1]
    
    if w_last:
        tan = quat_rotate(q, ref_tan)
        norm = quat_rotate(q, ref_norm)
    else:
        raise NotImplementedError("w_last=False not implemented")
    
    norm_tan = torch.cat([tan, norm], dim=len(tan.shape) - 1)
    return norm_tan


def calc_heading_quat_inv(q, w_last=True):
    """Calculate inverse heading quaternion (rotation around z-axis only)"""
    if isinstance(q, np.ndarray):
        q = torch.tensor(q)
    
    # Extract heading (yaw) from quaternion
    # For quaternion [x, y, z, w], heading is atan2(2*(w*z + x*y), 1 - 2*(y^2 + z^2))
    if w_last:
        x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    else:
        w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    
    # Calculate heading angle
    heading = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    
    # Create rotation quaternion around z-axis by -heading
    axis = torch.zeros_like(q[..., 0:3])
    axis[..., 2] = 1  # z-axis
    
    # Quaternion from angle-axis: q = [sin(θ/2)*axis, cos(θ/2)]
    half_angle = -heading / 2.0
    cos_half = torch.cos(half_angle)
    sin_half = torch.sin(half_angle)
    
    if w_last:
        heading_q = torch.cat([sin_half.unsqueeze(-1) * axis, cos_half.unsqueeze(-1)], dim=-1)
    else:
        heading_q = torch.cat([cos_half.unsqueeze(-1), sin_half.unsqueeze(-1) * axis], dim=-1)
    
    return heading_q

# Try to import imageio for video saving
try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    imageio = None
    HAS_IMAGEIO = False

from bfm_zero_inference_code.fb_cpr_aux.model import FBcprAuxModel


class MuJoCoBFMZeroEnv:
    """
    MuJoCo environment wrapper for BFM-Zero policy.
    
    Provides a clean interface:
    - obs, next_obs, reward, info = env.step(action)
    - obs = env.reset()
    """
    
    def __init__(self, robot_xml,
                 kp_gains=None, kd_gains=None, default_joint_pos=None,
                 action_scales=None, action_rescale=5.0,
                 sim_dt=0.005, n_substeps=4,
                 enable_video=False, video_path="simulation.mp4", video_fps=30):
        """
        Initialize the environment.
        
        Args:
            model_path: Path to BFM-Zero model
            robot_xml: Path to MuJoCo XML file
            kp_gains: PD control proportional gains (29 dim), or None for defaults
            kd_gains: PD control derivative gains (29 dim), or None for defaults
            default_joint_pos: Default joint positions (29 dim), or None for zeros
            action_scales: Per-joint action scales (29 dim), or None for ones
            action_rescale: Global action rescale factor
            sim_dt: Simulation timestep in seconds (default: 0.002 = 2ms)
            n_substeps: Number of simulation substeps per control step (default: 1)
            enable_video: Whether to record video during simulation
            video_path: Path to save video file
            video_fps: Frames per second for video
        """
        # Load MuJoCo model
        self.mjm = mujoco.MjModel.from_xml_path(robot_xml)
        self.mjd = mujoco.MjData(self.mjm)
        
        # Set simulation timestep
        self.sim_dt = sim_dt
        self.n_substeps = n_substeps
        self.mjm.opt.timestep = sim_dt
        
        # PD control parameters
        self.kp_gains = kp_gains if kp_gains is not None else np.ones(29, dtype=np.float32) * 100.0
        self.kd_gains = kd_gains if kd_gains is not None else np.ones(29, dtype=np.float32) * 10.0
        self.default_joint_pos = default_joint_pos if default_joint_pos is not None else np.zeros(29, dtype=np.float32)
        self.action_scales = action_scales if action_scales is not None else np.ones(29, dtype=np.float32)
        self.action_rescale = action_rescale
        
        # Video recording
        self.enable_video = enable_video
        self.video_path = video_path
        self.video_fps = video_fps
        self.frames = [] if enable_video else None
        self.renderer = None
        self.camera = None
        self.track_body_id = None
        if enable_video:
            try:
                self.renderer = mujoco.Renderer(self.mjm, height=480, width=640)
                
                # Try to find pelvis or root body for tracking
                try:
                    self.track_body_id = self.mjm.body("pelvis").id
                except:
                    try:
                        self.track_body_id = self.mjm.body("torso_link").id
                    except:
                        # Use root body (body 0)
                        self.track_body_id = 0
                
                # Set up tracking camera that follows the robot
                self.camera = mujoco.MjvCamera()
                self.camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                self.camera.trackbodyid = self.track_body_id
                self.camera.distance = 3.0  # Distance from tracked body
                self.camera.elevation = -20.0  # Camera elevation angle
                self.camera.azimuth = 45.0  # Camera azimuth angle
            except Exception as e:
                print(f"Warning: Could not create renderer for video: {e}")
                print("Video recording will be disabled.")
                self.enable_video = False
                self.frames = None
                self.camera = None
        
        # Internal state
        self.history_dof_pos = None  # History of dof_pos (4 timesteps)
        self.history_dof_vel = None  # History of dof_vel (4 timesteps)
        self.history_projected_gravity = None  # History of projected_gravity (4 timesteps)
        self.history_ang_vel = None  # History of ang_vel (4 timesteps)
        self.history_action = None  # History of actions (4 timesteps)
        self.last_action = None
        self.step_count = 0
        self.body_pos_prev = None
        self.body_quat_prev = None
        
        # Reset environment
        self.reset()
    
    def reset(self):
        """Reset the environment to initial state."""
        mujoco.mj_resetData(self.mjm, self.mjd)
        self.history_dof_pos = None
        self.history_dof_vel = None
        self.history_projected_gravity = None
        self.history_ang_vel = None
        self.history_action = None
        self.last_action = None
        self.step_count = 0
        
        # Reset video recording (clear frames for new recording)
        if self.enable_video:
            self.frames = []
        
        # Get initial observation
        obs = self._create_observation()
        return obs
    
    def step(self, action):
        """
        Step the environment with an action.
        
        Args:
            action: Action from policy (29 dim numpy array or torch tensor)
        
        Returns:
            observation: Current observation dict
            next_observation: Next observation dict (after step)
            reward: Reward (float, can be 0.0 if not computed)
            info: Info dict with additional information
        """
        # Convert action to numpy if needed
        if isinstance(action, torch.Tensor):
            action_np = action.cpu().numpy()
        else:
            action_np = np.array(action, dtype=np.float32)
        
        # Get current observation (before step)
        current_obs = self._create_observation()
        
        # Step simulation (with optional substeps)
        for i in (range(self.n_substeps)):
            # print(f"Step {i} of {self.n_substeps}")
            torques = self._compute_pd_torques(action_np)
        
            # Apply torques to MuJoCo
            self.mjd.ctrl[:29] = torques
            mujoco.mj_step(self.mjm, self.mjd)
        
        # Capture frame for video if enabled (with tracking camera)
        if self.enable_video and self.renderer is not None and self.camera is not None:
            # Update scene with current state and tracking camera
            self.renderer.update_scene(self.mjd, camera=self.camera)
            pixels = self.renderer.render()
            self.frames.append(pixels)
        
        # Update internal state (store action before creating next observation)
        self.last_action = action_np * self.action_rescale
        self.step_count += 1
        
        # Get next observation (after step)
        next_obs = self._create_observation()
        
        # Compute reward (placeholder - can be customized)
        reward = 0.0
        
        # Info dict
        info = {
            'step': self.step_count,
            'root_position': self.mjd.qpos[:3].copy(),
            'root_height': float(self.mjd.qpos[2]),
            'dt': self.sim_dt * self.n_substeps,  # Total timestep per control step
            'sim_dt': self.sim_dt,
            'time': float(self.mjd.time),
        }
        
        return current_obs, next_obs, reward, info
    
    def _compute_pd_torques(self, actions):
        """Compute PD control torques from actions."""
        # Scale actions: target = (action * action_scale * action_rescale) + default_pos
        # print( "action_scales", self.action_scales)
        # print( "action_rescale", self.action_rescale)
        actions_scaled = actions * self.action_scales * self.action_rescale
        target_pos = actions_scaled + self.default_joint_pos
        
        # Get current joint positions and velocities (skip freejoint)
        current_pos = self.mjd.qpos[7:7+29]  # Skip 7 freejoint DOFs
        current_vel = self.mjd.qvel[6:6+29]   # Skip 6 freejoint velocities
        
        # PD control: torque = Kp * (target - current) - Kd * velocity
        torques = self.kp_gains * (target_pos - current_pos) - self.kd_gains * current_vel
        return torques

    def _create_observation_backward(self):
        """Create observation dictionary from current MuJoCo state."""
        dof_pos = self.mjd.qpos[7:7+29].copy() - self.default_joint_pos # Skip 7 freejoint DOFs 
        dof_vel = self.mjd.qvel[6:6+29].copy()  # Skip 6 freejoint velocities
        
        # Get root orientation (quaternion) and compute projected gravity
        root_quat = self.mjd.qpos[3:7]  # Root quaternion [qw, qx, qy, qz]
        gravity_vec = np.array([0, 0, -1])  # Gravity vector in world frame
        
        # Rotate gravity to body frame (projected gravity)
        rot = Rotation.from_quat([root_quat[1], root_quat[2], root_quat[3], root_quat[0]])
        projected_gravity = rot.inv().apply(gravity_vec)
        
        # Get root angular velocity (3 dim)
        ang_vel = rot.apply(self.mjd.qvel[3:6].copy())  # [wx, wy, wz], scaled by 0.25 


        # 1. self.mjd.cvel[0, 3:6].copy()
        # 2. rot.apply(self.mjd.qvel[3:6].copy())
        # 3. self.mjd.cvel[0, 3:6].copy()


        # ang_vel = (self.mjd.qvel[3:6].copy())
        # print(f"ang_vel: {ang_vel}")
        # print(self.mjd.cvel[3:6].copy())
        # ang_vel = self.mjd.cvel[0, 3:6].copy()
        # ang_vel = ang_vel * 0
        # Construct state according to YAML config order:
        # [dof_pos_minus_default (29) + dof_vel (29) + projected_gravity (3) + base_ang_vel (3) + prev_actions (29)] = 93
        # But for "state" we only use: [dof_pos (29) + dof_vel (29) + projected_gravity (3) + ang_vel (3)] = 64
        state = torch.tensor(np.concatenate([dof_pos, dof_vel, projected_gravity, ang_vel]).astype(np.float32), dtype=torch.float32)

        privileged_state = self.get_privileged_state()
        new_obs = {
            "state": state.unsqueeze(0),
            "last_action": torch.tensor(dof_pos, dtype=torch.float32).unsqueeze(0) * 0,
            "privileged_state": torch.tensor(privileged_state, dtype=torch.float32).unsqueeze(0),
        }
        return new_obs
    
    def _create_observation(self):
        """Create observation dictionary from current MuJoCo state."""
        # Get DOF positions and velocities (29 DOFs)
        dof_pos = self.mjd.qpos[7:7+29].copy() - self.default_joint_pos # Skip 7 freejoint DOFs 
        dof_vel = self.mjd.qvel[6:6+29].copy()  # Skip 6 freejoint velocities
        
        # Get root orientation (quaternion) and compute projected gravity
        root_quat = self.mjd.qpos[3:7]  # Root quaternion [qw, qx, qy, qz]
        gravity_vec = np.array([0, 0, -1])  # Gravity vector in world frame
        
        # Rotate gravity to body frame (projected gravity)
        rot = Rotation.from_quat([root_quat[1], root_quat[2], root_quat[3], root_quat[0]])
        projected_gravity = rot.inv().apply(gravity_vec)
        
        # Get root angular velocity (3 dim)
        ang_vel = self.mjd.qvel[3:6].copy() * 0.25  # [wx, wy, wz], scaled by 0.25
        
        # Construct state according to YAML config order:
        # [dof_pos_minus_default (29) + dof_vel (29) + projected_gravity (3) + base_ang_vel (3) + prev_actions (29)] = 93
        # But for "state" we only use: [dof_pos (29) + dof_vel (29) + projected_gravity (3) + ang_vel (3)] = 64
        state = np.concatenate([dof_pos, dof_vel, projected_gravity, ang_vel]).astype(np.float32)
        
        # Initialize last_action if not provided
        if self.last_action is None:
            self.last_action = np.zeros(29, dtype=np.float32)
        
        # Update history for each component separately
        # History structure according to YAML config order:
        # [prev_actions_history (29*4=116) + base_ang_vel_history (3*4=12) + 
        #  dof_pos_minus_default_history (29*4=116) + dof_vel_history (29*4=116) + 
        #  projected_gravity_history (3*4=12)]
        # Total: 116 + 12 + 116 + 116 + 12 = 372
        
        if self.history_dof_pos is None:
            # Initialize all histories with zeros
            self.history_dof_pos = np.zeros((4, 29), dtype=np.float32)
            self.history_dof_vel = np.zeros((4, 29), dtype=np.float32)
            self.history_projected_gravity = np.zeros((4, 3), dtype=np.float32)
            self.history_ang_vel = np.zeros((4, 3), dtype=np.float32)
            self.history_action = np.zeros((4, 29), dtype=np.float32)
        else:
            # Shift all histories: move oldest to end, shift others right (most recent at index 0)
            self.history_dof_pos = np.roll(self.history_dof_pos, 1, axis=0)
            self.history_dof_vel = np.roll(self.history_dof_vel, 1, axis=0)
            self.history_projected_gravity = np.roll(self.history_projected_gravity, 1, axis=0)
            self.history_ang_vel = np.roll(self.history_ang_vel, 1, axis=0)
            self.history_action = np.roll(self.history_action, 1, axis=0)
        
        # Update most recent (index 0 after roll)
        self.history_dof_pos[0] = dof_pos
        self.history_dof_vel[0] = dof_vel
        self.history_projected_gravity[0] = projected_gravity
        self.history_ang_vel[0] = ang_vel
        self.history_action[0] = self.last_action
        
        # Concatenate histories according to YAML config order:
        # [prev_actions_history, base_ang_vel_history, dof_pos_minus_default_history, 
        #  dof_vel_history, projected_gravity_history]
        history = np.concatenate([
            self.history_action.reshape(-1),  # prev_actions_history: 4*29 = 116
            self.history_ang_vel.reshape(-1),  # base_ang_vel_history: 4*3 = 12
            self.history_dof_pos.reshape(-1),  # dof_pos_minus_default_history: 4*29 = 116
            self.history_dof_vel.reshape(-1),  # dof_vel_history: 4*29 = 116
            self.history_projected_gravity.reshape(-1),  # projected_gravity_history: 4*3 = 12
        ]).astype(np.float32)  # Total: 116 + 12 + 116 + 116 + 12 = 372
        
        # Privileged state (can be zeros or computed from model)
        privileged_state = np.zeros((463,), dtype=np.float32)
        
        obs = {
            "state": torch.tensor(state, dtype=torch.float32),
            "history_actor": torch.tensor(history, dtype=torch.float32),
            "last_action": torch.tensor(self.last_action.astype(np.float32), dtype=torch.float32),
            "privileged_state": torch.tensor(privileged_state, dtype=torch.float32),
        }
        
        return obs

    
    def save_video(self, video_path=None, fps=None):
        """
        Save recorded frames as a video file.
        
        Args:
            video_path: Path to save video. If None, uses self.video_path
            fps: Frames per second. If None, uses self.video_fps
        
        Returns:
            bool: True if video was saved successfully, False otherwise
        """
        if not self.enable_video or self.frames is None or len(self.frames) == 0:
            print("No frames recorded. Enable video recording first.")
            return False
        
        if not HAS_IMAGEIO:
            print("Error: imageio not installed. Install with: pip install imageio imageio-ffmpeg")
            return False
        
        video_path = video_path if video_path is not None else self.video_path
        fps = float(fps if fps is not None else self.video_fps)  # Ensure fps is a Python float
        
        # Check if ffmpeg is available for MP4
        file_ext = video_path.lower().split('.')[-1] if '.' in video_path else ''
        needs_ffmpeg = file_ext in ['mp4', 'avi', 'mov', 'mkv']
        
        if needs_ffmpeg:
            try:
                import imageio.plugins.ffmpeg
                has_ffmpeg = True
            except ImportError:
                has_ffmpeg = False
                print("Warning: FFMPEG plugin not available for MP4 format.")
                print("Attempting to save as GIF instead, or install with: pip install imageio[ffmpeg]")
                # Change extension to .gif
                video_path = video_path.rsplit('.', 1)[0] + '.gif'
                print(f"Will save as: {video_path}")
        
        try:
            print(f"Saving {len(self.frames)} frames to {video_path}...")
            imageio.mimsave(video_path, self.frames, fps=fps)
            print(f"Video saved successfully to {video_path}!")
            return True
        except Exception as e:
            error_msg = str(e)
            if 'ffmpeg' in error_msg.lower() or 'backend' in error_msg.lower():
                print(f"Error: {error_msg}")
                print("\nTo fix this, install FFMPEG support:")
                print("  pip install imageio[ffmpeg]")
                print("\nOr save as GIF format instead (no FFMPEG needed):")
                gif_path = video_path.rsplit('.', 1)[0] + '.gif'
                try:
                    print(f"Attempting to save as GIF: {gif_path}")
                    imageio.mimsave(gif_path, self.frames, fps=fps)
                    print(f"GIF saved successfully to {gif_path}!")
                    return True
                except Exception as e2:
                    print(f"Error saving GIF: {e2}")
                    return False
            else:
                print(f"Error saving video: {e}")
                return False
    
    def start_recording(self):
        """Start recording video frames."""
        if self.renderer is None:
            try:
                self.renderer = mujoco.Renderer(self.mjm, height=480, width=640)
            except Exception as e:
                print(f"Error creating renderer: {e}")
                return False
        
        self.enable_video = True
        self.frames = []
        print("Video recording started.")
        return True
    
    def stop_recording(self):
        """Stop recording video frames."""
        self.enable_video = False
        print(f"Video recording stopped. {len(self.frames) if self.frames else 0} frames captured.")

    def get_privileged_state(self):
        total_bodies = self.mjm.nbody
        valid_body_indices = []
        body_names_list = []
        head_link_idx = None
        head_link_name = None
        
        # Get all body names (make sure head is at the end)
        for i in range(total_bodies):
            try:
                body_name = mujoco.mj_id2name(self.mjm, mujoco.mjtObj.mjOBJ_BODY, i)
                if body_name:
                    # Exclude bodies starting with "dummy" or ending with "hand"
                    if not (body_name.startswith("dummy") or body_name.endswith("hand") or body_name.startswith("world")):
                        # Special handling for head_link: collect it separately to put at end
                        if body_name == "head_link":
                            head_link_idx = i
                            head_link_name = body_name
                        else:
                            valid_body_indices.append(i)
                            body_names_list.append(body_name)
                    else:
                        # Skip this body
                        continue
                else:
                    # Include unnamed bodies (might be root or important)
                    valid_body_indices.append(i)
                    body_names_list.append(f"body_{i}")
            except:
                # Include if we can't get name
                valid_body_indices.append(i)
                body_names_list.append(f"body_{i}")
        
        # Put head_link at the end if it exists
        if head_link_idx is not None:
            valid_body_indices.append(head_link_idx)
            body_names_list.append(head_link_name)
        
        num_bodies = len(valid_body_indices)
        valid_body_indices = np.array(valid_body_indices)
        
        # Get filtered body data
        body_pos = self.mjd.xpos[valid_body_indices, :].copy()  # (num_bodies, 3)
        body_quat = self.mjd.xquat[valid_body_indices, :].copy()  # (num_bodies, 4) [w, x, y, z]

        # use xpos minus xpos_prev to get body velocity, xpos is  (num_bodies, 3)
        # use previous quaternion to get body angular velocity
        if self.body_pos_prev is None:
            body_vel = np.zeros((num_bodies, 3))
            body_ang_vel = np.zeros((num_bodies, 3))
            self.body_pos_prev = body_pos
            self.body_quat_prev = body_quat
        else:
            body_vel = (body_pos - self.body_pos_prev) / (self.sim_dt * self.n_substeps)
            body_ang_vel = calc_angular_velocity(body_quat, self.body_quat_prev, self.sim_dt * self.n_substeps)
            # print(body_vel.shape)
            # print(body_ang_vel.shape)
            self.body_pos_prev = body_pos
            self.body_quat_prev = body_quat
        # Convert to torch tensors and add batch dimension
        body_pos_t = torch.tensor(body_pos, dtype=torch.float32).unsqueeze(0)  # (1, num_bodies, 3)
        # Convert quaternion from [w, x, y, z] to [x, y, z, w] format
        body_rot_t = torch.tensor(body_quat[:, [1, 2, 3, 0]], dtype=torch.float32).unsqueeze(0)  # (1, num_bodies, 4)
        body_vel_t = torch.tensor(body_vel, dtype=torch.float32).unsqueeze(0)  # (1, num_bodies, 3)
        body_ang_vel_t = torch.tensor(body_ang_vel, dtype=torch.float32).unsqueeze(0)  # (1, num_bodies, 3)
        
        # Compute privileged state using compute_humanoid_observations_max logic
        # print(f"body_pos_t shape: {body_pos_t}")
        root_pos = body_pos_t[:, 0:1, :]  # (1, 1, 3)
        root_rot = body_rot_t[:, 0:1, :]  # (1, 1, 4)
        
        # Calculate heading rotation inverse
        heading_rot_inv = calc_heading_quat_inv(root_rot, w_last=True)  # (1, 1, 4)
        heading_rot_inv_expand = heading_rot_inv.repeat(1, num_bodies, 1)  # (1, num_bodies, 4)
        flat_heading_rot_inv = heading_rot_inv_expand.reshape(-1, 4)  # (num_bodies, 4)
        
        # Local body positions
        root_pos_expand = root_pos.repeat(1, num_bodies, 1)  # (1, num_bodies, 3)
        local_body_pos = body_pos_t - root_pos_expand  # (1, num_bodies, 3)
        flat_local_body_pos = local_body_pos.reshape(-1, 3)  # (num_bodies, 3)
        flat_local_body_pos = quat_rotate(flat_heading_rot_inv, flat_local_body_pos)  # (num_bodies, 3)
        local_body_pos_obs = flat_local_body_pos.reshape(1, -1)  # (1, num_bodies*3)
        local_body_pos_obs = local_body_pos_obs[..., 3:]  # Remove root pos: (1, (num_bodies-1)*3)
        
        # Local body rotations
        flat_body_rot = body_rot_t.reshape(-1, 4)  # (num_bodies, 4)
        flat_local_body_rot = quat_mul(flat_heading_rot_inv, flat_body_rot, w_last=True)  # (num_bodies, 4)
        flat_local_body_rot_obs = quat_to_tan_norm(flat_local_body_rot, w_last=True)  # (num_bodies, 6)
        local_body_rot_obs = flat_local_body_rot_obs.reshape(1, -1)  # (1, num_bodies*6)

        # Local body velocities
        flat_body_vel = body_vel_t.reshape(-1, 3)  # (num_bodies, 3)
        flat_local_body_vel = quat_rotate(flat_heading_rot_inv, flat_body_vel)  # (num_bodies, 3)
        local_body_vel_obs = flat_local_body_vel.reshape(1, -1)  # (1, num_bodies*3)
        
        # Local body angular velocities
        flat_body_ang_vel = body_ang_vel_t.reshape(-1, 3)  # (num_bodies, 3)
        flat_local_body_ang_vel = quat_rotate(flat_heading_rot_inv, flat_body_ang_vel)  # (num_bodies, 3)
        local_body_ang_vel_obs = flat_local_body_ang_vel.reshape(1, -1)  # (1, num_bodies*3)
        
        # Get root height for privileged state (if needed)
        root_h = root_pos[:, :, 2:3].squeeze(0)  # (1, 1) - root height
        
        # Concatenate privileged state components (max_local_self_obs format)
        # Format: [local_body_pos (all bodies including root), local_body_rot, local_body_vel, local_body_ang_vel, root_height]
        privileged_state = torch.cat([
            root_h,       # 1 (root height)
            local_body_pos_obs,      # (num_bodies-1)*3 (includes root position, which is [0,0,0] in local frame)
            local_body_rot_obs,      # num_bodies*6
            local_body_vel_obs,      # num_bodies*3
            local_body_ang_vel_obs,  # num_bodies*3
        ], dim=-1).squeeze(0).numpy()  # Convert to numpy

        return privileged_state

    def set_state(self, dof_positions, dof_velocities=None, root_quat=None, root_pos=None, root_vel=None, root_ang_vel=None):
        """Set the state of the environment."""
        if root_quat is None:
            root_quat = np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion

        if root_pos is None:
            root_pos = np.array([0.0, 0.0, 0.8])  # Default height

        if root_vel is None:
            root_vel = np.array([0.0, 0.0, 0.0])

        if root_ang_vel is None:
            root_ang_vel = np.array([0.0, 0.0, 0.0])

        rot = Rotation.from_quat([root_quat[1], root_quat[2], root_quat[3], root_quat[0]])
        local_root_ang_vel = rot.inv().apply(root_ang_vel)

        self.mjd.qpos[0:3] = root_pos
        self.mjd.qpos[3:7] = root_quat
        self.mjd.qpos[7:7+29] = dof_positions
        self.mjd.qvel[0:3] = root_vel
        self.mjd.qvel[3:6] = local_root_ang_vel
        if dof_velocities is None:
            self.mjd.qvel[6:6+29] = 0.0
        else:
            self.mjd.qvel[6:6+29] = dof_velocities

        mujoco.mj_forward(self.mjm, self.mjd)

    def get_image(self):
        """Get the image of the environment."""
        self.renderer.update_scene(self.mjd, camera=self.camera)
        if self.renderer is not None:
            return self.renderer.render()
        else:
            return None
        
def calc_angular_velocity(quat_cur, quat_prev, dt):
    """
    Calculate angular velocity from two quaternions.
    
    Supports (4,) and (num_bodies, 4) shape for inputs. Outputs (3,) or (num_bodies, 3) accordingly.
    Both quaternions must be in [w, x, y, z] format.
    """
    from scipy.spatial.transform import Rotation as R
    quat_cur = np.asarray(quat_cur)
    quat_prev = np.asarray(quat_prev)

    # Convert to (N,4) for broadcasting
    orig_shape = quat_cur.shape
    if quat_cur.ndim == 1:
        quat_cur = quat_cur[None, :]
        quat_prev = quat_prev[None, :]

    # Convert [w, x, y, z] -> [x, y, z, w]
    quat_cur_xyzw = np.stack([quat_cur[:,1], quat_cur[:,2], quat_cur[:,3], quat_cur[:,0]], axis=-1)
    quat_prev_xyzw = np.stack([quat_prev[:,1], quat_prev[:,2], quat_prev[:,3], quat_prev[:,0]], axis=-1)

    rot_cur = R.from_quat(quat_cur_xyzw)
    rot_prev = R.from_quat(quat_prev_xyzw)
    delta_rot = rot_prev.inv() * rot_cur
    rotvec = delta_rot.as_rotvec()
    angular_velocity = rotvec / dt

    if orig_shape == (4,):
        return angular_velocity[0]
    else:
        return angular_velocity