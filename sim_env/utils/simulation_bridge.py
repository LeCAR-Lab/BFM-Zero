import numpy as np
import glfw
import mujoco
import zmq

from loguru import logger

import sys
sys.path.append(".")
from utils.strings import resolve_matching_names_values, unitree_joint_names
from utils.math import quat_mul, quat_conjugate, yaw_quat
from utils.common import LowStateMessage, LowCmdMessage, PORTS


class SimulationBridge:

    def __init__(
        self,
        mj_model: mujoco.MjModel,
        mj_data: mujoco.MjData,
        robot_config: dict,
        scene_config: dict,
    ):
        self.robot_config = robot_config
        self.scene_config = scene_config
        robot_type = robot_config["ROBOT_TYPE"]
        supported_types = {
            "g1_29dof",
            "h1",
            "go2",
            "h1-2_21dof",
            "h1-2_27dof",
        }
        if robot_type not in supported_types:
            raise ValueError(
                f"Invalid robot type '{robot_type}'. Expected one of {supported_types}."
            )
        self.mj_model = mj_model
        self.mj_data = mj_data

        self.torques = np.zeros(self.mj_model.nu)

        # ZMQ communication setup
        self.zmq_context = zmq.Context.instance()

        self.low_state_port = robot_config.get(
            "LOW_STATE_PORT", PORTS.get("low_state", 5590)
        )
        low_state_bind_addr = robot_config.get("LOW_STATE_BIND_ADDR", "*")
        low_state_endpoint = f"tcp://{low_state_bind_addr}:{self.low_state_port}"
        self.low_state_pub = self.zmq_context.socket(zmq.PUB)
        self.low_state_pub.setsockopt(zmq.SNDHWM, 1)
        self.low_state_pub.setsockopt(zmq.LINGER, 0)
        self.low_state_pub.bind(low_state_endpoint)

        self.low_cmd_port = robot_config.get(
            "LOW_CMD_PORT", PORTS.get("low_cmd", 5591)
        )
        low_cmd_host = robot_config.get("LOW_CMD_HOST", "127.0.0.1")
        low_cmd_endpoint = f"tcp://{low_cmd_host}:{self.low_cmd_port}"
        self.low_cmd_sub = self.zmq_context.socket(zmq.SUB)
        self.low_cmd_sub.setsockopt(zmq.SUBSCRIBE, b"")
        self.low_cmd_sub.setsockopt(zmq.CONFLATE, 1)
        self.low_cmd_sub.setsockopt(zmq.RCVTIMEO, 0)
        self.low_cmd_sub.setsockopt(zmq.LINGER, 0)
        self.low_cmd_sub.connect(low_cmd_endpoint)

        total_joints = len(unitree_joint_names)
        self.cmd_q = np.zeros(total_joints, dtype=np.float32)
        self.cmd_dq = np.zeros(total_joints, dtype=np.float32)
        self.cmd_tau = np.zeros(total_joints, dtype=np.float32)
        self.cmd_kp = np.zeros(total_joints, dtype=np.float32)
        self.cmd_kd = np.zeros(total_joints, dtype=np.float32)
        self.has_received_command = False

        self.init_joint_indices()

    def init_joint_indices(self):
        joint_names_mujoco = [
            self.mj_model.joint(i).name for i in range(self.mj_model.njnt)
        ]
        actuator_names_mujoco = [
            self.mj_model.actuator(i).name + "_joint" for i in range(self.mj_model.nu)
        ]
        self.joint_indices_unitree = []
        self.qpos_adrs = []
        self.qvel_adrs = []
        self.act_adrs = []

        shared_joint_names = set(joint_names_mujoco) & set(unitree_joint_names)
        for name in shared_joint_names:
            print(f"shared_joint_names: {name}")
            self.joint_indices_unitree.append(unitree_joint_names.index(name))

            joint_idx = joint_names_mujoco.index(name)
            self.qpos_adrs.append(self.mj_model.jnt_qposadr[joint_idx])
            self.qvel_adrs.append(self.mj_model.jnt_dofadr[joint_idx])
            # print(f"actuator_names_mujoco: {shared_joint_names}")
            self.act_adrs.append(actuator_names_mujoco.index(name))
        
        if "floating_base_joint" in joint_names_mujoco:
            root_joint_idx = joint_names_mujoco.index("floating_base_joint")
        elif "pelvis_root" in joint_names_mujoco:
            root_joint_idx = joint_names_mujoco.index("pelvis_root")
        else:
            raise ValueError("No root joint found in the MuJoCo model.")
        self.root_qpos_adr = self.mj_model.jnt_qposadr[root_joint_idx]
        self.root_qvel_adr = self.mj_model.jnt_dofadr[root_joint_idx]

        joint_effort_limit_dict = self.robot_config["joint_effort_limit"]
        joint_indices, joint_names_matched, joint_effort_limit = (
            resolve_matching_names_values(
                joint_effort_limit_dict,
                joint_names_mujoco,
                preserve_order=True,
                strict=False,
            )
        )
        self.joint_effort_limit_mjc = np.array(joint_effort_limit)
        self.joint_idx_in_ctrl = np.array(
            [actuator_names_mujoco.index(name) for name in joint_names_matched]
        )

    def compute_torques(self):
        self.torques[:] = 0.0
        self._poll_low_cmd()

        if self.has_received_command:
            for unitree_idx, qpos_addr, qvel_addr, act_addr in zip(
                self.joint_indices_unitree,
                self.qpos_adrs,
                self.qvel_adrs,
                self.act_adrs,
            ):
                q_des = self.cmd_q[unitree_idx]
                dq_des = self.cmd_dq[unitree_idx]
                tau_ff = self.cmd_tau[unitree_idx]
                kp = self.cmd_kp[unitree_idx]
                kd = self.cmd_kd[unitree_idx]

                self.torques[act_addr] = (
                    tau_ff
                    + kp * (q_des - self.mj_data.qpos[qpos_addr])
                    + kd * (dq_des - self.mj_data.qvel[qvel_addr])
                )
        # Set the torque limit
        self.torques[self.joint_idx_in_ctrl] = np.clip(
            self.torques[self.joint_idx_in_ctrl],
            -self.joint_effort_limit_mjc,
            self.joint_effort_limit_mjc,
        )

    def _poll_low_cmd(self):
        """Non-blocking command subscriber that keeps the most recent message."""
        if self.low_cmd_sub is None:
            return

        updated = False
        while True:
            try:
                data = self.low_cmd_sub.recv(flags=zmq.DONTWAIT)
            except zmq.Again:
                break

            try:
                low_cmd = LowCmdMessage.from_bytes(data)
            except Exception as exc:
                logger.warning(f"Failed to decode low command message: {exc}")
                continue

            if low_cmd.q_target.size != len(unitree_joint_names):
                logger.warning(
                    "Received low command with unexpected size {}",
                    low_cmd.q_target.size,
                )
                continue

            self.cmd_q[:] = low_cmd.q_target
            self.cmd_dq[:] = low_cmd.dq_target
            self.cmd_tau[:] = low_cmd.tau_ff
            self.cmd_kp[:] = low_cmd.kp
            self.cmd_kd[:] = low_cmd.kd
            updated = True

        if updated:
            self.has_received_command = True

    def publish_low_state(self):
        if self.mj_data == None:
            return

        joint_pos_partial = self.mj_data.qpos[self.qpos_adrs]
        joint_vel_partial = self.mj_data.qvel[self.qvel_adrs]
        joint_torque_partial = self.mj_data.actuator_force[self.act_adrs]

        joint_pos_full = np.zeros(len(unitree_joint_names), dtype=np.float32)
        joint_vel_full = np.zeros(len(unitree_joint_names), dtype=np.float32)
        joint_tau_full = np.zeros(len(unitree_joint_names), dtype=np.float32)
        for mjc_idx, unitree_idx in enumerate(self.joint_indices_unitree):
            joint_pos_full[unitree_idx] = joint_pos_partial[mjc_idx]
            joint_vel_full[unitree_idx] = joint_vel_partial[mjc_idx]
            joint_tau_full[unitree_idx] = joint_torque_partial[mjc_idx]

        # quaternion: w, x, y, z
        root_quat_w = self.mj_data.qpos[self.root_qpos_adr + 3:self.root_qpos_adr+7]
        root_quat_yaw_w = yaw_quat(root_quat_w)
        root_quat_b = quat_mul(quat_conjugate(root_quat_yaw_w), root_quat_w)

        # angular velocity: x, y, z
        root_ang_vel_b = self.mj_data.qvel[self.root_qvel_adr + 3:self.root_qvel_adr+6]
        low_state_msg = LowStateMessage(
            quaternion=root_quat_b,
            gyroscope=root_ang_vel_b,
            joint_positions=joint_pos_full,
            joint_velocities=joint_vel_full,
            joint_torques=joint_tau_full,
            tick=int(self.mj_data.time * 1e3),
        )

        try:
            self.low_state_pub.send(low_state_msg.to_bytes(), flags=zmq.DONTWAIT)
        except zmq.Again:
            pass