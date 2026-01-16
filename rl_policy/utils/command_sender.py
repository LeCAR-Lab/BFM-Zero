import time
import numpy as np
import zmq

from utils.strings import resolve_matching_names_values
from utils.strings import unitree_joint_names
from utils.common import LowCmdMessage, PORTS


class CommandSender:
    def __init__(self, robot_config, policy_config):
        self.robot_type = robot_config["ROBOT_TYPE"]
        if self.robot_type == "g1_real":
            self.robot = robot_config["robot"]
        else:
            supported_types = {
                "h1",
                "go2",
                "g1_29dof",
                "h1-2_21dof",
                "h1-2_27dof",
            }
            if self.robot_type not in supported_types:
                raise NotImplementedError(
                    f"Robot type {self.robot_type} is not supported yet"
                )

        # init robot and kp kd
        self._kp_level = 1.0  # 0.1

        self.policy_config = policy_config
        joint_kp_dict = self.policy_config["joint_kp"]
        joint_indices, joint_names, joint_kp = resolve_matching_names_values(
            joint_kp_dict,
            unitree_joint_names,
            preserve_order=True,
            strict=False,
        )
        self.joint_kp_unitree_default = np.zeros(len(unitree_joint_names))
        self.joint_kp_unitree_default[joint_indices] = joint_kp
        self.joint_kp_unitree = self.joint_kp_unitree_default.copy()

        joint_kd_dict = self.policy_config["joint_kd"]
        joint_indices, joint_names, joint_kd = resolve_matching_names_values(
            joint_kd_dict,
            unitree_joint_names,
            preserve_order=True,
            strict=False,
        )
        self.joint_kd_unitree = np.zeros(len(unitree_joint_names))
        self.joint_kd_unitree[joint_indices] = joint_kd

        default_joint_pos_dict = self.policy_config["default_joint_pos"]
        joint_indices, joint_names, default_joint_pos = resolve_matching_names_values(
            default_joint_pos_dict,
            unitree_joint_names,
            preserve_order=True,
            strict=False,
        )
        self.default_joint_pos_unitree = np.zeros(len(unitree_joint_names))
        self.default_joint_pos_unitree[joint_indices] = default_joint_pos

        joint_names_isaac = self.policy_config["isaac_joint_names"]
        self.joint_indices_unitree = [unitree_joint_names.index(name) for name in joint_names_isaac]

        # init low cmd publisher
        if self.robot_type != "g1_real":
            self.zmq_context = zmq.Context.instance()
            self.low_cmd_port = robot_config.get(
                "LOW_CMD_PORT", PORTS.get("low_cmd", 55901)
            )
            bind_addr = robot_config.get("LOW_CMD_BIND_ADDR", "*")
            bind_endpoint = f"tcp://{bind_addr}:{self.low_cmd_port}"

            self.lowcmd_socket: zmq.Socket = self.zmq_context.socket(zmq.PUB)
            self.lowcmd_socket.setsockopt(zmq.SNDHWM, 1)
            self.lowcmd_socket.setsockopt(zmq.LINGER, 0)
            self.lowcmd_socket.bind(bind_endpoint)
            # Give subscribers time to connect before sending commands
            time.sleep(0.1)
        else:
            self.lowcmd_socket = None

        self.InitLowCmd()

    @property
    def kp_level(self):
        return self._kp_level

    @kp_level.setter
    def kp_level(self, value):
        self._kp_level = value
        self.joint_kp_unitree[:] = self.joint_kp_unitree_default * self._kp_level

    def InitLowCmd(self):
        self.cmd_q = np.zeros(len(unitree_joint_names))
        self.cmd_dq = np.zeros(len(unitree_joint_names))
        self.cmd_tau = np.zeros(len(unitree_joint_names))

        self.cmd_q[:] = self.default_joint_pos_unitree

    def send_command(self, cmd_q, cmd_dq, cmd_tau):
        if self.robot_type != "g1_real":
            self.cmd_q[self.joint_indices_unitree] = cmd_q
            self.cmd_dq[self.joint_indices_unitree] = cmd_dq
            self.cmd_tau[self.joint_indices_unitree] = cmd_tau
            
            message = LowCmdMessage(
                q_target=self.cmd_q,
                dq_target=self.cmd_dq,
                tau_ff=self.cmd_tau,
                kp=self.joint_kp_unitree,
                kd=self.joint_kd_unitree,
            )
            try:
                self.lowcmd_socket.send(message.to_bytes(), flags=zmq.DONTWAIT)
            except zmq.Again:
                pass
        else:
            cmd = self.robot.create_zero_command()

            # Apply kp_level scaling (kd remains constant, consistent with original implementation)
            kp_scaled = self.joint_kp_unitree * self._kp_level
            kd_scaled = self.joint_kd_unitree

            q_target = list(cmd.q_target)
            dq_target = list(cmd.dq_target)
            tau_ff = list(cmd.tau_ff)
            kp = list(cmd.kp)
            kd = list(cmd.kd)
            for i_policy, idx_unitree in enumerate(self.joint_indices_unitree):
                q_target[idx_unitree] = float(cmd_q[i_policy])
                dq_target[idx_unitree] = float(cmd_dq[i_policy])
                tau_ff[idx_unitree] = float(cmd_tau[i_policy])
                kp[idx_unitree] = float(kp_scaled[idx_unitree])
                kd[idx_unitree] = float(kd_scaled[idx_unitree])

            cmd.q_target = q_target
            cmd.dq_target = dq_target
            cmd.tau_ff = tau_ff
            cmd.kp = kp
            cmd.kd = kd

            self.robot.write_low_command(cmd)