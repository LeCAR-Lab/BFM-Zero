#!/usr/bin/env python3
"""
Common ZMQ communication configuration for motion data visualization.
Defines ports and message formats for body poses and joint states.
"""

UNITREE_LEGGED_CONST = dict(
  HIGHLEVEL = 0xEE,
  LOWLEVEL = 0xFF,
  TRIGERLEVEL = 0xF0,
  PosStopF = 2146000000.0,
  VelStopF = 16000.0,
  MODE_MACHINE = 5,
  MODE_PR = 0,
)


import zmq
import numpy as np
import struct

# ZMQ Port Configuration
PORTS = {
    "joint_names": 5550,
    "body_names": 5551,
    "torso_link_pose": 5552, # for T1
    'pelvis_pose': 5555, # for G1
    'box_pose': 5556,
    'joint_pos': 5559,
    'joint_vel': 5560,  # Reserved for future use
    "suitcase_pose": 5561,
    "plasticbox_pose": 5562,
    "stool_pose": 5563,
    "ball_pose": 5564,
    "foldchair_pose": 5565,
    "foldchair_joint_pos": 5566,
    "door_pose": 5567,
    "door_panel_pose": 5568,
    "door_joint_pos": 5569,
    "box_small_pose": 5570,
    "box_target_pose": 5571,
    "stool_low_pose": 5564,
    "foam_pose": 5565,
    "bread_box_pose": 5566,
    "stair_pose": 5572,
    "low_state": 5590,
    "low_cmd": 5591,
}

class PoseMessage:
    """Message format for body pose (position + quaternion)"""
    def __init__(self, position: np.ndarray, quaternion: np.ndarray):
        """
        Args:
            position: 3D position [x, y, z]
            quaternion: Quaternion [w, x, y, z]
        """
        self.position = np.array(position, dtype=np.float32)
        self.quaternion = np.array(quaternion, dtype=np.float32)
    
    def to_bytes(self) -> bytes:
        """Convert to binary format for ZMQ transmission"""
        # Pack as 7 float32 values: [px, py, pz, qw, qx, qy, qz]
        data = np.concatenate([self.position, self.quaternion]).astype(np.float32)
        return data.tobytes()
    
    @classmethod
    def from_bytes(cls, data: bytes) -> 'PoseMessage':
        """Create from binary data"""
        values = np.frombuffer(data, dtype=np.float32)
        if len(values) != 7:
            raise ValueError(f"Expected 7 float32 values, got {len(values)}")
        return cls(values[:3], values[3:])

class JointStateMessage:
    """Message format for joint state (positions and optionally velocities)"""
    def __init__(self, positions: np.ndarray, velocities: np.ndarray | None = None):
        """
        Args:
            positions: Joint positions array
            velocities: Joint velocities array (optional)
        """
        self.positions = np.array(positions, dtype=np.float32)
        self.velocities = np.array(velocities, dtype=np.float32) if velocities is not None else None
    
    def to_bytes(self) -> bytes:
        """Convert to binary format for ZMQ transmission"""
        # Pack header with number of positions and whether velocities are included
        header = struct.pack('II', len(self.positions), 1 if self.velocities is not None else 0)
        
        # Pack positions
        pos_data = self.positions.astype(np.float32).tobytes()
        
        # Pack velocities if available
        if self.velocities is not None:
            vel_data = self.velocities.astype(np.float32).tobytes()
            return header + pos_data + vel_data
        else:
            return header + pos_data
    
    @classmethod
    def from_bytes(cls, data: bytes) -> 'JointStateMessage':
        """Create from binary data"""
        # Unpack header
        num_positions, has_velocities = struct.unpack('II', data[:8])
        
        # Extract positions
        pos_size = num_positions * 4  # 4 bytes per float32
        positions = np.frombuffer(data[8:8+pos_size], dtype=np.float32)
        
        # Extract velocities if present
        velocities = None
        if has_velocities:
            velocities = np.frombuffer(data[8+pos_size:8+pos_size*2], dtype=np.float32)
        
        return cls(positions, velocities)

class ZMQPublisher:
    """ZMQ Publisher wrapper"""
    def __init__(self, port: int):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(f"tcp://*:{port}")
        
    def publish_pose(self, position: np.ndarray, quaternion: np.ndarray):
        """Publish a pose message"""
        msg = PoseMessage(position, quaternion)
        self.socket.send(msg.to_bytes())

    def publish_joint_state(self, positions: np.ndarray, velocities: np.ndarray | None = None):
        """Publish joint state message"""
        msg = JointStateMessage(positions, velocities)
        self.socket.send(msg.to_bytes())
    
    def publish_names(self, joint_names: list[str]):
        """Publish a list of joint names"""
        # Convert list to bytes
        names_bytes = '\n'.join(joint_names).encode('utf-8')
        self.socket.send(names_bytes)
    
    def close(self):
        """Close the publisher"""
        self.socket.close()
        self.context.term()

class ZMQSubscriber:
    """ZMQ Subscriber wrapper"""
    def __init__(self, port: int, ip: str = "localhost"):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.setsockopt(zmq.CONFLATE, 1)
        self.socket.connect(f"tcp://{ip}:{port}")
        self.socket.setsockopt(zmq.SUBSCRIBE, b"")  # Subscribe to all messages
        self.socket.setsockopt(zmq.RCVTIMEO, 10)  # 10ms timeout
        
    def receive_pose(self) -> PoseMessage | None:
        """Receive a pose message"""
        try:
            data = self.socket.recv()
            return PoseMessage.from_bytes(data)
        except zmq.Again:
            return None  # No message available
        except Exception as e:
            print(f"Error receiving pose: {e}")
            return None
    
    def receive_joint_state(self) -> JointStateMessage | None:
        """Receive joint state message"""
        try:
            data = self.socket.recv()
            return JointStateMessage.from_bytes(data)
        except zmq.Again:
            return None  # No message available
        except Exception as e:
            print(f"Error receiving joint state: {e}")
            return None
    
    def receive_names(self) -> list[str] | None:
        """Receive a list of joint names"""
        try:
            data = self.socket.recv()
            return data.decode('utf-8').split('\n')
        except zmq.Again:
            return None  # No message available
        except Exception as e:
            print(f"Error receiving joint names: {e}")
            return None

    def close(self):
        """Close the subscriber"""
        self.socket.close()
        self.context.term()


class LowCmdMessage:
    """Binary message containing joint-space command targets."""

    def __init__(
        self,
        q_target: np.ndarray,
        dq_target: np.ndarray,
        tau_ff: np.ndarray,
        kp: np.ndarray,
        kd: np.ndarray,
    ):
        arrays = [
            np.asarray(q_target, dtype=np.float32),
            np.asarray(dq_target, dtype=np.float32),
            np.asarray(tau_ff, dtype=np.float32),
            np.asarray(kp, dtype=np.float32),
            np.asarray(kd, dtype=np.float32),
        ]

        length = arrays[0].size
        if any(arr.size != length for arr in arrays[1:]):
            raise ValueError("All arrays in LowCmdMessage must have the same length")

        self.q_target = arrays[0]
        self.dq_target = arrays[1]
        self.tau_ff = arrays[2]
        self.kp = arrays[3]
        self.kd = arrays[4]

    def to_bytes(self) -> bytes:
        count = self.q_target.size
        header = struct.pack('<I', count)
        payload = b''.join(
            arr.astype(np.float32, copy=False).tobytes()
            for arr in (self.q_target, self.dq_target, self.tau_ff, self.kp, self.kd)
        )
        return header + payload

    @classmethod
    def from_bytes(cls, data: bytes) -> 'LowCmdMessage':
        if len(data) < 4:
            raise ValueError("LowCmdMessage data is too short")

        (count,) = struct.unpack('<I', data[:4])
        offset = 4
        segment_size = count * 4
        arrays = []
        for _ in range(5):
            end = offset + segment_size
            if end > len(data):
                raise ValueError("LowCmdMessage data is incomplete")
            arrays.append(np.frombuffer(data[offset:end], dtype=np.float32).copy())
            offset = end

        return cls(*arrays)


class LowStateMessage:
    """Binary message containing base and joint state information."""

    def __init__(
        self,
        quaternion: np.ndarray,
        gyroscope: np.ndarray,
        joint_positions: np.ndarray,
        joint_velocities: np.ndarray,
        joint_torques: np.ndarray | None = None,
        tick: int = 0,
    ):
        self.quaternion = np.asarray(quaternion, dtype=np.float32)
        if self.quaternion.size != 4:
            raise ValueError("Quaternion must have exactly 4 elements")

        self.gyroscope = np.asarray(gyroscope, dtype=np.float32)
        if self.gyroscope.size != 3:
            raise ValueError("Gyroscope must have exactly 3 elements")

        self.joint_positions = np.asarray(joint_positions, dtype=np.float32)
        self.joint_velocities = np.asarray(joint_velocities, dtype=np.float32)
        if self.joint_positions.size != self.joint_velocities.size:
            raise ValueError("Joint position and velocity arrays must match in length")

        if joint_torques is not None:
            joint_torques = np.asarray(joint_torques, dtype=np.float32)
            if joint_torques.size != self.joint_positions.size:
                raise ValueError("Joint torque array must match joint positions length")
        self.joint_torques = joint_torques
        self.tick = int(tick)

    def to_bytes(self) -> bytes:
        count = self.joint_positions.size
        has_torque = 1 if self.joint_torques is not None else 0
        header = struct.pack('<III', count, self.tick, has_torque)

        payload_parts = [
            self.quaternion.astype(np.float32, copy=False).tobytes(),
            self.gyroscope.astype(np.float32, copy=False).tobytes(),
            self.joint_positions.astype(np.float32, copy=False).tobytes(),
            self.joint_velocities.astype(np.float32, copy=False).tobytes(),
        ]

        if self.joint_torques is not None:
            payload_parts.append(self.joint_torques.astype(np.float32, copy=False).tobytes())

        return header + b''.join(payload_parts)

    @classmethod
    def from_bytes(cls, data: bytes) -> 'LowStateMessage':
        header_size = struct.calcsize('<III')
        if len(data) < header_size + 28:  # header + quaternion(16) + gyro(12)
            raise ValueError("LowStateMessage data is too short")

        count, tick, has_torque = struct.unpack('<III', data[:header_size])
        offset = header_size

        quat_end = offset + 16
        gyro_end = quat_end + 12
        quaternion = np.frombuffer(data[offset:quat_end], dtype=np.float32).copy()
        gyroscope = np.frombuffer(data[quat_end:gyro_end], dtype=np.float32).copy()

        segment_size = count * 4
        pos_end = gyro_end + segment_size
        vel_end = pos_end + segment_size
        if vel_end > len(data):
            raise ValueError("LowStateMessage joint data is incomplete")

        joint_positions = np.frombuffer(data[gyro_end:pos_end], dtype=np.float32).copy()
        joint_velocities = np.frombuffer(data[pos_end:vel_end], dtype=np.float32).copy()

        joint_torques = None
        if has_torque:
            torque_end = vel_end + segment_size
            if torque_end > len(data):
                raise ValueError("LowStateMessage torque data is incomplete")
            joint_torques = np.frombuffer(data[vel_end:torque_end], dtype=np.float32).copy()

        return cls(
            quaternion=quaternion,
            gyroscope=gyroscope,
            joint_positions=joint_positions,
            joint_velocities=joint_velocities,
            joint_torques=joint_torques,
            tick=tick,
        )