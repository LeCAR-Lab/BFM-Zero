"""This script is used to visualize the motion data in MuJoCo

It subscribes to the joint positions and root pose from ZMQ publishers and updates the MuJoCo model.
"""

import numpy as np
import time
import threading
import argparse
import sched
import os
from datetime import datetime

import zmq

import mujoco
import mujoco.viewer

import sys
sys.path.append(".")

from utils.strings import unitree_joint_names

scene = "./data/robots/g1/scene_29dof_nohand.xml"  # Robot scene, for Sim2Sim

temp_scene_base_name = "scene_temp.xml"

class MuJoCoMocapViewer:
    def __init__(
        self,
        frequency: int = 50,
        mujoco_model_path: str = scene,
        joint_pos_ip: str = "10.42.0.129",
        joint_pos_port: int = 5555,
        root_pose_ip: str = "localhost",
        root_pose_port: int = 5556,
        record: bool = False,
    ):
        self.record = record
        self.freq = frequency
        self.update_dt = 1.0 / self.freq
        
        # Initialize MuJoCo model and viewer
        self.model = mujoco.MjModel.from_xml_path(mujoco_model_path)
        self.data = mujoco.MjData(self.model)
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data, show_left_ui=False, show_right_ui=False)
        model_body_names = [self.model.body(i).name for i in range(self.model.nbody)]
        pelvis_body_id = model_body_names.index("pelvis")
        self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        self.viewer.cam.trackbodyid = pelvis_body_id

        # Initialize ZMQ context
        self.zmq_context = zmq.Context()
        
        # Initialize joint position subscriber (remote machine)
        self.joint_pos_subscriber = self.zmq_context.socket(zmq.SUB)
        self.joint_pos_subscriber.connect(f"tcp://{joint_pos_ip}:{joint_pos_port}")
        self.joint_pos_subscriber.setsockopt(zmq.SUBSCRIBE, b"joint_pos")
        self.joint_pos_subscriber.setsockopt(zmq.RCVTIMEO, 100)  # 100ms timeout

        
        # Initialize root pose subscriber (local machine)
        self.root_pose_subscriber = self.zmq_context.socket(zmq.SUB)
        self.root_pose_subscriber.connect(f"tcp://{root_pose_ip}:{root_pose_port}")
        self.root_pose_subscriber.setsockopt(zmq.SUBSCRIBE, "pelvis".encode('utf-8'))
        self.root_pose_subscriber.setsockopt(zmq.RCVTIMEO, 100)  # 100ms timeout
        
        print(f"Connected to joint_pos publisher at {joint_pos_ip}:{joint_pos_port}")
        print(f"Connected to root_pose publisher at {root_pose_ip}:{root_pose_port}")
        
        # Get joint IDs and addresses
        mujoco_joint_names = [self.model.joint(i).name for i in range(self.model.njnt)]
        tgt_joint_names = mujoco_joint_names
        src_joint_names = unitree_joint_names
        shared_joint_names = list(sorted(set(tgt_joint_names) & set(src_joint_names)))
        
        # Create mapping from unitree joint names to mujoco joint indices
        src_joint_ids = []
        tgt_joint_ids = []
        
        for joint_name in shared_joint_names:
            mujoco_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            mujoco_joint_qpos_adr = self.model.jnt_qposadr[mujoco_joint_id]
            src_joint_ids.append(src_joint_names.index(joint_name))
            tgt_joint_ids.append(mujoco_joint_qpos_adr)
        self.src_joint_ids = np.array(src_joint_ids)
        self.tgt_joint_ids = np.array(tgt_joint_ids)
        
        # Get pelvis (floating base) joint address
        pelvis_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'floating_base_joint')
        assert pelvis_joint_id != -1, "floating_base_joint not found in model"
        self.pelvis_joint_qpos_adr = self.model.jnt_qposadr[pelvis_joint_id]

        # Initialize data storage
        self.joint_positions = np.zeros(len(src_joint_names))
        self.root_pose = np.zeros(7)  # [x, y, z, qw, qx, qy, qz]
        self.data_lock = threading.Lock()
        
        # Initialize recording variables
        self.qpos_records = []
        self.record_timestamps = []
        self.last_data_size_mb = 0
        self.record_counter = 0
        print(f"Recording qpos data with {self.model.nq} DOF")
        
        # Start ZMQ subscriber threads
        self.running = True
        self.joint_pos_thread = threading.Thread(target=self._joint_pos_subscriber_thread, daemon=True)
        self.joint_pos_thread.start()
        
        self.root_pose_thread = threading.Thread(target=self._root_pose_subscriber_thread, daemon=True)
        self.root_pose_thread.start()
        
        # Start MuJoCo update timer
        self.update_thread = threading.Thread(target=self._mujoco_update_loop, daemon=True)
        self.update_thread.start()
        
    def _joint_pos_subscriber_thread(self):
        """Thread function to continuously receive joint position data from remote machine"""
        while self.running:
            try:
                # Receive multipart message
                message = self.joint_pos_subscriber.recv_multipart(zmq.NOBLOCK)
                if len(message) == 2:
                    topic = message[0].decode('utf-8')
                    data_bytes = message[1]
                    
                    if topic == "joint_pos":
                        # Convert bytes back to numpy array
                        data = np.frombuffer(data_bytes, dtype=np.float64)
                        
                        with self.data_lock:
                            if len(data) <= len(self.joint_positions):
                                self.joint_positions[:] = data
                        
            except zmq.Again:
                # No message available, continue
                time.sleep(0.001)
            except Exception as e:
                print(f"Error in joint_pos subscriber thread: {str(e)}")
                time.sleep(0.01)
    
    def _root_pose_subscriber_thread(self):
        """Thread function to continuously receive root pose data from local machine"""
        while self.running:
            try:
                # Receive multipart message
                message = self.root_pose_subscriber.recv_multipart(zmq.NOBLOCK)
                if len(message) == 2:
                    received_obj_name = message[0].decode('utf-8')
                    pose_bytes = message[1]
                    
                    # Convert bytes back to numpy array with explicit dtype
                    pose_data = np.frombuffer(pose_bytes, dtype=np.float64)
                    
                    # Store in mocap_data with thread-safe access
                    with self.data_lock:
                        self.root_pose[:] = pose_data
                        
            except zmq.Again:
                # No message available, continue
                time.sleep(0.001)
            except Exception as e:
                print(f"Error in root_pose subscriber thread: {str(e)}")
                time.sleep(0.01)

    def _record_qpos(self):
        if not self.record:
            return
        
        """Record current qpos and timestamp"""
        current_time = time.time()
        qpos_copy = self.data.qpos.copy()
        
        self.qpos_records.append(qpos_copy)
        self.record_timestamps.append(current_time)
        self.record_counter += 1
        
        # Check data size every 100 records
        if self.record_counter % 100 == 0:
            self._check_data_size()

    def _check_data_size(self):
        """Check current data size and print reminder if it increased by 1MB"""
        if len(self.qpos_records) > 0:
            # Estimate size: each qpos has self.model.nq float64 values (8 bytes each)
            # Plus timestamps (8 bytes each)
            bytes_per_record = self.model.nq * 8 + 8  # qpos + timestamp
            total_bytes = len(self.qpos_records) * bytes_per_record
            current_size_mb = total_bytes / (1024 * 1024)
            
            if current_size_mb - self.last_data_size_mb >= 1.0:
                print(f"Recording data size: {current_size_mb:.2f} MB ({len(self.qpos_records)} records)")
                self.last_data_size_mb = int(current_size_mb)

    def _save_recorded_data(self):
        if not self.record:
            return
        
        """Save recorded qpos data to npz file"""
        if len(self.qpos_records) == 0:
            print("No data recorded to save.")
            return
            
        # Convert lists to numpy arrays
        qpos_array = np.array(self.qpos_records)
        timestamps_array = np.array(self.record_timestamps)
        
        # Generate filename with timestamp
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"recorded_qpos_{timestamp_str}.npz"
        
        # Save to npz file
        np.savez_compressed(
            filename,
            qpos=qpos_array,
            timestamps=timestamps_array,
            frequency=self.freq,
            nq=self.model.nq,
            joint_names=[self.model.joint(i).name for i in range(self.model.njnt)]
        )
        
        file_size_mb = os.path.getsize(filename) / (1024 * 1024)
        print(f"Saved {len(self.qpos_records)} qpos records to '{filename}' ({file_size_mb:.2f} MB)")

    def _mujoco_update_step_scheduled(self):
        """Scheduled MuJoCo update step"""
        if self.viewer is None:
            return
            
        loop_start = time.perf_counter()
        
        with self.data_lock:
            # Update joint positions
            self.data.qpos[self.tgt_joint_ids] = self.joint_positions[self.src_joint_ids]
            self.data.qpos[self.pelvis_joint_qpos_adr: self.pelvis_joint_qpos_adr + 7] = self.root_pose
        
        # Record qpos
        self._record_qpos()
        
        # Forward simulation and sync viewer
        mujoco.mj_forward(self.model, self.data)
        self.viewer.sync()
        
        elapsed = time.perf_counter() - loop_start
        if elapsed > self.update_dt:
            print(f"Update step took {elapsed:.6f} seconds, expected {self.update_dt:.6f}")

    def _mujoco_update_loop(self):
        """Update MuJoCo simulation at specified frequency using scheduler"""
        print(f"Starting MuJoCo update loop at {self.freq} Hz")
        
        # 使用scheduler进行精确时间控制
        scheduler = sched.scheduler(time.perf_counter, time.sleep)
        next_run_time = time.perf_counter()
        
        while self.running and self.viewer.is_running():
            if self.viewer is None:
                time.sleep(1.0)
                print("Waiting for MuJoCo model to be loaded...")
                continue
                
            # 调度下一次执行
            scheduler.enterabs(next_run_time, 1, self._mujoco_update_step_scheduled, ())
            scheduler.run()
            
            next_run_time += self.update_dt

    def main_loop(self):
        """Main loop - keep the program running"""
        try:
            print("MuJoCo Mocap Viewer running... Press Ctrl+C to stop")
            while self.running and self.viewer.is_running():
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down...")
            self.running = False
            
            # Save recorded data before closing
            print("Saving recorded data...")
            self._save_recorded_data()
            print("Recorded data saved.")
            
            if self.viewer:
                self.viewer.close()
            self.joint_pos_subscriber.close()
            self.root_pose_subscriber.close()
            self.zmq_context.term()
            print("Shutdown complete.")

def main():
    parser = argparse.ArgumentParser(description="MuJoCo Mocap Viewer with Dual ZMQ Subscribers")
    parser.add_argument("--model", type=str, default=scene, help="Path to MuJoCo model file")
    parser.add_argument("--joint_pos_ip", type=str, default="10.42.0.129", help="IP address for joint position publisher")
    parser.add_argument("--joint_pos_port", type=int, default=5555, help="Port for joint position publisher")
    parser.add_argument("--freq", type=int, default=50, help="Update frequency (Hz)")
    parser.add_argument("--record", action="store_true", help="Record qpos data")
    
    args = parser.parse_args()

    root_pose_ip = "localhost"
    from utils.common import PORTS
    root_pose_port = PORTS["pelvis"]
    
    viewer = MuJoCoMocapViewer(
        frequency=args.freq,
        mujoco_model_path=args.model,
        joint_pos_ip=args.joint_pos_ip,
        joint_pos_port=args.joint_pos_port,
        root_pose_ip=root_pose_ip,
        root_pose_port=root_pose_port,
        record=args.record,
    )
    viewer.main_loop()

if __name__ == "__main__":
    main()
