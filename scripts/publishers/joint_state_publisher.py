"""This script listens to the low state of the robot and publishes the joint positions via ZMQ"""

import numpy as np
import yaml
import zmq
import time
import threading
import argparse
import sched

from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowState_go
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowState_hg
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize

import sys
sys.path.append(".")

from utils.strings import unitree_joint_names

class JointStatePublisher:
    """
    Receives joint state from Unitree SDK and publishes via ZMQ as numpy array
    """
    def __init__(self, robot_config, dest_joint_names, zmq_port=5555, publish_freq=50):
        # initialize robot related processes
        if robot_config.get("INTERFACE", None):
            ChannelFactoryInitialize(robot_config["DOMAIN_ID"], robot_config["INTERFACE"])
        else:
            ChannelFactoryInitialize(robot_config["DOMAIN_ID"])

        # Initialize channel subscriber
        robot_type = robot_config["ROBOT_TYPE"]
        if robot_type == "h1" or robot_type == "go2":
            self.robot_lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_go)
            self.robot_lowstate_subscriber.Init(self.LowStateHandler_go, 1)
        elif robot_type == "g1_29dof" or robot_type == "h1-2_27dof" or robot_type == "h1-2_21dof":
            self.robot_lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_hg)
            self.robot_lowstate_subscriber.Init(self.LowStateHandler_hg, 1)
        else:
            raise NotImplementedError(f"Robot type {robot_type} is not supported")

        # Initialize joint mapping
        self.num_dof = len(dest_joint_names)
        self.joint_indices_in_source = [unitree_joint_names.index(name) for name in dest_joint_names]
        
        # Initialize joint state arrays
        self.joint_pos = np.zeros(self.num_dof)
        self.joint_vel = np.zeros(self.num_dof)
        
        # Initialize robot state
        self.robot_low_state = None
        
        # Initialize ZMQ publisher
        self.zmq_context = zmq.Context()
        self.publisher = self.zmq_context.socket(zmq.PUB)
        self.publisher.bind(f"tcp://*:{zmq_port}")
        print(f"ZMQ publisher bound to port {zmq_port}")
        
        # Publishing frequency
        self.publish_freq = publish_freq
        self.publish_interval = 1.0 / publish_freq
        
        # Start publishing thread
        self.publish_thread = threading.Thread(target=self._publish_loop, daemon=True)
        self.publish_thread.start()
        
    def _publish_loop(self):
        """Publishing loop that runs in a separate thread with precise timing"""
        publish_cnt = 0
        start_time = time.time()
        
        # 使用scheduler进行精确时间控制
        scheduler = sched.scheduler(time.perf_counter, time.sleep)
        next_run_time = time.perf_counter()
        
        while True:
            try:
                # 调度下一次执行
                scheduler.enterabs(next_run_time, 1, self._publish_step_scheduled, ())
                scheduler.run()
                
                next_run_time += self.publish_interval
                publish_cnt += 1
                
                # Print FPS every 100 iterations
                if publish_cnt % 100 == 0:
                    current_time = time.time()
                    actual_freq = 100 / (current_time - start_time)
                    print(f"Publishing frequency: {actual_freq:.1f} Hz (target: {self.publish_freq} Hz)")
                    start_time = current_time
                    
            except KeyboardInterrupt:
                print("Publishing loop interrupted")
                break
            except Exception as e:
                print(f"Error in publishing loop: {e}")
                time.sleep(0.01)
    
    def _publish_step_scheduled(self):
        """Execute one publishing step with timing measurement"""
        if not self.robot_low_state:
            return
            
        loop_start = time.perf_counter()
        
        # Extract joint data from robot state
        source_joint_state = self.robot_low_state.motor_state
        for dst_idx, src_idx in enumerate(self.joint_indices_in_source):
            self.joint_pos[dst_idx] = source_joint_state[src_idx].q
            self.joint_vel[dst_idx] = source_joint_state[src_idx].dq
        
        # Publish joint positions
        self.publisher.send_multipart([
            b"joint_pos",
            self.joint_pos.astype(np.float64).tobytes()
        ])
        # print(self.joint_pos)
        
        # Measure execution time
        elapsed = time.perf_counter() - loop_start
        if elapsed > self.publish_interval:
            print(f"Publish step took {elapsed:.6f} seconds, expected {self.publish_interval:.6f}")

    def LowStateHandler_go(self, msg: LowState_go):
        print("received low state")
        self.robot_low_state = msg
    
    def LowStateHandler_hg(self, msg: LowState_hg):
        print("received low state")
        self.robot_low_state = msg

def main():
    parser = argparse.ArgumentParser(description="Joint State ZMQ Publisher")
    parser.add_argument("--robot_config", type=str, default="config/robot/g1.yaml", help="Robot config file")
    parser.add_argument("--port", type=int, default=5555, help="ZMQ port")
    parser.add_argument("--freq", type=int, default=50, help="Publishing frequency")
    
    args = parser.parse_args()
    with open(args.robot_config) as file:
        robot_config = yaml.load(file, Loader=yaml.FullLoader)
    dest_joint_names = unitree_joint_names
    
    publisher = JointStatePublisher(
        robot_config=robot_config,
        dest_joint_names=dest_joint_names,
        zmq_port=args.port,
        publish_freq=args.freq
    )
    
    print(f"Publishing joint state for {robot_config['ROBOT_TYPE']} on port {args.port} at {args.freq} Hz")
    print("Press Ctrl+C to stop...")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down...")
        publisher.zmq_context.term()

if __name__ == "__main__":
    main() 