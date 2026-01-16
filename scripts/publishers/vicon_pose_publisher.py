"""This script listens to the Vicon data and publishes the object poses via ZMQ"""

import time
import numpy as np
from scipy.spatial.transform import Rotation as R

from typing import List

# Import Vicon DataStream SDK
from pyvicon_datastream import tools

import zmq
from threading import Thread

class Vicon:
    def __init__(
        self,
        vicon_object_names: List[str],
        publish_names: List[str],
        frequency: int = 200,
        vicon_tracker_ip: str = "128.2.184.3",
    ):
        
        # Vicon DataStream IP and object name
        self.vicon_tracker_ip = vicon_tracker_ip

        self.freq = frequency
        self.vicon_object_names = vicon_object_names
        self.publish_names = publish_names
        
        # Connect to Vicon DataStream
        self.tracker = tools.ObjectTracker(self.vicon_tracker_ip)
        if self.tracker.is_connected:
            print(f"Connected to Vicon DataStream at {self.vicon_tracker_ip}")
        else:
            print(f"Failed to connect to Vicon DataStream at {self.vicon_tracker_ip}")
            raise Exception(f"Connection to {self.vicon_tracker_ip} failed")

        # Initialize ZMQ publishers
        self.init_publisher()

        # Frequency counter
        self.freq_counter = 0
        
    def init_publisher(self):
        # Initialize ZMQ context and publishers with fixed ports
        self.zmq_context = zmq.Context()
        self.pose_publishers = {}
        
        # Use fixed ports for each object
        import sys
        sys.path.append(".")
        from utils.common import PORTS
        
        # Define port mapping for vicon objects
        vicon_ports = {name: PORTS[name] for name in self.publish_names}
        
        for vicon_obj, publish_name in zip(self.vicon_object_names, self.publish_names):
            port = vicon_ports[publish_name]  # Default port if not found
            socket = self.zmq_context.socket(zmq.PUB)
            socket.bind(f"tcp://*:{port}")
            self.pose_publishers[publish_name] = socket
            print(f"Publishing {publish_name} (vicon: {vicon_obj}) poses on port {port}")

        # Give time for sockets to bind
        time.sleep(1)

        # Start state publishing thread
        self.publish_rate = self.freq
        self.state_thread = Thread(target=self.state_publisher_thread, daemon=True)
        self.state_thread.start()
        
    def get_vicon_data(self, vicon_object_name):
        position = self.tracker.get_position(vicon_object_name)
        
        if not position:
            print(f"Cannot get the pose of `{vicon_object_name}`.")
            return None, None, None

        try:
            obj = position[2][0]
            _, _, x, y, z, roll, pitch, yaw = obj
            current_time = time.time()

            # Position and orientation
            position = np.array([x, y, z]) / 1000. # Convert to meters
            rotation = R.from_euler('XYZ', [roll, pitch, yaw], degrees=False)
            quaternion = rotation.as_quat()  # [x, y, z, w]

            return current_time, position, quaternion
        except Exception as e:
            print(f"Error retrieving Vicon data: {e}")
            return None, None, None

    def log_frequency(self):
        # Log the frequency information
        print(f"Vicon data acquisition frequency: {self.freq_counter} Hz")
        self.freq_counter = 0  # Reset the counter for the next second

    def state_publisher_thread(self):
        print("Starting Vicon state publisher thread")
        
        # Timer for frequency logging
        last_log_time = time.time()
        
        while True:
            try:
                for vicon_object_name, publish_name in zip(self.vicon_object_names, self.publish_names):
                    current_time, position, quaternion = self.get_vicon_data(vicon_object_name)
                    if position is None:
                        print(f"Failed to get Vicon data for {vicon_object_name}.")
                        continue

                    # Combine position and quaternion into a single numpy array
                    # Format: [x, y, z, qx, qy, qz, qw]
                    quaternion = np.roll(quaternion, 1)
                    pose_data = np.concatenate([position, quaternion]).astype(np.float64)
                    
                    # Send via ZMQ (same format as push_door.py)
                    self.pose_publishers[publish_name].send_multipart([
                        publish_name.encode('utf-8'),
                        pose_data.tobytes()
                    ])
                    
                # Increment frequency counter
                self.freq_counter += 1
                
                # Log frequency every second
                current_time = time.time()
                if current_time - last_log_time >= 1.0:
                    self.log_frequency()
                    last_log_time = current_time
                
                time.sleep(1.0 / self.publish_rate)
                
            except Exception as e:
                print(f"Error in Vicon state publisher thread: {str(e)}")
                time.sleep(0.1)

    def main_loop(self):
        print("Starting Vicon data acquisition...")
        try:
            while True:
                time.sleep(1)  # Main thread just waits, publishing happens in background thread
        except KeyboardInterrupt:
            print("Exiting Vicon data acquisition.")

if __name__ == "__main__":
    publish_names = ["Wall", "Door", "pelvis"]
    object_names = [f"haoyang_{name}" for name in publish_names]
    vicon = Vicon(vicon_object_names=object_names, publish_names=publish_names)

    try:
        vicon.main_loop()
    except KeyboardInterrupt:
        pass

