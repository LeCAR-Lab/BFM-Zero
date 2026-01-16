import mujoco
import mujoco.viewer
import time
from threading import Thread
import sched
import os

import sys
sys.path.append(".")
from sim_env.utils.simulation_bridge import SimulationBridge
from sim_env.utils.elastic_band import ElasticBand

class BaseSimulator:
    def __init__(self, robot_config, scene_config):
        self.robot_config = robot_config
        self.scene_config = scene_config
        self.sim_dt = self.scene_config["SIMULATE_DT"]
        self.viewer_dt = self.scene_config["VIEWER_DT"]

        self.init_scene()
        self.sim_bridge = SimulationBridge(
            self.mj_model, self.mj_data, self.robot_config, self.scene_config
        )

        # for more scenes
        self.init_subscriber()
        self.init_publisher()

        self.sim_thread = Thread(target=self.SimulationThread)

        try:
            if os.name == 'posix':
                import ctypes
                libc = ctypes.CDLL("libc.so.6")
                # set real-time scheduling policy
                SCHED_FIFO = 1
                class sched_param(ctypes.Structure):
                    _fields_ = [("sched_priority", ctypes.c_int)]
                
                param = sched_param()
                param.sched_priority = 50
                try:
                    libc.sched_setscheduler(0, SCHED_FIFO, ctypes.byref(param))
                    print("Set real-time scheduling priority")
                except:
                    print("Could not set real-time priority (try running with sudo)")
        except:
            pass

    def init_subscriber(self):
        pass

    def init_publisher(self):
        pass
    
    def init_scene(self):
        robot_scene = self.scene_config["ROBOT_SCENE"]
        self.mj_model = mujoco.MjModel.from_xml_path(robot_scene)
        self.mj_data = mujoco.MjData(self.mj_model)
        self.mj_model.opt.timestep = self.sim_dt
        # Enable the elastic band
        if self.scene_config["ENABLE_ELASTIC_BAND"]:
            self.elastic_band = ElasticBand()
            if "h1" in self.robot_config["ROBOT_TYPE"] or "g1" in self.robot_config["ROBOT_TYPE"]:
                self.band_attached_link = self.mj_model.body("torso_link").id
            else:
                self.band_attached_link = self.mj_model.body("base_link").id
            key_callback = self.elastic_band.MujocoKeyCallback
        else:
            key_callback = None

        self.viewer = mujoco.viewer.launch_passive(
            self.mj_model,
            self.mj_data,
            key_callback=key_callback,
            show_left_ui=False,
            show_right_ui=False,
        )
        # get pelvis body id
        self.pelvis_body_id = self.mj_model.body("pelvis").id
        self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        self.viewer.cam.trackbodyid = self.pelvis_body_id

    def sim_step(self):
        self.sim_bridge.publish_low_state()
        if self.scene_config["ENABLE_ELASTIC_BAND"]:
            if self.elastic_band.enable:
                pos = self.mj_data.xpos[self.band_attached_link]
                lin_vel = self.mj_data.cvel[self.band_attached_link, 3:6]
                self.mj_data.xfrc_applied[self.band_attached_link, :3] = (
                    self.elastic_band.Advance(pos, lin_vel)
                )
        self.sim_bridge.compute_torques()
        self.mj_data.ctrl[:] = self.sim_bridge.torques
        mujoco.mj_step(self.mj_model, self.mj_data)

    def SimulationThread(self):
        sim_cnt = 0
        start_time = time.time()
        
        scheduler = sched.scheduler(time.perf_counter, time.sleep)
        next_run_time = time.perf_counter()
        
        while self.viewer.is_running():
            scheduler.enterabs(next_run_time, 1, self._sim_step_scheduled, ())
            scheduler.run()
            
            next_run_time += self.sim_dt
            sim_cnt += 1
            
            if sim_cnt % (self.viewer_dt / self.sim_dt) == 0:
                self.viewer.sync()
        
            # Get FPS
            if sim_cnt % 100 == 0:
                current_time = time.time()
                print(f"FPS: {100 / (current_time - start_time)}")
                start_time = current_time

    def _sim_step_scheduled(self):
        loop_start = time.perf_counter()
        self.sim_step()
        elapsed = time.perf_counter() - loop_start
        if elapsed > self.sim_dt:
            print(f"Sim step took {elapsed:.6f} seconds, expected {self.sim_dt}")


if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description="Robot")
    parser.add_argument(
        "--robot_config", type=str, default="config/robot/g1.yaml", help="robot config file"
    )
    parser.add_argument(
        "--scene_config", type=str, default="config/scene/g1_29dof_nohand.yaml", help="scene config file"
    )
    args = parser.parse_args()

    with open(args.robot_config) as file:
        robot_config = yaml.load(file, Loader=yaml.FullLoader)
    with open(args.scene_config) as file:
        scene_config = yaml.load(file, Loader=yaml.FullLoader)

    simulation = BaseSimulator(robot_config, scene_config)
    simulation.sim_thread.start()