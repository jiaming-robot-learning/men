
import habitat
import numpy as np
from men.mapping.occupancy_map import OccupancyMap2D, OccupancyMapArgs
from men.visualization.occupancy_map_vis import OccupancyMapVis
from matplotlib import pyplot as plt
from men.visualization.viewer import OpenCVViewer
from habitat.config.default_structured_configs import (
    PointGoalWithGPSCompassSensorConfig,
    GPSSensorConfig,
    CompassSensorConfig,
)
from men.utils import pose as pu


ACTION_MAP = {
    'a': {'action':'turn_left', 'action_args': None},
    'd': {'action':'turn_right', 'action_args': None},
    's': {'action':'stop', 'action_args': None},
    'w': {'action':'move_forward', 'action_args': None},
}


config = habitat.get_config("benchmark/nav/pointnav/pointnav_hm3d.yaml")
# add gps and compass sensors
with habitat.config.read_write(config):
    config.habitat.task.lab_sensors = {
        "pointgoal_with_gps_compass": PointGoalWithGPSCompassSensorConfig(),
        "compass_sensor": CompassSensorConfig(),
        "gps_sensor": GPSSensorConfig(),
    }
    config.habitat.dataset.split = 'val'
agent_config = config.habitat.simulator.agents.main_agent


occ_map = OccupancyMap2D(
    min_depth = agent_config.sim_sensors.depth_sensor.min_depth,
    max_depth = agent_config.sim_sensors.depth_sensor.max_depth,
    camera_height = agent_config.sim_sensors.depth_sensor.position[1],
    frame_height = agent_config.sim_sensors.depth_sensor.height,
    frame_width = agent_config.sim_sensors.depth_sensor.width,
    vision_range = agent_config.sim_sensors.depth_sensor.max_depth,
    hfov = agent_config.sim_sensors.depth_sensor.hfov,
    normalized_depth = agent_config.sim_sensors.depth_sensor.normalize_depth,
)

viewer = OpenCVViewer(action_map=ACTION_MAP)
visualizer = OccupancyMapVis()

with habitat.Env(config) as env:
    obs = env.reset()  
    occ_map.reset()
    last_pose = np.zeros(3)

    print("Agent acting inside environment.")
    count_steps = 0

    while not env.episode_over:
        
        img = visualizer.visualize(rgb=obs['rgb'], depth=obs['depth'], map=occ_map.local_map[0])
        res = viewer.imshow(img)
        action = res['action']
        obs = env.step(action)  
        
        curr_pose = np.array([obs['gps'][0], -obs['gps'][1], obs['compass'][0]]) # y is flipped
        pose_delta = np.array(pu.get_rel_pose_change(curr_pose, last_pose))[None, ...]
        last_pose = curr_pose

        depth = obs['depth'].squeeze()[None, ...]
        
        occ_map.update(depth, delta_pose=pose_delta)
        info = env.get_metrics()

        count_steps += 1

    print("Episode finished after {} steps.".format(count_steps))


