

from men.mapping.occupancy_map import OccupancyMap2D, OccupancyMapArgs

def occupancy_map_from_config(config):
    
    agent_config = config.habitat.simulator.agents.main_agent
    device_id = config.habitat.simulator.habitat_sim_v0.gpu_device_id
    occ_map = OccupancyMap2D(
        min_depth = agent_config.sim_sensors.depth_sensor.min_depth,
        max_depth = agent_config.sim_sensors.depth_sensor.max_depth,
        camera_height = agent_config.sim_sensors.depth_sensor.position[1],
        frame_height = agent_config.sim_sensors.depth_sensor.height,
        frame_width = agent_config.sim_sensors.depth_sensor.width,
        vision_range = agent_config.sim_sensors.depth_sensor.max_depth,
        hfov = agent_config.sim_sensors.depth_sensor.hfov,
        normalized_depth = agent_config.sim_sensors.depth_sensor.normalize_depth,
        device = f'cuda:{device_id}' if device_id >= 0 else 'cpu'
    )

    return occ_map

    
def unnormalize_depth(depth, config):
    """Unnormalize depth from [0, 1] to [min_depth, max_depth].
    Args:
        depth: (h, w) depth map
        config: habitat config
    Returns:
        un_depth: (h, w) unnormalized depth map
    """
    if not config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.normalize_depth:
        un_depth = depth 
        
    else:
        min_depth = config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.min_depth
        max_depth = config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.max_depth
        un_depth = depth * (max_depth - min_depth) + min_depth

    if un_depth.ndim == 3:
        un_depth = un_depth.squeeze(-1)
    return un_depth