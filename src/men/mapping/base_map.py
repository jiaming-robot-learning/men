
import numpy as np
import torch
from men.utils import depth as du
from men.utils import rotation as ru
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass(kw_only=True)
class MapArgs():
    """
    Dataclass for map arguments.
    """
    
    #--------------------
    # required args
    #--------------------
    min_depth:          float
    max_depth:          float
    camera_height:      float
    frame_height:       int                 # observation height
    frame_width:        int                 # observation width
    map_resolution:     int                 # cm per pixel
    vision_range:       int                 # cm
    hfov:               int                 # horizontal field of view in degrees
    
    #--------------------
    # optional args
    #--------------------
    num_processes:      int     = 1
    device:             str     = 'auto'
    global_map_size:    int     = 4800      # global map size in cm
    local_map_size:     int     = 2400      # optional for planning
    ds:                 float   = 1         # downscaling factor
    du_scale:           float   = 1         # depth resolution
    map_pred_threshold: float   = 1.0
    exp_pred_threshold: float   = 1.0
    require_grad:       bool    = False

    num_obj_categories: int     = 0        # for semantic mapping 
    exp_map:            bool    = True     # explored map
    traj_map:           bool    = False    # trajectory map

    
class BaseMap(ABC):
    def __init__(self, **kwargs):

        self.args = args = MapArgs(kwargs)
        self.num_processes = args.num_processes
        self.screen_h = args.frame_height
        self.screen_w = args.frame_width
        self.map_resolution = args.map_resolution
        self.z_resolution = args.map_resolution
        self.vision_range = args.vision_range
        self.fov = args.hfov
        self.du_scale = args.du_scale
        self.require_grad = args.require_grad
        
        if args.device == 'auto':
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")

        self.map_pred_threshold = args.map_pred_threshold
        self.exp_pred_threshold = args.exp_pred_threshold
        self.exp_channel = args.exp_map
        self.traj_channel = args.traj_map

        # TODO how to init agent height and camera matrix
        self.max_height = int(360 / self.z_resolution)
        self.min_height = int(-40 / self.z_resolution)

        self.agent_height = args.camera_height * 100.
        self.shift_loc = [self.vision_range *
                          self.map_resolution // 2, 0, np.pi / 2]
        self.camera_matrix = du.get_camera_matrix(
            self.screen_w, self.screen_h, self.fov)

        self._global_map = None
        self._map_channels = None
        self._init_map_state()

    
    def _init_map_state(self):
        self._map_channels = {
            'occupancy': 0
        }
        if self.exp_channel:
            self._map_channels['explored'] = 1
        if self.traj_channel:
            self._map_channels['trajectory'] = 2
            
        self._global_map = torch.zeros(
            self.num_processes, len(self._map_channels),
            self.args.global_map_size // self.map_resolution,
            self.args.global_map_size // self.map_resolution,
        ).float().to(self.device)
        
    def _process_obs(self,obs):
        return obs

    @abstractmethod
    def _get_ego_map(self,obs):
        """
        Get the ego-centric map from the current observation.
        """
        pass
            
    @abstractmethod
    def _update_map(self,local_map, delta_pose):
        pass
    
    def update(self,obs, delta_pose, camera_pose=None):
        """
        Update the map with the current observation.
        """
        with torch.set_grad_enabled(self.require_grad):
            obs = self._process_obs(obs)
            local_map = self._get_ego_map(obs, camera_pose)
            self._update_map(local_map, delta_pose)
            