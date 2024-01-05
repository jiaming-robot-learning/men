
import numpy as np
import torch
from ..utils import depth as du
from ..utils import rotation as ru
from ..utils import pose as pu
from ..utils import mapping as mu
from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch.nn.functional as F
import json
import dataclasses
from matplotlib import pyplot as plt

@dataclass
class BaseMapArgs():
    """
    Dataclass for map arguments.
    """
    
    #--------------------
    # required args
    #--------------------
    min_depth:          float               # min depth in meters
    max_depth:          float               # max depth in meters
    normalized_depth:   bool                # whether the depth is normalized
    camera_height:      float               # camera height from ground in meters
    frame_height:       int                 # observation height
    frame_width:        int                 # observation width
    vision_range:       int                 # vision range in meters
    hfov:               int                 # horizontal field of view in degrees
    
    #--------------------
    # optional args
    #--------------------
    voxel_max_height:   int     = 300       # max height of the voxel grid
    voxel_min_height:   int     = -40       # min height of the voxel grid
    map_resolution:     int     = 5         # cm per pixel
    num_processes:      int     = 1
    device:             str     = 'auto'
    global_map_size_cm: int     = 4800      # global map size in cm
    global_downscaling: int     = 2         # local map downsample factor
    du_scale:           float   = 1         # depth downscaling factor
    require_grad:       bool    = False

    
class BaseMap(ABC):
    """
    Base class for all maps. The map class contains the following components:
    - map state, including global map, local map, and global pose. By default,
        the map is represented as a 2D grid map, but it can be extended to 3D by overridding the
        _init_map_state() and _get_total_map_channels() methods.
    - map parameters, including map resolution, map size, vision range, etc.
    - map update functions,
    """
    def __init__(self, **kwargs):

        if getattr(self, 'args', None) is None:
            self.args = BaseMapArgs(**kwargs)
            
        args = self.args
        # environment parameters
        self.num_env = args.num_processes
        self.screen_h = args.frame_height
        self.screen_w = args.frame_width
        self.require_grad = args.require_grad
        self.dtype = torch.float32
        if args.device == 'auto':
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(args.device)
        # map parameters
        self.xy_resolution = int(args.map_resolution)
        self.z_resolution = int(args.map_resolution)
        self.global_map_size_cm = int(args.global_map_size_cm)
        self.local_map_size_cm = int(args.global_map_size_cm // args.global_downscaling)
        self.global_map_size = int(args.global_map_size_cm // args.map_resolution)
        self.local_map_size = int(self.local_map_size_cm // args.map_resolution)
        self.global_downscaling = int(args.global_downscaling)
        self.vision_range = int(args.vision_range * 100 // args.map_resolution)
        self.du_scale = args.du_scale
        self.map_size_parameters = mu.MapSizeParameters(
            args.map_resolution, args.global_map_size_cm, self.global_downscaling
        )

        # agent and camera parameters
        self.fov = args.hfov
        self.normalized_depth = args.normalized_depth
        self.min_depth = args.min_depth * 100. # convert to cm
        self.max_depth = args.max_depth * 100. # convert to cm
        self.voxel_max_height = int(args.voxel_max_height / self.z_resolution)  
        self.voxel_min_height = int(args.voxel_min_height / self.z_resolution) # why -40? TODO: check 0
        self.agent_height = args.camera_height * 100. # this should be the height from the ground
        self.shift_loc = [self.vision_range * self.xy_resolution // 2, 0, np.pi / 2]
        self.camera_matrix = du.get_camera_matrix(
            self.screen_w, self.screen_h, self.fov)

        # map states
        self.total_map_channels = self._get_total_map_channels() 
        self._global_map = torch.zeros(
            self.num_env, self.total_map_channels,
            self.global_map_size,
            self.global_map_size,
            dtype=self.dtype,
            device=self.device,
        )
        
        self._local_map = torch.zeros(
            self.num_env, self.total_map_channels,
            self.local_map_size,
            self.local_map_size,
            dtype=self.dtype,
            device=self.device, 
        )
        self._map_channels = None
        
        # Global and local (x, y, o) sensor pose
        # This is in the world frame (x: forward, y: left, z: up)  unit: meter
        # Note 1: this is not the same as GPS:
        # global_pose = gps + global_map_size_cm/2/100
        # Note 2: x corresponds to the 2nd axis in the map frame,
        # and y corresponds to the 1st axis in the map frame
        self._global_pose = torch.zeros(
            self.num_env, 3, device=self.device, dtype=self.dtype)
          
        # always be (global_map_size_cm/200, global_map_size_cm/200),
        # since the local map is centered at the agent,
        # unless when the agent is at the edge of the map
        self._local_pose = torch.zeros(self.num_env, 3, device=self.device)

        # Origin of local map (3rd dimension stays 0)
        # This is in the world frame (x: forward, y: left, z: up)  unit: meter
        self._origins = torch.zeros(self.num_env, 3, device=self.device)

        # Local map boundaries
        # map frame: x: right, y: down
        self._lmb = torch.zeros(
            self.num_env, 4, dtype=torch.int32, device=self.device
        )  
        
        # Current gps and compass of the agent, in the episode frame,
        # which is defined as the frame when the agent is initialized:
        # x: forward, y: left, z: up 
        # self._cur_pose = torch.zeros(
        #     self.num_env, 3, device=self.device, dtype=self.dtype)

        # pre-compute variables
        x1 = self.local_map_size // 2 - self.vision_range // 2
        x2 = x1 + self.vision_range
        y1 = self.local_map_size // 2
        y2 = y1 + self.vision_range
        self.agent_view_boundaries = (x1, x2, y1, y2)

    @property
    def map_args(self):
        return self.args
    
    @property
    def global_pose(self) -> np.ndarray:
        """Get the global pose of the agent."""
        return np.copy(self._global_pose.cpu().numpy())
    
    @property
    def global_map(self) -> np.ndarray:
        """Get the global map."""
        return np.copy(self._global_map.cpu().numpy())
    
    @property
    def local_map(self) -> np.ndarray:
        """Get the local map."""
        return np.copy(self._local_map.cpu().numpy())
    
    
    @abstractmethod
    def _get_total_map_channels(self):
        """Get the total number of map channels."""
        pass
    
    def save_map_args(self, save_dir: str):
        """Save map arguments to a json file."""
        with open(save_dir, 'w') as f:
            json.dump(dataclasses.asdict(self.args), f)
    
    @staticmethod
    def load_map_args(load_dir: str):
        """Load map arguments from a json file."""
        with open(load_dir, 'r') as f:
            args = json.load(f)
        args = BaseMapArgs(**args)
        return args
        
    def reset_env(self, env_idx: int):
        """Reset the map for a specific environment.
        Always reset the map at the beginning of each episode, so that the agent
        is always at the center of the map.

        Args: env_idx: int, the index of the environment
        """
        mu.init_map_and_pose_for_env(
            env_idx,
            self._local_map,
            self._global_map,
            self._local_pose,
            self._global_pose,
            self._lmb,
            self._origins,
            self.map_size_parameters,
        )
    
        # self._cur_pose[env_idx] = 0.
        
    def reset(self):
        """Reset for all environments.
        """
        for env_idx in range(self.num_env):
            self.reset_env(env_idx)

    @abstractmethod
    def _process_obs(self,obs: np.ndarray, **kwargs):
        """Process the observation before feeding it to the map.
        Args: 
            obs: ndarray of size (B, C, H, W), 
                where B is the batch size, H is the height of the image, W is the width of the image
        """
        pass

    @abstractmethod
    def _get_ego_map(self,obs: np.ndarray, **kwargs):
        """Get the ego-centric map from the current observation.
        
        Args:
            obs: ndarray of size (B, H, W), 
                where B is the batch size, H is the height of the image, W is the width of the image
            camera_pose: (B, 4, 4) tensor, where B is the batch size,
        
        Returns:
            ego_map: (B, C, vr, vr) tensor, where B is the batch size,
                C is the number of channels, vr is the vision range
        
        """
        pass
            
    def update_global_map_pose(self,
                          ego_maps: torch.tensor, 
                          delta_pose: torch.tensor,
                          **kwargs):
        """Update the global map with ego-centric maps.
        
        Perform spatial transformation and update the global map.
        Here we use a local map to update the global map, which is more efficient.
        
        Args:
            ego_maps: (B, C, vr, vr) tensor, where B is the batch size,
                C is the number of channels, vr is the vision range
            delta_pose: (B, 3) tensor, where B is the batch size,
                3 is the delta pose (x, y, theta) in cm and rad
            
        """
        
        local_map = torch.zeros(
            self.num_env,
            self.total_map_channels,
            self.local_map_size,
            self.local_map_size,
            device=self.device,
            dtype=self.dtype,
        )

        x1, x2, y1, y2 = self.agent_view_boundaries
        
        local_map[:, :, y1:y2, x1: x2] = ego_maps

        current_local_pose = pu.get_new_pose_batch(self._local_pose.clone(), delta_pose)        
        st_pose = current_local_pose.clone().detach()

        st_pose[:, :2] = -(
            (
                st_pose[:, :2] * 100.0 / self.xy_resolution
                - self.local_map_size / 2
            )
            / (self.local_map_size / 2)
        )
        st_pose[:, 2] = 90.0 - (st_pose[:, 2])

        rot_mat, trans_mat = ru.get_grid(st_pose, local_map.size(), self.dtype)
        rotated = F.grid_sample(local_map, rot_mat, align_corners=True)
        translated = F.grid_sample(rotated, trans_mat, align_corners=True)
        
        maps = torch.cat((self._local_map.unsqueeze(1), translated.unsqueeze(1)), 1)
        current_map, _ = torch.max(maps, 1)
        current_map = torch.clamp(current_map, min=0.0, max=1.0)
        
        # update global map and poses
        # self._cur_pose = current_pose.clone().detach()
        for env_idx in range(self.num_env):
            
            x1, x2, y1, y2 = self._lmb[env_idx]
            self._global_map[env_idx, :, x1:x2, y1:y2] = current_map[env_idx]
            self._global_pose[env_idx] = current_local_pose[env_idx] + self._origins[env_idx] 
            mu.recenter_local_map_and_pose_for_env(
                env_idx,
                self._local_map,
                self._global_map,
                self._local_pose,
                self._global_pose,
                self._lmb,
                self._origins,
                self.map_size_parameters,
            )

    def update(self,obs, 
               delta_pose: np.ndarray = None, 
               cur_pose: np.ndarray = None, 
               **kwargs):
        """Update the map with the current observation.
        
        Either delta_pose or cur_pose should be provided.
        
        Args:
            obs: ndarray of size (B, H, W), 
                where B is the batch size, H is the height of the image, W is the width of the image
            delta_pose: (B, 3) tensor, where B is the batch size,
                3 is the delta pose (x, y, theta) in cm and rad. 
                Note dx is in the forward direction, dy is in the left direction, and dtheta counter-clockwise
        Returns:
        
        """
        if delta_pose is not None:
            assert cur_pose is None
            delta_pose = torch.from_numpy(delta_pose).float().to(self.device)
        else:
            assert cur_pose is not None
            
            delta_pose = pu.get_rel_pose_change(cur_pose, self._cur_pose)
            delta_pose = torch.from_numpy(delta_pose).float().to(self.device)
            
        with torch.set_grad_enabled(self.require_grad):
            obs = self._process_obs(obs, **kwargs)
            local_map = self._get_ego_map(obs, **kwargs)
            self.update_global_map_pose(local_map, delta_pose, **kwargs)
            