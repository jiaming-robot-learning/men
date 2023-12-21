

from typing_extensions import override
import numpy as np
import torch
from ..utils import depth as du
from ..utils import rotation as ru
from .base_map import BaseMap, BaseMapArgs
    
from matplotlib import pyplot as plt


class OccupancyMapArgs(BaseMapArgs):
    
    map_pred_threshold: float   = 1.0
    exp_pred_threshold: float   = 1.0
    

class OccupancyMap2D(BaseMap):
    """Grid-based 2D occupancy map.
    
    """
    def __init__(self,**kwargs):
        
        self.args = OccupancyMapArgs(**kwargs)
        super().__init__()
        args = self.args

        self.map_pred_threshold = args.map_pred_threshold
        self.exp_pred_threshold = args.exp_pred_threshold
        self._map_channels = {
            'occupied': 0,
            'explored': 1,
        }
        
        # map min height (20cm), ignore the ground
        self.filtered_min_height = int(
            20 / self.z_resolution - self.voxel_min_height
        ) 
        self.max_mapped_height = int(
            (self.agent_height + 1) / self.z_resolution - self.voxel_min_height
        )
        

    def _get_total_map_channels(self):
        return 2
    
    @override
    def _process_obs(self, obs: np.ndarray, **kwargs):
        """
        Unnormalize the depth image.

        Args:
            obs: (bs, h, w) depth image
        
        
        """
        # obs *= 100 # convert to cm
        if self.normalized_depth:
            obs = obs * ( self.max_depth - self.min_depth )
            obs = obs + self.min_depth
            
        return obs
    
    @override
    def _get_ego_map(self, 
                    obs: np.ndarray,
                    camera_pose: np.ndarray = None,
                    **kwargs
                    ) -> torch.Tensor:
        """
        Args:
            camera_pose: (bs, 4, 4) camera pose in world frame
        """
        bs, h, w = obs.shape
        
        if camera_pose is not None:
            # TODO: this should be placed somewhere else
            # TODO: make consistent between sim and real
            # hab_angles = pt.matrix_to_euler_angles(camera_pose[:, :3, :3], convention="YZX")
            # angles = pt.matrix_to_euler_angles(camera_pose[:, :3, :3], convention="ZYX")
            angles = np.array(
                [ru.euler_from_matrix(p[:3, :3], "rzyx") for p in camera_pose]
            )
            # For habitat - pull x angle
            # tilt = angles[:, -1]
            # For real robot
            tilt = np.rad2deg(angles[:, 1])

            # Get the agent pose
            # hab_agent_height = camera_pose[:, 1, 3] * 100
            agent_pos = camera_pose[:, :3, 3] * 100
            agent_height = agent_pos[:, 2]
        else:
            tilt = np.zeros(bs)
            agent_height = self.agent_height
        
        depth = torch.from_numpy(obs).float().to(self.device)

        point_cloud = du.get_point_cloud_from_z_t(
            depth, self.camera_matrix, self.device, scale=self.du_scale)

        point_cloud_base = du.transform_camera_view_t(
            point_cloud, agent_height, tilt, self.device)

        point_cloud_map_coords = du.transform_pose_t(
            point_cloud_base, self.shift_loc, self.device)

        max_h = self.voxel_max_height
        min_h = self.voxel_min_height
        vr = self.vision_range
        feat_channels = 1
        
        self.init_grid = torch.zeros(
            self.num_env, feat_channels, vr, vr,
            self.voxel_max_height - self.voxel_min_height
        ).float().to(self.device)

        self.feat = torch.ones(
            self.num_env, feat_channels,
            self.screen_h // self.du_scale * self.screen_w // self.du_scale
        ).float().to(self.device)

        XYZ_cm_std = point_cloud_map_coords.float()
        XYZ_cm_std[..., :2] = (XYZ_cm_std[..., :2] / self.xy_resolution)
        XYZ_cm_std[..., :2] = (XYZ_cm_std[..., :2] -
                               vr // 2.) / vr * 2.
        XYZ_cm_std[..., 2] = XYZ_cm_std[..., 2] / self.z_resolution
        XYZ_cm_std[..., 2] = (XYZ_cm_std[..., 2] -
                              (max_h + min_h) // 2.) / (max_h - min_h) * 2.

        XYZ_cm_std = XYZ_cm_std.permute(0, 3, 1, 2)
        XYZ_cm_std = XYZ_cm_std.view(XYZ_cm_std.shape[0],
                                     XYZ_cm_std.shape[1],
                                     XYZ_cm_std.shape[2] * XYZ_cm_std.shape[3])

        voxels = du.splat_feat_nd(
            self.init_grid * 0., self.feat, XYZ_cm_std).transpose(2, 3)

        min_z = self.filtered_min_height
        max_z = self.max_mapped_height

        # TODO: dialate?
        # if self.dilate_obstacles:
            
        #     fp_map_pred =  torch.nn.functional.conv2d(
        #         fp_map_pred, self.dialate_kernel.to(device), padding=self.dilate_size // 2
        #     ).clamp(0, 1)

        agent_height_proj = voxels[..., min_z:max_z].sum(4)
        occ_map_pred = agent_height_proj[:, 0:1, :, :]
        occ_map_pred = occ_map_pred / self.map_pred_threshold
        # occ_map_pred = torch.clamp(occ_map_pred, min=0.0, max=1.0)

        all_height_proj = voxels.sum(4)
        exp_map_pred = all_height_proj[:, 0:1, :, :]
        exp_map_pred = exp_map_pred / self.exp_pred_threshold
        # exp_map_pred = torch.clamp(exp_map_pred, min=0.0, max=1.0)

        ego_maps = torch.cat([occ_map_pred, exp_map_pred], dim=1)
            
        return ego_maps 