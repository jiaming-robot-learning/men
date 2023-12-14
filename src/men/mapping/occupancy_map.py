
# import math
# import os
# import cv2
# import numpy as np
# from PIL import Image
# from torchvision import transforms
# import torch.nn as nn
# import torch.nn.functional as F
# from task.util.semantic_annotation import beacon_class_list
# from utils.vis import save_image, plot_image
# import envs.utils.pose as pu
# import agents.utils.visualization as vu

import numpy as np
import torch
from men.utils import depth as du
from men.utils import rotation as ru
from .base_map import BaseMap
    
class OcupancyMap2D(BaseMap):
    """
    Grid-based 2D occupancy map.
    
    """
    def __init__(self,**kwargs):
        super().__init__(kwargs)
        
        vr = self.vision_range
        self.n_channels = 1 # depth only

        self.init_grid = torch.zeros(
            self.num_processes, self.n_channels, vr, vr,
            self.max_height - self.min_height
        ).float().to(self.device)

        self.feat = torch.ones(
            self.num_processes, self.n_channels,
            self.screen_h // self.du_scale * self.screen_w // self.du_scale
        ).float().to(self.device)

    
    def _get_ego_map(self, 
                     obs: np.ndarray,
                     camera_pose: np.ndarray = None
                     ) -> torch.Tensor:
        """
        Get the ego-centric map from the current observation.
        Args:
            obs: ndarray of size (B, H, W), 
            where B is the batch size, H is the height of the image, W is the width of the image
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

        point_cloud_t = du.get_point_cloud_from_z_t(
            depth, self.camera_matrix, self.device, scale=self.du_scale)

        agent_view_t = du.transform_camera_view_t(
            point_cloud_t, agent_height, tilt, self.device)

        agent_view_centered_t = du.transform_pose_t(
            agent_view_t, self.shift_loc, self.device)

        max_h = self.max_height
        min_h = self.min_height
        z_resolution = self.z_resolution
        vision_range = self.vision_range
        
        XYZ_cm_std = agent_view_centered_t.float()
        XYZ_cm_std[..., :2] = (XYZ_cm_std[..., :2] / self.map_resolution)
        XYZ_cm_std[..., :2] = (XYZ_cm_std[..., :2] -
                               vision_range // 2.) / vision_range * 2.
        XYZ_cm_std[..., 2] = XYZ_cm_std[..., 2] / z_resolution
        XYZ_cm_std[..., 2] = (XYZ_cm_std[..., 2] -
                              (max_h + min_h) // 2.) / (max_h - min_h) * 2.

        XYZ_cm_std = XYZ_cm_std.permute(0, 3, 1, 2)
        XYZ_cm_std = XYZ_cm_std.view(XYZ_cm_std.shape[0],
                                     XYZ_cm_std.shape[1],
                                     XYZ_cm_std.shape[2] * XYZ_cm_std.shape[3])

        voxels = du.splat_feat_nd(
            self.init_grid * 0., self.feat, XYZ_cm_std).transpose(2, 3)

        # TODO: what is this??
        min_z = int(25 / z_resolution - min_h)
        max_z = int((self.agent_height + 1) / z_resolution - min_h)

        agent_height_proj = voxels[..., min_z:max_z].sum(4)
        occ_map_pred = agent_height_proj[:, 0:1, :, :]
        occ_map_pred = occ_map_pred / self.map_pred_threshold
        occ_map_pred = torch.clamp(occ_map_pred, min=0.0, max=1.0)

        if self.exp_channel:
            all_height_proj = voxels.sum(4)
            exp_map_pred = all_height_proj[:, 0:1, :, :]
            exp_map_pred = exp_map_pred / self.exp_pred_threshold
            exp_map_pred = torch.clamp(exp_map_pred, min=0.0, max=1.0)

            result = torch.cat((occ_map_pred, exp_map_pred), dim=1)
        else:
            result = occ_map_pred
            
        return result