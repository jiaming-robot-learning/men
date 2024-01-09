
import numpy as np
import torch
import cv2
from matplotlib import pyplot as plt
from habitat.utils.visualizations import maps
from skimage.draw import line_aa

# color palate for mapping
MAP_INVALID_POINT = 0
MAP_VALID_POINT = 1
MAP_BORDER_INDICATOR = 2
MAP_EXPLORED_AREA = 4
MAP_TARGET_POINT_INDICATOR = 6
MAP_SHORTEST_PATH_COLOR = 7
MAP_VIEW_POINT_INDICATOR = 8
MAP_TARGET_BOUNDING_BOX = 9
TOP_DOWN_MAP_COLORS = np.full((256, 3), 150, dtype=np.uint8)
TOP_DOWN_MAP_COLORS[10:] = cv2.applyColorMap(
    np.arange(246, dtype=np.uint8), cv2.COLORMAP_TURBO
).squeeze(1)[:, ::-1]
TOP_DOWN_MAP_COLORS[MAP_INVALID_POINT] = [255, 255, 255]  # White
TOP_DOWN_MAP_COLORS[MAP_VALID_POINT] = [150, 150, 150]  # Light Grey
TOP_DOWN_MAP_COLORS[MAP_BORDER_INDICATOR] = [50, 50, 50]  # Grey
TOP_DOWN_MAP_COLORS[MAP_EXPLORED_AREA] = [205, 240, 255]  # Blue
TOP_DOWN_MAP_COLORS[MAP_TARGET_POINT_INDICATOR] = [200, 0, 0]  # Red
TOP_DOWN_MAP_COLORS[MAP_SHORTEST_PATH_COLOR] = [0, 200, 0]  # Green
TOP_DOWN_MAP_COLORS[MAP_VIEW_POINT_INDICATOR] = [245, 150, 150]  # Light Red
TOP_DOWN_MAP_COLORS[MAP_TARGET_BOUNDING_BOX] = [0, 175, 0]  # Green

WAY_POINT_RADIUS = 1
AGENT_RADIUS = 8
GOAL_RADIUS = 3

def depth_to_rgb(depth_map, resize=None):
    """Convert depth map to rgb map
    Args:
        depth_map: (h, w) depth map, normalized to [0, 1]
    """
    depth_map = (depth_map * 255).astype(np.uint8)
    rgb_map = cv2.applyColorMap(depth_map, cv2.COLORMAP_BONE)
    rgb_map = rgb_map[:, :, ::-1]
    
    if resize is not None:
        rgb_map = cv2.resize(rgb_map, (resize, resize))
    return rgb_map

def local_to_eps_frame(points: np.ndarray,
                       eps_pose: np.ndarray
                       ) -> np.ndarray:
    """Transform points from local frame to episodic frame.
    Args:
        points: ndarray, shape [num_points, 2], (x, y) in local frame, in m
        pose: ndarray, shape [3] (x, y, theta), in eps frame
    Returns:
        points: ndarray, shape [num_points, 2]
    
    """
    if eps_pose.ndim == 1:
        pose = eps_pose[None,:]
    else:
        pose = eps_pose
        
    x = points[..., 0] * np.cos(pose[:,2]) - points[...,1] \
        * np.sin(pose[:,2]) + pose[:,0] # [batch_size, num_points]
    y = points[..., 0] * np.sin(pose[:,2]) + points[...,1] \
        * np.cos(pose[:,2]) + pose[:,1] # [batch_size, num_points]
    
    return np.stack([x, y], axis=1) # [num_points, 2]

def eps_frame_to_local(p: np.ndarray,
                       eps_pose: np.ndarray
                       ) -> np.ndarray:
    """Transform points from episodic frame to local frame.
    Args:
        points: ndarray, shape [num_points, 2], (x, y) in eps frame, in m
        pose: ndarray, shape [3] (x, y, theta), in eps frame
        Returns:
            points: ndarray, shape [num_points, 2]
    """
    
    if p.ndim == 1:
        points = p[None,:]
    else:
        points = p
    pose = eps_pose
    R = np.array([[np.cos(pose[2]), -np.sin(pose[2])],
                  [np.sin(pose[2]), np.cos(pose[2])]])
    t = pose[None,:2] @ R # [1, 2]
    
    local_p = points[:,:2] @ R  - t # [num_points, 2]
    return local_p
                           
def draw_lines(wps):
    xs = []
    ys = []
    vals = []
    for i in range(wps.shape[0]-1):
        rr,cc, val = line_aa(int(wps[i,0]), int(wps[i,1]), int(wps[i+1,0]), int(wps[i+1,1]))
        xs.append(rr)
        ys.append(cc)
        vals.append(val)
    
    xs = np.concatenate(xs)
    ys = np.concatenate(ys)
    vals = np.concatenate(vals)
    return xs, ys, vals

class OccupancyMapVis():
    
    def __init__(self, map_args,grid_size=400) -> None:
        self.map_args = map_args
        self.grid_size = grid_size

    def pose_to_xy_local(self, pose):
        """Convert pose to xy coordinates in the map
        Args:
            pose: ndarray (3,) pose (x, y, theta)
            map_args: map arguments
        """
        map_args = self.map_args
        p = pose[:2] * 100 // map_args.map_resolution + \
            map_args.global_map_size_cm // 2 // map_args.map_resolution // map_args.global_downscaling
        p = p.astype(np.int32)
        p = p[1], p[0]
        return p
    
    def pose_to_xy_full_map_batch(self, pose):
        """Convert pose to xy coordinates in the map
        Args:
            pose: ndarray (n, 3) pose (x, y, theta)
            map_args: map arguments
        """
        map_args = self.map_args
        p = pose[:,:2] * 100 // map_args.map_resolution + \
            map_args.global_map_size_cm // 2 // map_args.map_resolution
        p = p.astype(np.int32)
        p = np.stack([p[:,1], p[:,0]], axis=1)
        return p

    def pose_to_xy_full_map(self, pose):
        """Convert pose to xy coordinates in the map
        Args:
            pose: ndarray (3,) pose (x, y, theta)
            map_args: map arguments
        """
        map_args = self.map_args
        p = pose[:2] * 100 // map_args.map_resolution + \
            map_args.global_map_size_cm // 2 // map_args.map_resolution
        p = p.astype(np.int32)
        p = p[1], p[0]
        return p
    
    def draw_agent_full_map(self,
                          map: np.ndarray,
                          pose: np.ndarray,
                          ) -> np.ndarray:
        """ Draw agent on map
        Args:
            map: rgb map (h, w, 3)
            pose: agent pose in episodic frame (x, y, theta)
        
        """
        half_size = self.map_args.global_map_size_cm // 2 // self.map_args.map_resolution
        xy = pose[:2] * 100 // self.map_args.map_resolution + half_size
        xy = xy.astype(np.int32)
        xy = xy[1], xy[0]
        theta = np.pi/2 - pose[2]
        # TODO: not use habitat's draw_agent
        map = maps.draw_agent(
            image=map,
            agent_center_coord=xy,
            agent_rotation=theta,
            agent_radius_px=AGENT_RADIUS,
        )
        return map

    def draw_agent_partial(self, map: np.ndarray) -> np.ndarray:
        """ Draw agent on map
        Args:
            map: rgb map (h, w, 3) in agent frame
            pose: agent pose in episodic frame (x, y, theta)
        
        """
        half_local_size = self.map_args.global_map_size_cm // 2 \
            // self.map_args.map_resolution // self.map_args.global_downscaling
        xy = [half_local_size, half_local_size]
        theta = np.pi/2 
        map = maps.draw_agent(
            image=map,
            agent_center_coord=xy,
            agent_rotation=theta,
            agent_radius_px=AGENT_RADIUS,
        )

        return map
    
    def visualize_partial_map(self, 
                              partial_map: np.ndarray,
                              pose: np.ndarray = None,
                              rel_path_pred: np.ndarray = None,
                              path_gt: np.ndarray = None,
                              rel_goal: np.ndarray = None,
                              resize: int = None,
                              ) -> np.ndarray:
        """Visualize partial map defined in agent frame
        Args:
            partial_map: ndarray (h, w) partial map (0 is free, 1 is occupied)
            rel_path_pred: ndarray (b, n, 2) path, each row is (x, y) in agent frame
            rel_goal: ndarray (2,) goal (x, y), in agent frame
            resize: int, resize the map
            path_gt: ndarray (n, 2) path, each row is (x, y) in episodic frame
        """
        
        map_args = self.map_args
        if isinstance(partial_map, torch.Tensor):
            partial_map = partial_map.cpu().detach().clone().numpy()
        
        partial_map = partial_map.astype(np.uint8)
        partial_map = TOP_DOWN_MAP_COLORS[partial_map]
        partial_map = self.draw_agent_partial(partial_map)

        # draw path
        if rel_path_pred is not None:
            if isinstance(rel_path_pred, torch.Tensor):
                rel_path_pred = rel_path_pred.cpu().detach().clone().numpy()
            else:
                rel_path_pred = np.copy(rel_path_pred) 

            if rel_path_pred.ndim == 2:
                rel_path_pred = rel_path_pred[None,...]
                
            for b in range(rel_path_pred.shape[0]):
                for i in range(rel_path_pred.shape[1]):
                    p = self.pose_to_xy_local(rel_path_pred[b,i])
                    p = np.clip(p, 0+WAY_POINT_RADIUS, partial_map.shape[0]-WAY_POINT_RADIUS-1)
                    partial_map[p[0]-WAY_POINT_RADIUS:p[0]+WAY_POINT_RADIUS, 
                            p[1]-WAY_POINT_RADIUS:p[1]+WAY_POINT_RADIUS] = TOP_DOWN_MAP_COLORS[70+10*b]
        
        # draw goal
        if rel_goal is not None:
            if isinstance(rel_goal, torch.Tensor):
                rel_goal = rel_goal.cpu().detach().clone().numpy()
            p = self.pose_to_xy_local(rel_goal)
            rel_goal = np.clip(p, 0+WAY_POINT_RADIUS, partial_map.shape[0]-WAY_POINT_RADIUS-1)
            partial_map[rel_goal[0]-GOAL_RADIUS:rel_goal[0]+GOAL_RADIUS,
                rel_goal[1]-GOAL_RADIUS:rel_goal[1]+GOAL_RADIUS] = TOP_DOWN_MAP_COLORS[MAP_TARGET_POINT_INDICATOR]

        if path_gt is not None:
            if isinstance(path_gt, torch.Tensor):
                path_gt = path_gt.cpu().detach().clone().numpy()
            else:
                path_gt = np.copy(path_gt) 
            
            path_gt = eps_frame_to_local(path_gt, pose)

            for i in range(path_gt.shape[0]):
                p = self.pose_to_xy_local(path_gt[i])
                p = np.clip(p, 0+WAY_POINT_RADIUS, partial_map.shape[0]-WAY_POINT_RADIUS-1)
                partial_map[p[0]-WAY_POINT_RADIUS:p[0]+WAY_POINT_RADIUS, 
                        p[1]-WAY_POINT_RADIUS:p[1]+WAY_POINT_RADIUS] = TOP_DOWN_MAP_COLORS[MAP_TARGET_POINT_INDICATOR]
        
        if resize is not None:
            partial_map = cv2.resize(partial_map, (resize, resize))

        return partial_map

    def visualize_full_map(self,
                         full_map: np.ndarray,
                         pose: np.ndarray = None,
                         rel_goal: np.ndarray = None,
                         rel_path_pred: np.ndarray = None,
                         resize: int = None,
                         exp_maps: np.ndarray = None,
                         path_gt: np.ndarray = None,
                         ) -> np.ndarray:
        """Visualize map, pose, goal, and path
        Args:
            full_map: ndarray (h, w) full map (0 is free, 1 is occupied)
            pose: ndarray (3,) pose (x, y, theta), in episodic frame
            rel_goal: ndarray (2,) goal (x, y), in agent frame
            rel_path_pred: ndarray (n, 2) path, each row is (x, y) in agent frame
            resize: int, resize the map
            exp_maps: ndarray (h, w) exp map (0 is unexplored, 1 is explored)
            path_gt: ndarray (n, 2) path, each row is (x, y) in episodic frame
        """
        map_args = self.map_args
        
        if isinstance(full_map, torch.Tensor):
            full_map = full_map.cpu().detach().clone().numpy()
        full_map = full_map.astype(np.uint8)

        
        # convert map to rgb
        rgb_map = TOP_DOWN_MAP_COLORS[full_map]
        
        # draw agent
        if pose is not None:
            if isinstance(pose, torch.Tensor):
                pose = pose.cpu().detach().clone().numpy()
            else:
                pose = np.copy(pose)
                
            rgb_map = self.draw_agent_full_map(rgb_map, pose)

        # overlay with exp map
        if exp_maps is not None:
            if isinstance(exp_maps, torch.Tensor):
                exp_maps = exp_maps.cpu().detach().clone().numpy()
            exp_maps = exp_maps.astype(np.uint8)
            rgb_map += TOP_DOWN_MAP_COLORS[exp_maps * MAP_EXPLORED_AREA] * exp_maps[:,:,None]
            rgb_map = np.clip(rgb_map, 0, 255)
        
        # draw goal
        if rel_goal is not None:
            if isinstance(rel_goal, torch.Tensor):
                rel_goal = rel_goal.cpu().detach().clone().numpy()

            goal_eps_frame = local_to_eps_frame(rel_goal,pose)[0]
            rel_goal = goal_eps_frame
            rel_goal = self.pose_to_xy_full_map(rel_goal)
            rgb_map[rel_goal[0]-GOAL_RADIUS:rel_goal[0]+GOAL_RADIUS,
                rel_goal[1]-GOAL_RADIUS:rel_goal[1]+GOAL_RADIUS] = TOP_DOWN_MAP_COLORS[MAP_TARGET_POINT_INDICATOR]


        # draw gt path
        if path_gt is not None and len(path_gt.shape) > 0 :
            if isinstance(path_gt, torch.Tensor):
                path_gt = path_gt.cpu().detach().clone().numpy()

            path_gt = self.pose_to_xy_full_map_batch(path_gt)
            xs, ys, vals = draw_lines(path_gt)
            rgb_map[xs, ys] = TOP_DOWN_MAP_COLORS[MAP_SHORTEST_PATH_COLOR]

        # draw path
        if rel_path_pred is not None:
            if isinstance(rel_path_pred, torch.Tensor):
                rel_path_pred = rel_path_pred.cpu().detach().clone().numpy()
            else:
                rel_path_pred = np.copy(rel_path_pred) 

            if rel_path_pred.ndim == 2:
                rel_path_pred = rel_path_pred[None,...]
            
            for b in range(rel_path_pred.shape[0]):
                # first convert path to episodic frame
                rel_path_pred_b = local_to_eps_frame(rel_path_pred[b], pose)
                xys = self.pose_to_xy_full_map_batch(rel_path_pred_b)

                xs, ys, vals = draw_lines(xys)
                rgb_map[xs, ys] = TOP_DOWN_MAP_COLORS[70+10*b]

                # for i in range(xys.shape[0]):
                #     p = xys[i]
                #     rgb_map[p[0]-WAY_POINT_RADIUS:p[0]+WAY_POINT_RADIUS, 
                #             p[1]-WAY_POINT_RADIUS:p[1]+WAY_POINT_RADIUS] = TOP_DOWN_MAP_COLORS[70+10*b]
            
        # crop a local rgb map around the agent
        local_size = 240
        agent_xy = self.pose_to_xy_full_map(pose)
        local_map = rgb_map[max(agent_xy[0]-local_size,0):min(agent_xy[0]+local_size,960), 
                            max(agent_xy[1]-local_size,0):min(agent_xy[1]+local_size,960)]

        if resize is not None:
            local_map = cv2.resize(local_map, (resize, resize))
        return rgb_map, local_map

    def visualize(self,
                  rgb: np.ndarray = None,
                  depth: np.ndarray = None,
                  third_person: np.ndarray = None,
                  full_map: np.ndarray = None,
                  partial_map: np.ndarray = None,
                  rel_goal: np.ndarray = None,
                  goal: np.ndarray = None,
                  gt_path: np.ndarray = None,
                  rel_gt_path: np.ndarray = None,
                  pose: np.ndarray = None,
                  rel_pred_path: np.ndarray = None,
                  ) -> np.ndarray:
        """Generate visualization for the given observation.
        Args:
            rgb: (H,W,3) RGB image
            depth: (H,W) depth image
            third_person: (H,W,3) RGB image
            map: (H,W,2) map image
            goal: (2,) goal in current agent frame
            gt_path: (n,2) path in eps frame
            pose: (3,) current pose in eps frame
            
        Returns:
            img: (H,W,3) RGB image

        """
        
        if rel_gt_path is not None:
            gt_path = local_to_eps_frame(rel_gt_path, pose)
        if goal is not None:
            rel_goal = eps_frame_to_local(goal, pose)[0]
            
        imgs = []
        if rgb is not None:
            rgb = rgb.astype(np.uint8)
            rgb = cv2.resize(rgb, (self.grid_size, self.grid_size))
            imgs.append(rgb)

        if depth is not None:
            depth = depth_to_rgb(depth, self.grid_size).astype(np.uint8)       
            imgs.append(depth)


        if partial_map is not None \
            and rel_goal is not None \
            and pose is not None:
                
            partial_map_img = self.visualize_partial_map(partial_map, 
                                                        rel_goal=rel_goal, 
                                                        path_gt=gt_path, 
                                                        pose=pose,
                                                        rel_path_pred=rel_pred_path,
                                                        resize=self.grid_size)
            partial_map_img = np.flipud(partial_map_img).astype(np.uint8)
            imgs.append(partial_map_img)
            
        if full_map is not None \
            and rel_goal is not None \
            and pose is not None:
            full_map_img, local_map_img = self.visualize_full_map(full_map, 
                                                                pose=pose, 
                                                                rel_goal=rel_goal, 
                                                                path_gt=gt_path,
                                                                rel_path_pred=rel_pred_path,
                                                                resize=self.grid_size)
            local_map_img = np.flipud(local_map_img).astype(np.uint8)
            imgs.append(local_map_img)

        num_grid = len(imgs)
        row_num = 2 if num_grid > 2 else 1
        col_num = np.ceil(num_grid / row_num).astype(np.int32)
        canvas = np.zeros((self.grid_size * row_num, self.grid_size * col_num, 3), dtype=np.uint8)
        
        for i in range(num_grid):
            row = i // col_num
            col = i % col_num
            canvas[row*self.grid_size:(row+1)*self.grid_size, 
                   col*self.grid_size:(col+1)*self.grid_size] = imgs[i]

        return canvas