
import numpy as np
import torch
import cv2

from matplotlib import pyplot as plt

class OccupancyMapVis():
    
    def __init__(self) -> None:
        self.canvas_size = (1500,500)
    
    def visualize(self,
                  rgb: np.ndarray = None,
                  depth: np.ndarray = None,
                  third_person: np.ndarray = None,
                  map: np.ndarray = None,
                  ) -> np.ndarray:
        """Generate visualization for the given observation.
        Args:
            rgb: (H,W,3) RGB image
            depth: (H,W) depth image
            third_person: (H,W,3) RGB image
            map: (H,W,2) map image
            
        Returns:
            img: (H,W,3) RGB image

        """
        
        self.canvas = np.zeros((1000,1000,3), dtype=np.uint8)
        rgb = cv2.resize(rgb, (500,500), interpolation=cv2.INTER_NEAREST)

        occ_map = cv2.cvtColor((map[0]*255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        occ_map = cv2.resize(occ_map, (500,500))
        exp_map = cv2.cvtColor(map[1]*255, cv2.COLOR_GRAY2RGB)
        exp_map = cv2.resize(exp_map, (500,500))
        exp_map = exp_map.astype(np.uint8)
        
        img = np.concatenate((rgb,occ_map,exp_map), axis=1)
        return img