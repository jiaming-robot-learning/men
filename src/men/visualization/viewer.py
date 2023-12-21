
import cv2
import numpy as np
import time
# ENV = "home_robot"

# if ENV == "home_robot":
#     from home_robot.core.interfaces import DiscreteNavigationAction, ContinuousFullBodyAction
    
#     ACTIONS = {
#         "a": DiscreteNavigationAction.TURN_LEFT,
#         "d": DiscreteNavigationAction.TURN_RIGHT,
#         "s": DiscreteNavigationAction.STOP,
#         "w": DiscreteNavigationAction.MOVE_FORWARD,
#         "i": ContinuousFullBodyAction(np.array([0.1] + [0] * 3 + [0] + [0] + [0] * 4)), # arm forward
#         "k": ContinuousFullBodyAction(np.array([-0.1] + [0] * 3 + [0] + [0] + [0] * 4)), # arm backward
#         "o": ContinuousFullBodyAction(np.array([0] + [0] * 3 + [0.1] + [0] + [0] * 4)), # arm up
#         "l": ContinuousFullBodyAction(np.array([0] + [0] * 3 + [-0.1] + [0] + [0] * 4)), # arm down
#     }

class OpenCVViewer:
    def __init__(self, 
                 action_map=None,
                 name="OpenCVViewer", 
                 exit_on_escape=True):
        self.action_map = action_map
        self.name = name
        cv2.namedWindow(name, cv2.WINDOW_NORMAL) # use cv2.WINDOW_NORMAL to allow window resizing for large images
        self.exit_on_escape = exit_on_escape
        if action_map is None:
            self.non_blocking = True
        else:
            self.non_blocking = False

    def imshow(self, image: np.ndarray, rgb=True, delay=0):
        if image.ndim == 2:
            image = np.tile(image[..., np.newaxis], (1, 1, 3))
        elif image.ndim == 3 and image.shape[-1] == 1:
            image = np.tile(image, (1, 1, 3))
        assert image.ndim == 3, image.shape

        if rgb:
            image = image[..., ::-1]
        cv2.imshow(self.name, image)

        if self.non_blocking:
            cv2.waitKey(200)
            return
        else:
            action = None
            info = ""
            
            key = cv2.waitKey(delay)
            if key == 27:  # escape
                exit(0)
            elif key == -1:  # timeout
                info = "timeout"
            else:
                c = chr(key).lower()        
                action = self.action_map.get(c, None)
                info = c

            return {'action':action, 'info':info}
                

    def close(self):
        cv2.destroyWindow(self.name)

    def __del__(self):
        self.close()