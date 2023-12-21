
#!/usr/bin/env python3

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from habitat_sim.utils.viz_utils import semantic_to_rgb
import imageio
from habitat.utils.visualizations import maps
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
from habitat.utils.visualizations.utils import images_to_video
import os

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg


import numpy as np
import torch



def render_plt_image(data):
    """
    Render image using matplotlib, and return the rendered image
    Args:
        data: tensor of size (C, H, W) or np [h, w, 3]
    """
    fig = Figure(figsize=(8, 8))
    canvas = FigureCanvasAgg(fig)
    
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()

    ax = fig.add_axes([0, 0, 1, 1], frameon=False, xticks=[], yticks=[])
    ax.imshow(data,cmap='coolwarm',vmin=0,vmax=1.2)
    # ax.axis('off')
    # plt.gcf().delaxes(plt.gca())
    # fig.tight_layout()
    canvas.draw()  # Draw the canvas, cache the renderer

    buf = canvas.buffer_rgba()
    # convert to a NumPy array
    image = np.asarray(buf)
    return image

def save_image(sm,file_name=None,dir=None,add_time=False):
    if dir is None:
        dir = 'out/'
    if file_name is None:
        file_name = 'tmp.png'
    if not os.path.isdir(dir):
        os.makedirs(dir)
        
    parts = file_name.split('.')
    name = parts[0]
    if len(parts)==2:
        ext = parts[1]
    else:
        ext = 'png'
    if add_time:
        t = datetime.now().strftime('%d_%H_%M')
        file_name = f'{name}_{t}.{ext}'
    else:
        file_name = f'{name}.{ext}'

    # plt.imshow(sm)
    # plt.savefig(f'{dir}/{file_name}')
    imageio.imsave(f'{dir}/{file_name}',sm)

def save_img_tensor(tensor,fn,dir=None):
    if tensor.shape[0]==3:
        img = tensor.permute(1,2,0)
    else:
        img = tensor
    save_image(img.detach().cpu().numpy(),fn,dir)
    
def plot_image(sm):
    """
    plot image using matplot lib
    """
    # sm = semantic_to_rgb(sm)
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.axis('off')
    ax.imshow(sm)
     
    
    plt.show()

def plot_multiple_images(images,row=1):

    """
    plot multiple images using matplot lib
    """
    col = len(images) // row
    _, axes = plt.subplots(row, col, figsize=(12, 18))
    for i in range(row):
        for j in range(col):
            np_img = images[i*col+j]
            if isinstance(np_img, torch.Tensor):
                np_img = np_img.cpu().detach().numpy()
            axes[i,j].imshow(np_img)
    plt.show()

def display_grayscale(image):
    img_bgr = np.repeat(image, 3, 2)
    cv2.imshow("Depth Sensor", img_bgr)
    return cv2.waitKey(0)


def display_rgb(image):
    img_bgr = image[..., ::-1]
    cv2.imshow("RGB", img_bgr)
    return cv2.waitKey(0)

def draw_top_down_map(info, output_size):
    return maps.colorize_draw_agent_and_fit_to_height(
        info["top_down_map"], output_size
    )


def img_frombytes(data):
    """
    Used to solve a PIL bug that can't import bool array properly
    """
    size = data.shape[::-1]
    databytes = np.packbits(data, axis=1)
    return Image.frombytes(mode='1', size=size, data=databytes)

def annotate_map_with_goal(map, goal):
    """
    Annotate the map with the goal location
    """
    if goal is not None:
        draw = ImageDraw.Draw(map)
        y, x = goal
        r = 5
        if map.mode =='L':
            color = 118
        else:
            color = (0, 255, 0,255)
        draw.ellipse((x-r, y-r, x+r, y+r), fill=color)
    return map

def tile_images(images):
    r"""Tile multiple images into single image

    Args:
        images: list of images where each image has dimension
            (height x width x channels)

    Returns:
        tiled image (new_height x width x channels)
    """
    assert len(images) > 0, "empty list of images"
    np_images = np.asarray(images)
    n_images, height, width, n_channels = np_images.shape
    new_height = int(np.ceil(np.sqrt(n_images)))
    new_width = int(np.ceil(float(n_images) / new_height))
    # pad with empty images to complete the rectangle
    np_images = np.array(
        images
        + [images[0] * 0 for _ in range(n_images, new_height * new_width)]
    )
    # img_HWhwc
    out_image = np_images.reshape(
        new_height, new_width, height, width, n_channels
    )
    # img_HhWwc
    out_image = out_image.transpose(0, 2, 1, 3, 4)
    # img_Hh_Ww_c
    out_image = out_image.reshape(
        new_height * height, new_width * width, n_channels
    )
    return out_image


class Recording():
    def __init__(self) -> None:
        self._images = []

    # def add_frame(self,rgb,depth,map, gt_local,pred_local):
    #     img = visualize_gt(rgb,depth,map, gt_local,pred_local)
    #     self._images.append(img)
    #     return img

    def add_frame(self,img):
        self._images.append(img)

    def save_video(self,fname, dir='recordings'):
        images_to_video(self._images, dir, fname)
        self._images = []

    def clear(self):
        self._images = []


