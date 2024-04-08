import torch
import torchvision
import torchvision.datasets as datasets
from torchvision import transforms

import numpy as np

import matplotlib.pyplot as plt
from PIL import Image
import imageio

import os

from enum import Enum

class Variation(Enum):
    ZoomIn = "zoom_in"
    ZoomOut = "zoom_out"
    Rotating = "rotating"
    

class MNISTVideoDataset:
    def __init__(self, mnist):
        self.mnist = mnist

    def zoom_out(tensor_img, steps=20, zoom=0.95):
        """Generate zoom out video of n frames from an original image.

        :param tensor_img: the original image
        :type tensor_img: torch.Tensor
        :param steps: number of frames, defaults to 20
        :type steps: int, optional
        :param zoom: the zoom magnitude per step, defaults to 0.95
        :type zoom: float, optional
        """        
        frames = []
        for step in range(steps):
            # Calculate new size and padding
            current_size = [
                int(tensor_img.size(1) * (zoom**step)),
                int(tensor_img.size(2) * (zoom**step)),
            ]
            padding = [
                tensor_img.size(1) - current_size[0],
                tensor_img.size(2) - current_size[1],
            ]

            # Resize the image
            resized_img = transforms.functional.resize(tensor_img, current_size)

            # Pad the image
            padded_img = transforms.functional.pad(
                resized_img,
                padding[1] // 2,
                padding[0] // 2,
                padding[1] - padding[1] // 2,
                padding[0] - padding[0] // 2,
                padding_mode="constant",
                fill=0,
            )

            # Add the padded image to the the list of new frames
            frames.append(padded_img)

        return frames

    def zoom_in():
        pass

    def create_dataset():

