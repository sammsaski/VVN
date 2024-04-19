# standard library
import os
from typing import Any

# imported packages
import torch
import torchvision
import torchvision.datasets as datasets
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import imageio
from IPython.display import Image as ImageDisplay
from IPython.display import display

# local imports
import augmentations
from augmentations import Variation

def create_gif(images, filepath):
    frames = images.squeeze(0)
    frames_to_pil = [transforms.ToPILImage()(frame) for frame in frames]
    imageio.mimsave(filepath, frames_to_pil, fps=20)

class MNISTVideo(datasets.vision.VisionDataset):
    def __init__(self, video_filepath, label_filepath): 
        self.video_filepath = video_filepath
        self.label_filepath = label_filepath

        # load the data
        self.data = np.load(video_filepath)
        self.labels = np.load(label_filepath)

    def __getitem__(self, index: int) -> Any:
        video, label = self.data[index], int(self.labels[index])
        return video, label

    def __len__(self) -> int:
        return len(self.data) 

if __name__ == "__main__":
    video_filepath = ''
    label_filepath = ''

    # load the data as a torchvision dataset
    data = MNISTVideo(video_filepath=video_filepath, label_filepath=label_filepath)

    # load with data loader
    train_loader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=True)
    dataiter = iter(train_loader)

    # get a sample
    images, labels = next(dataiter)

    # create, save, and display gif of the sample
    gif_filepath = 'sample.gif'
    create_gif(images, gif_filepath)
    display(ImageDisplay(filename=gif_filepath))

