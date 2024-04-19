# standard library
import os

# imported packages
import torch
import torchvision
import torchvision.datasets as datasets
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import imageio

# local imports
import augmentations
from augmentations import Variation

class MNISTVideoGenerator:
    def __init__(self, mnist):
        self.mnist = mnist # reference to the original MNIST dataset data loader

    def create_dataset(self, filepath, train=True, augment=Variation.ZoomIn, new_size=(64, 64)):
        frames = []
        labels = []
        for batch_idx, (data, target) in enumerate(self.mnist):
            aug = [] # the augmented frames in the video
            image = data[0]

            # --- 1. convert image to 64x64 --- #
            to_pil_image = transforms.ToPILImage()
            image = to_pil_image(image.squeeze(0))

            # create a new 64x64 image of black pixels
            new_image = Image.new('L', new_size, 'black')

            # calculate where in the image to paste the original image (which was 28x28)
            left = (new_size[0] - image.width) // 2
            top = (new_size[1] - image.height) // 2

            # paste the original image in the new image
            new_image.paste(image, (left, top))

            # convert back to tensor
            new_image_tensor = transforms.ToTensor()(new_image)

            # --- 2. generate frames --- #
            if augment == Variation.ZoomIn:
                aug = augmentations.zoom_in(new_image_tensor) # list of frames in the video
    
            elif augment == Variation.ZoomOut:
                aug = augmentations.zoom_out(new_image_tensor)
                
            else:
                raise Exception("Please provide a valid augmentation.")

            # convert to np.array
            np_aug = np.array(aug)

            # add video to dataset
            frames.append(np_aug)

            # --- 3. add the labels to the dataset --- #
            labels.append(target)

        # convert dataset to numpy for saving
        dataset = np.array(frames)
        labels = np.array(labels)

        # build filename
        data_filename = f'mnistvideo_{augment.value}_seq.npy'
        labels_filename = f'mnistvideo_{augment.value}_seq.npy'

        # save as .npy file
        np.save(os.path.join(filepath, data_filename), dataset) # video frames
        np.save(os.path.join(filepath, labels_filename), labels) # labels
        
        return

    def load_dataset(self):
        pass

    def create_gif(self):
        pass

    def show_gif(self):
        pass


if __name__ == "__main__":
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])

    # download the MNIST dataset
    train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # create the data loaders
    train_loader = torch.utils.data.DataLoader(train, batch_size=1, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=1, shuffle=False)

    # instantiate MNISTVideo
    MNISTVideo = MNISTVideoGenerator(mnist=train_loader)

    # create the MNISTVideo dataset from the original MNIST dataset
    MNISTVideo.create_dataset()


