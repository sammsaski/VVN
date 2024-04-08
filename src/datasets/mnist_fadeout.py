import torch
import torchvision
import torchvision.datasets as datasets
from torchvision import transforms

import numpy as np

import matplotlib.pyplot as plt
from PIL import Image
import imageio

import os


def zoom_out(tensor_img, steps=20, zoom=0.95):
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


def create_dataset(data_loader, save_path, new_size=(64, 64), fps=20):
    videos = []

    for batch_idx, (data, target) in enumerate(data_loader):
        image = data[0]

        #########################################
        ### --- 1. CONVERT IMAGE TO 64x64 --- ###
        #########################################
        image = transforms.ToPILImage()(image.squeeze(0))

        # Create a new 64x64 image of black pixels
        new_image = Image.new("L", new_size, "black")

        # Calculate where in the image to paste the original iamge (which was 28x28)
        left = (new_size[0] - image.width) // 2
        top = (new_size[1] - image.height) // 2

        # Paste the original image in the new image
        new_image.paste(image, (left, top))

        # Convert back to tensor
        new_image_tensor = transforms.ToTensor()(new_image)

        #############################################
        ### --- 2. GENERATE ZOOMED OUT FRAMES --- ###
        #############################################
        frames = zoom_out(new_image_tensor)

        ##########################################
        ### --- 3. CREATE AND SAVE THE GIF --- ###
        ##########################################
        frames_to_pil = [transforms.ToPILImage()(frame) for frame in frames]
        # imageio.mimsave(os.join(save_path, f"{batch_idx}.gif"), frames_to_pil, fps=fps)
        np_frames = np.array(frames_to_pil)
        videos.append(np_frames)

    return videos


if __name__ == "__main__":
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
        ]
    )

    train = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    train_loader = torch.utils.data.DataLoader(train, batch_size=1, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=1, shuffle=False)

    # Create the training dataset
    videos = create_dataset()

    # Save it to .npy file
    np.save("data/mnist_zoom_out.npy", videos)
