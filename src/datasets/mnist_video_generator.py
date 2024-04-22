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
        train_frames = []
        test_frames = []
        train_labels = []
        test_labels = []

        # keep track of number of samples per class
        train_class_count = {i: 0 for i in range(10)}
        test_class_count = {i: 0 for i in range(10)}

        # when train AND test completion == 10; stop.
        train_completion = 0
        test_completion = 0

        for batch_idx, (data, target) in enumerate(self.mnist):

            # if we've already satisfied the requirements for current class in
            # both train and test, then skip it
            if train_class_count[int(target)] == 1000 and test_class_count[int(target)] == 100:
                continue

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
                aug = augmentations.zoom_in(new_image_tensor, steps=8, zoom_factor=1.3) # list of frames in the video
    
            elif augment == Variation.ZoomOut:
                aug = augmentations.zoom_out(new_image_tensor, steps=8, zoom_factor=0.9)
                
            else:
                raise Exception("Please provide a valid augmentation.")

            # convert to np.array
            np_aug = np.array(aug)

            # for logging
            if batch_idx % 1000 == 0:
                print(f'Batch {batch_idx} of {len(self.mnist)}')
            
            if train_class_count[int(target)] < 1000:
                # add video to dataset
                train_frames.append(np_aug)
            
                # add the labels to the dataset
                train_labels.append(target)

                train_class_count[int(target)] += 1
                if train_class_count[int(target)] == 1000:
                    train_completion += 1
                
            elif test_class_count[int(target)] < 100"
                # add video to dataset
                test_frames.append(np_aug)

                # add the labels to the dataset
                test_labels.append(target)

                test_class_count[int(target)] += 1
                if test_class_count[int(target)] == 100:
                    test_completion += 1
            
            # stopping criteria
            if train_completion == 10 and test_completion == 10:
                break

        # dataset conversion to .npy
        training_set = np.array(train_frames)
        training_labels = np.array(train_labels)
        testing_set = np.array(test_frames)
        testing_labels = np.array(test_labels)

        # build filename
        training_data_filename = f'mnistvideo_{augment.value}_{new_size[0]}x{new_size[1]}_train_data_seq.npy'
        training_labels_filename = f'mnistvideo_{augment.value}_{new_size[0]}x{new_size[1]}_train_labels_seq.npy'
        testing_data_filename = f'mnistvideo_{augment.value}_{new_size[0]}x{new_size[1]}_test_data_seq.npy'
        testing_labels_filename = f'mnistvideo_{augment.value}_{new_size[0]}x{new_size[1]}_test_labels_seq.npy'

        # save as .npy file
        np.save(os.path.join(filepath, training_data_filename), training_set) # video frames
        np.save(os.path.join(filepath, training_labels_filename), training_labels) # labels
        np.save(os.path.join(filepath, testing_data_filename), testing_set) # video frames
        np.save(os.path.join(filepath, testing_labels_filename), testing_labels) # labels
        
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

    # create the MNIST data loaders
    train_loader = torch.utils.data.DataLoader(train, batch_size=1, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=1, shuffle=False)

    # instantiate MNISTVideoGenerator
    MNISTVideoTrain = MNISTVideoGenerator(mnist=train_loader)
   
    # create the MNISTVideo dataset from the original MNIST dataset
    MNISTVideoTrain.create_dataset(filepath='', augment=Variation.ZoomOut, new_size=(32, 32))
    MNISTVideoTrain.create_dataset(filepath='', augment=Variation.ZoomIn, new_size=(32, 32))


