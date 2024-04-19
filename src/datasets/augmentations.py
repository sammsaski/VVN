# standard library
from enum import Enum

# imported packages
from torchvision import transforms

# Supported augmentations; used for safe applications of augmentations
class Variation(Enum):
    ZoomIn = "zoom_in"
    ZoomOut = "zoom_out"
    Rotating = "rotating"
 
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
        pad = [
            tensor_img.size(1) - current_size[0],
            tensor_img.size(2) - current_size[1],
        ]

        # Resize the image
        resized_img = transforms.functional.resize(tensor_img, current_size)

        # Pad the image
        padded_img = transforms.functional.pad(
            resized_img,
            padding=(pad[1]//2, pad[0]//2, pad[1]-pad[1]//2, pad[0]-pad[0]//2),
            padding_mode="constant",
            fill=0,
        )

        # Add the padded image to the the list of new frames
        frames.append(padded_img)

    return frames

def zoom_in(tensor_img, steps=20, zoom_factor=1.2):
    frames = []
    for step in range(steps):
        # Calculate new size and padding
        current_size = [
            int(tensor_img.size(1) * (zoom_factor ** (steps - step - 1))),
            int(tensor_img.size(2) * (zoom_factor ** (steps - step - 1)))
        ]
            
        # calculate cropping
        crop_size = [tensor_img.size(1) - current_size[0], tensor_img.size(2) - current_size[1]]

        # crop and resize the image
        cropped_img = transforms.functional.crop(
            tensor_img, top=crop_size[0]//2, left=crop_size[1]//2, height=current_size[0], width=current_size[1]
        )
        resized_img = transforms.functional.resize(cropped_img, [tensor_img.size(1), tensor_img.size(2)])

        # add the resized image to the list of frames
        frames.append(resized_img)

    return frames


