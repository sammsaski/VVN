# standard library
import os
import json

# imported packages
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import lightning as L

# local imports

NUM_CLASSES = 10
IMG_RES = 32
NUM_FRAMES = 8

def write_callback_metrics(filepath, metrics):
    with open(filepath, 'w') as json_file:
        json.dump(filepath, metrics, indent=4)

def train_zoom_in():
    # Load the training + testing datasets
    video_filepath = ''
    label_filepath = ''

    test_video_filepath = ''
    test_label_filepath = ''

    # Create the training, validation sets
    data = MNISTVideo(video_filepath=video_filepath, label_filepath=label_filepath)

    train_size = int(0.8 * len(data))
    val_size = len(data) - train_size
    train_data, val_data = random_split(data, [train_size, val_size])

    # Create the testing set
    test_data = MNISTVideo(video_filepath=test_video_filepath, label_filepath=test_label_filepath)

    train_loader = torch.utils.DataLoader(train_data, batch_size=16, shuffle=True)
    val_loader = torch.utils.DataLoader(val_data, batch_size=16, shuffle=True)
    test_loader = torch.utils.DataLoader(test_data, batch_size=16, shuffle=False)

    # Training sequence
    model = C3D(img_res=IMG_RES, num_frames=NUM_FRAMES, num_classes=NUM_CLASSES)
    trainer = L.Trainer(max_epochs=75)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloader=val_loader)

    # Testing sequence
    trainer.test(model=model, dataloaders=test_loader)

    # Save the model
    model.eval()
    dummy_input = torch.randn(1, 8, 32, 32)
    onnx_path = os.path.join('', '')
    torch.onnx.export(model,
                      dummy_input,
                      onnx_path,
                      opset_version=11,
                      input_names=['input'],
                      output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
    
    # convert metrics to float from tensor
    metrics = trainer.callback_metrics
    metrics = {key: float(value) for key, value in metrics.items()}

    # write the metrics to a json file
    write_callback_metrics('', metrics)


def train_zoom_out():
    # Load the training + testing datasets
    video_filepath = ''
    label_filepath = ''

    test_video_filepath = ''
    test_label_filepath = ''

    # Create the training, validation sets
    data = MNISTVideo(video_filepath=video_filepath, label_filepath=label_filepath)

    train_size = int(0.8 * len(data))
    val_size = len(data) - train_size
    train_data, val_data = random_split(data, [train_size, val_size])

    # Create the testing set
    test_data = MNISTVideo(video_filepath=test_video_filepath, label_filepath=test_label_filepath)

    train_loader = torch.utils.DataLoader(train_data, batch_size=16, shuffle=True)
    val_loader = torch.utils.DataLoader(val_data, batch_size=16, shuffle=True)
    test_loader = torch.utils.DataLoader(test_data, batch_size=16, shuffle=False)

    # Training sequence
    model = C3D(img_res=IMG_RES, num_frames=NUM_FRAMES, num_classes=NUM_CLASSES)
    trainer = L.Trainer(max_epochs=75)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloader=val_loader)

    # Testing sequence
    trainer.test(model=model, dataloaders=test_loader)

    # Save the model
    model.eval()
    dummy_input = torch.randn(1, 8, 32, 32)
    onnx_path = os.path.join('', '')
    torch.onnx.export(model,
                      dummy_input,
                      onnx_path,
                      opset_version=11,
                      input_names=['input'],
                      output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
    
    # convert metrics to float from tensor
    metrics = trainer.callback_metrics
    metrics = {key: float(value) for key, value in metrics.items()}

    # write the metrics to a json file
    write_callback_metrics('', metrics)


if __name__=="__main__":
    print('Starting C3D model training ...')
    print('Starting C3D Model training on Zoom In video dataset.')
    
    train_zoom_in() # perform training
    
    print('Completed C3D Model training on Zoom In video dataset.')
    print('Starting C3D Model training on Zoom Out video dataset.')
    
    train_zoom_out() # perform training
    
    print('Completed C3D Model training on Zoom Out video dataset.')
