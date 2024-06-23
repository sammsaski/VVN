# python standard library
import csv
import onnxruntime
import os
import random
from collections import defaultdict
from typing import List

# third-party packages
import numpy as np

# local modules
from vvn.config import Config

# define global variables
PATH_TO_MODELS = os.path.join(os.getcwd(), 'models')
PATH_TO_DATA = os.path.join(os.getcwd(), 'data')

# set seed
random.seed(42)

def prepare_filetree(config: Config):
    # TODO: come up with a more flexible way to do this
    # create all directories/files for each type of experiment being run
    for sgt in ['random', 'inorder']: 
        for at in ['single_frame', 'all_frames']:
            for dst in ['zoom_in', 'zoom_out']:
                for length in ['4', '8', '16']:
                    for eps_filename in [f'eps={e}_255' for e in range(1, 4)]:
                        fp = build_output_filepath(config, eps_filename)

                        # create the parent directories if they don't already exist
                        os.makedirs(os.path.join(output_dir, sgt, at, dst, length), exist_ok=True)

                        # if the file doesn't exist yet, create it
                        if not os.path.isfile(fp):
                            with open(fp, 'a', newline='') as f:
                                # write CSV headers
                                writer = csv.writer(f)
                                writer.writerow(['Sample Number', 'Result', 'Time', 'Method'])

def build_output_filepath(config: Config, filename=None, parent_only=False):
    """
    For our purposes, we will be naming the file based on
    the epsilon value used for the verification.
    """
    # error handling
    if not filename and not parent_only:
        raise Exception(f'No filename given. Please provide a filename when parent_only={parent_only}.')

    # convert all config values to str
    str_config = config.tostr()

    # get the values we need for building the output filepath
    output_dir = str_config.output_dir
    sgt = str_config.sample_gen_type
    attack_type = str_config.attack_type
    dst = str_config.ds_type
    length = str_config.sample_len

    fp = os.path.join(output_dir, sgt, attack_type, dst, length)

    return fp if parent_only else os.path.join(fp, filename) + '.csv'

# TODO: check that the output from the models is the exact same
#       whether in python or matlab
def get_correct_samples(ds_type, sample_len, modelpath, datapath) -> List[int]:
    # check that ds_type + sample_len are the correct types/values
    # TODO: consider making data directory naming more consistent
    ds_type = "ZoomIn" if ds_type == "zoom_in" else "ZoomOut" # have to convert because of naming conventions
    sample_len = str(sample_len)
    
    # load the data + labels; example : VVN/data/ZoomOut/test/mnistvideo_zoom_out_4f_test_dat_seq.npy
    data = np.load(os.path.join(datapath, ds_type, 'test', f'mnistvideo_{ds_type}_{sample_len}f_test_data_seq.npy'))
    labels = np.load(os.path.join(datapath, f'mnistvideo_{ds_type}_test_labels_seq.npy'))

    # specify model
    model_ds_type = ds_type.replace('_', '')
    modelpath = os.path.join(modelpath, f'{model_ds_type}_{sample_len}f.onnx')

    # load the model + start onnx runtime session
    session = onnxruntime.InferenceSession(modelpath)

    # specify input name for inference
    input_name = session.get_inputs()[0].name

    # run inference
    model_outputs = []

    for i in range(data.shape[0]):
        sample = data[i:i+1]
        sample = sample.transpose(0, 2, 1, 3, 4)
        output = session.run(None, {input_name: sample})
        model_outputs.append(output[0])

    # convert model_outputs from logits for each class to prediction
    model_outputs = [np.argmax(model_outputs[i], axis=1) for i in range(data.shape[0])]
    
    # get the true labels and compare them to the outputs
    labels = labels.astype(int).tolist()

    # filter for only correctly classified samples
    return [i for i in range(data.shape[0]) if model_outputs[i] == labels[i]]

def generate_indices(config) -> List[int]:
    # unpack config settings
    sample_gen_type = config.sample_gen_type
    class_size = config.class_size
    ds_type = config.ds_type
    sample_len = config.sample_len

    # randomly generate indices of samples to verify from test set
    if sample_gen_type == 'random':

        # get the indices of all correctly classified samples
        correct_samples = get_correct_samples(ds_type, sample_len, PATH_TO_MODELS, PATH_TO_DATA)

        # partition the correctly classified samples by class
        indices = defaultdict(list, {value: [i for i in correct_samples] for value in range(0, 9)})

        # TODO: unpack this so its just a list and not a list of lists
        return [random.sample(indices[class_label], class_size) for class_label in indices.keys()]

    # inorder generation of indices of samples to verify from test set 
    else:
        pass 

if __name__ == "__main__":
    pass 


















