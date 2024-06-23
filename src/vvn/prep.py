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
from vvnconfig import Config

# define global variables
PATH_TO_MODELS = ''
PATH_TO_TEST_DATA = ''
PATH_TO_TEST_LABELS = ''

# set seed
random.seed(42)

def prepare_filetree(config: Config):
    # TODO: come up with a more flexible way to do this
    # create all directories/files for each type of experiment being run
    for sgt in ['random', 'inorder']:
        for dst in ['zoom_in', 'zoom_out']:
            for length in ['4', '8', '16']:
                for eps_filename in [f'eps={e}_255' for e in range(1, 4)]:
                    fp = build_output_filepath(config, eps_filename)

                    # create the parent directories if they don't already exist
                    os.makedirs(os.path.join(output_dir, sgt, dst, length), exist_ok=True)

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
    dst = str_config.ds_type
    length = str_config.sample_len

    fp = os.path.join(output_dir, sgt, dst, length)

    return fp if parent_only else os.path.join(fp, filename) + '.csv'

# TODO: need to run inference on whole test set to make sure
#       we are verifying samples that are originally correctly classified
def get_correct_samples(modelpath, datapath, labelpath) -> List[int]:
    # load the data + labels
    data = np.load(datapath)
    labels = np.load(labelpath)

    # load the model + start onnx runtime session
    session = onnxruntime.InferenceSession(modelpath)

    # run inference
    outputs = session.run(None, {input_name: data})
    
    # return the correctly classified samples

def generate_indices(sample_gen_type, labels, class_size) -> List[int]:
    # randomly generate indices of samples to verify from test set
    if sample_gen_type == 'random':
        indices = defaultdict(list, {value: [i for i, v in enumerate(labels) if v == value] for value in set(labels)})

        # TODO: unpack this so its just a list and not a list of lists
        return [random.sample(indices[class_label], class_size) for class_label in indices.keys()]

    # inorder generation of indices of samples to verify from test set 
    else:
        pass 
