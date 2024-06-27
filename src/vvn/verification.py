# python standard library
import csv
import os
import random
from typing import Dict, List, Tuple, Any, Literal 
from dataclasses import dataclass
from collections import defaultdict

# third-party packages
import matlab.engine
import numpy as np

# set seed
random.seed(42)

def prepare_engine():
    eng = matlab.engine.start_matlab()
    print('started matlab engine!')

    # add nnv path + npy-matlab path
    eng.addpath(os.getcwd())
    eng.addpath(eng.genpath('Users/samuel/milos/rgit/nnv'))
    eng.addpath(eng.genpath('Users/samuel/milos/rgit/npy-matlab/npy-matlab'))

    return eng

def prepare_filetree(currdir, sample_gen_type, ds_type, sample_len, epsilon):
    # convert all args to string
    sample_gen_type = list(map(str, sample_gen_type))
    ds_type = list(map(str, ds_type))
    sample_len = list(map(str, sample_len))
    epsilon = list(map(str, epsilon))

    # create all directories/files for each type of experiment being run
    for sgt in sample_gen_type:
        for dst in ds_type:
            for length in sample_len:
                for eps in epsilon:
                    fp = os.path.join(currdir, sgt, dst, length, eps) + '.csv'
                    if not os.path.isdir(fp):
                        with open('fp', 'a', newline='') as f:
                            # write CSV headers
                            writer = csv.writer(f)
                            writer.writerow(['Sample Number', 'Result', 'Time', 'Method'])

def generate_random_indices(labels, class_size):
    indices = defaultdict(list, {value: [i for i, v in enumerate(labels) if v == value] for value in set(labels)})

    return [random.sample(indices[class_label], class_size) for class_label in indices.keys()]

def generate_indices(sample_gen_type, labels, class_size):
    if sample_gen_type == "random":
        return generate_random_indices(labels, class_size)
    else:
        return generate_inorder_indices(labels, class_size)

def generate_inorder_indices(labels, subset_size):
    pass
    
@dataclass
class Config:
    # Experiment settings
    sample_gen_type: Literal['random', 'inorder'] # "random" or "inorder"
    class_size: int # 10 samples per class
    output_dir: str # /path/to/VVN/results
    timeout: int # 3600 (s)
    epsilon: List # [1/255, 2/255, 3/255]
    labels: List # list of labels from the dataset

    # Verification settings
    ds_type: Literal['zoom_in', 'zoom_out'] # might need to change this to ZoomIn/ZoomOut
    sample_len: Literal[4, 8, 16] # length of the videos in number of frames
    attack_type: Literal['single_frame', 'all_frames'] # whether we attack all frames or not
    data_index: int # how to access the sample we want to verify from the full dataset
    iteration_num: int # the number of sample we're verifying in the current experiment

    def epsilon_len(self) -> int:
        return len(self.epsilon)

    def get_verification_configs(self) -> Dict:
        return {
            "ds_type": self.ds_type,
            "sample_len": self.sample_len,
            "attack_type": self.attack_type,
            "timeout": self.timeout
        }

def write_results(output_file, res, t, met):
    with open(output_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([res, t, met])

def execute(func):
    """
    A wrapper to pass our defined configuration file to the verification
    script.
    """
    def wrap(*args, **kwargs):
        # start matlab
        eng = prepare_engine()

        # save matlab as kwarg to pass to function call
        kwargs['eng'] = eng

        # unpack config
        try:
            sample_gen_type = config.sample_gen_type
            labels = config.labels
            class_size = config.class_size
            output_dir = config.output_dir
        except:
            eng.quit()
            raise Exception("config not provided.")

        # generate the subset of samples to verify
        indices = generate_indices(sample_gen_type, labels, class_size)

        # start verification
        for sample_num, index in enumerate(indices):
            print(f'Iteration {sample_num + 1}')

            # select epsilon
            for eps_index in range(1, len(num_epsilon)+1):
                
                # verify the sample with a specific epsilon value
                res, t, met = func(*args, **kwargs)

                # check if directory + file exists, then save results
                prepare_output_files(output_dir, output_file)
                write_results(output_file, sample_num + 1, res, t, met)
                
        # close matlab
        eng.quit()

    return wrap
          
@execute
def verify(
    config,
    eng=None
) -> Tuple[Any, Any, Any]:

    # check that MATLAB engine was started correctly and is accessible.
    if not eng:
        raise Exception('MATLAB Engine was not correctly started and shared.')

    # unpack configs
    locals().update(config)

    # create output_file
    output_file = os.path.join(output_dir, sample_gen_type, ds_type, sample_len, eps) + '.csv'

    future = eng.verify(ds_type, sample_len, attack_type, data_index, output_file, nargout=3, background=True)

    try:
        [status, total_time, met] = future.result(timeout=float(timeout)) 

    except matlab.engine.TimeoutError:
        print("timeout")
        status = 3
        total_time = "--" 
        met = "timeout"

    future.cancel()

    return status, total_time, met

if __name__ == "__main__":
    # load the labels for defining the subset of samples to verify
    labels = np.load('')
    labels = labels.astype(int).tolist()

    # define the config
    config = Config(
        sample_gen_type="",
        class_size=0,
        epsilon=[],
        labels=[],
        ds_type="",
        sample_len=0,
        attack_type="",
        data_index=0,
        iteration_num=0, # might not need this
        timeout=0,
        output_dir="",
        output_file=""
    )
    
    # run verification
    verify(config) 
