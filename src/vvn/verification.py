import sys
import matlab.engine
import time
import os
import random
import numpy as np

random.seed(42)

def prepare_engine():
    eng = matlab.engine.start_matlab()

    # add nnv path + npy-matlab path
    eng.addpath(os.getcwd())
    eng.addpath(eng.genpath('/home/verivital/nnv'))
    eng.addpath(eng.genpath('/home/verivital/npy-matlab/npy-matlab'))

    return eng

def generate_random_indices(labels, subset_size):
    indices = defaultdict(list, {value: [i for i, v in enumerate(labels) if v == value] for value in set(labels)})

    return [random.sample(indices[class_label], subset_size) for class_label in indices.keys()]

"""
FINISH THIS LATER; NOT NEEDED

def generate_inorder_indices(labels, subset_size):
    c = {class_label: 0 for class_label in range(10)}
    
    i = 0
    while any([class_label != subset_size for class_label in c.keys()]):
        label = labels[i]

        if c[label] == subset_size:
            i += 1
            continue
"""

def randomize(func):
    def wrap(*args, **kwargs):
        eng = prepare_engine()

        status = 2 # initialize with an 'Unknown' status

        # counter for number of instances verified by class
        c = {i: 0 for i in range(10)} 

        while any([i != 10 for i in c.keys()]):
            sample_idx = random.randint(0, 1000)

            sample = data[sample_idx]
            label = labels[sample_idx]

            # skip the sample if we've already verified 10 of its class
            if c[label] >= 10:
                continue

            res, time, met = func(*args, **kwargs, sample=sample)

            # write the results

        # close matlab
        eng.quit()

    return wrap

def inorder(func):
    def wrap(*args, **kwargs):
        eng = prepare_engine()

        status = 2 # Initialize with an 'Unknown' status

        # counter for number of instances verified by class
        c = {class_label: 0 for class_label in range(10)}

        i = 0
        while any([class_label != 10 for class_label in c.keys()]):
            sample = data[i]
            label = labels[i]

            # skip the sample iff we've already verified 10 of its class 
            if c[label] >= 10:
                continue

            res, time, met = func(eng, *args, **kwargs, sample=sample)

            # write the results

        # close matlab
        eng.quit()
    
    return wrap

def verify(
    eng,
    ds_type: str, 
    sample_len: int, 
    attack_type: str, 
    data_index: int, 
    output_file: str,
    timeout: int
) -> None:

    future = eng.verify(ds_type, sample_len, attack_type, data_index, output_file, nargout=3, background=True)

    try:
        [status, total_time, met] = future.result(timeout=float(timeout))

    except matlab.engine.TimeoutError:
        print("timeout")
        total_time = TimeoutError
        status = 3

    future.cancel()

if __name__ == "__main__":
    # run randomized verification
    randomize(verify)

    # run inorder verification
    inorder(verify)
