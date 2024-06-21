import sys
import matlab.engine
import os
import random
import numpy as np
from contextlib import contextmanager
import io

random.seed(42)

def prepare_engine():
    eng = matlab.engine.start_matlab()
    print('started matlab engine!')

    # add nnv path + npy-matlab path
    eng.addpath(os.getcwd())
    eng.addpath(eng.genpath('Users/samuel/milos/rgit/nnv/'))
    eng.addpath(eng.genpath('Users/samuel/milos/rgit/npy-matlab/npy-matlab'))

    return eng 

"""
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 91, 92, 95, 98, 100, 101, 103, 104, 105, 108, 129, 131, 133, 134, 135, 136, 137, 140, 143, 144, 145, 146, 147, 148, 149, 150, 152, 156, 157, 158, 159, 160, 161, 162, 163, 165, 166, 167, 169, 176, 272, 282, 285, 287, 288, 289, 290, 291, 292, 293, 295, 299, 300, 301, 304, 305, 309, 316, 320, 328, 330, 331, 332, 335, 336, 345, 368, 380, 395, 422, 573, 576, 579, 581, 600, 605, 607, 618, 628, 632, 822, 823, 825, 834, 838, 848, 863, 866, 870, 872]    
"""

def generate_inorder_indices():
    return [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 91, 92, 95, 98, 100, 101, 103, 104, 105, 108, 129, 131, 133, 134, 135, 136, 137, 140, 143, 144, 145, 146, 147, 148, 149, 150, 152, 156, 157, 158, 159, 160, 161, 162, 163, 165, 166, 167, 169, 176, 272, 282, 285, 287, 288, 289, 290, 291, 292, 293, 295, 299, 300, 301, 304, 305, 309, 316, 320, 328, 330, 331, 332, 335, 336, 345, 368, 380, 395, 422, 573, 576, 579, 581, 600, 605, 607, 618, 628, 632, 822, 823, 825, 834, 838, 848, 863, 866, 870, 872]    

def verify(
    eng,
    ds_type: str, 
    sample_len: int, 
    attack_type: str, 
    data_index: int, 
    iteration_num: int,
    timeout: int
) -> None:

    future = eng.verify(ds_type, sample_len, attack_type, data_index, nargout=3, background=True)

    try:
        [status, total_time, met] = future.result(timeout=float(timeout))
        print(status, total_time, met)

    except matlab.engine.TimeoutError:
        print("timeout")
        total_time = TimeoutError
        status = 3
        print(status, total_time)

    future.cancel()


if __name__ == "__main__":
    timeout = 3600
    eng = prepare_engine()
    
    indices = generate_inorder_indices()
    print('indices generated')

    # for index, i in enumerate(indices):
    for index in range(67, len(indices)):
        print(f'Iteration {index}')
        i = indices[index]

        for e in range(1, 4):
            future = eng.verify(
                "zoom_out",
                16,
                "all_frames",
                i,
                index+1,
                e,
                nargout=2,
                background=True,
                stdout=io.StringIO()
            )

            try:
                [r, t] = future.result(timeout=float(timeout))

            except matlab.engine.TimeoutError:
                print("there was a timeout")
                t = TimeoutError
                r = 3
                print(f'{r}, {t}')

            future.cancel()

    eng.quit()


    """
    timeout = 3600 # timeout is in seconds
    eng = matlab.engine.start_engine()

    # add nnv path + npy-matlab path
    eng.addpath(os.getcwd())
    eng.addpath(eng.genpath('/home/verivital/nnv'))
    eng.addpath(eng.genpath('/home/verivital/npy-matlab/npy-matlab'))

    indices = generate_inorder_indices()

    for index, i in enumerate(indices):
        print(f'Iteration {index}')

        future = eng.verify(
            ds_type="zoom_out",
            sample_len=16,
            attack_type="all_frames",
            data_index=i,
            output_file=""
        )

        try:
            [status, total_time, met] = future.result(timeout=float(timeout))

        except matlab.engine.TimeoutError:
            print("there was a timeout")
            total_time = TimeoutError
            status = 3

        future.cancel()
    """


        
