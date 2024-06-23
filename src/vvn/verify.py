# python standard libray
import csv
import os
from typing import Tuple

# third-party packages
import matlab.engine
import numpy as np

# local modules
import vvn.prep as vp
from vvn.config import Config

# define global variables
PARENT_PATH = os.path.dirname(os.getcwd())
NNV_PATH = os.path.join(PARENT_PATH, 'nnv')
NPY_MATLAB_PATH = os.path.join(PARENT_PATH, 'npy-matlab', 'npy-matlab')

# TODO: Write docstrings
def prepare_engine(nnv_path, npy_matlab_path) -> matlab.engine.Engine:
    if not nnv_path or not npy_matlab_path:
        raise Exception('One of nnv_path or npy_matlab_path is not defined. Please ensure these have been set before running.')

    # start matlab
    eng = matlab.engine.start_matlab()
    print('started matlab engine!')

    # add nnv path + npy-matlab path
    eng.addpath(os.getcwd())
    eng.addpath(eng.genpath(nnv_path))
    eng.addpath(eng.genpath(npy_matlab_path))

    # save reference to it for calling matlab scripts to engine later
    return eng

def verify(ds_type, sample_len, attack_type, eng, index, eps_index) -> Tuple[int, float | str, str]:
    # check that MATLAB engine was started correctly and is accessible
    if not eng:
        raise Exception('MATLAB Engine was not correctly started and shared. Please make sure to run `prepare_engine`.')

    # call to MATLAB script to run verification
    future = eng.verify(ds_type, sample_len, attack_type, index, eps_index, nargout=3, background=True)

    try:
        [res, t, met] = future.result(timeout=float(timeout))

    except matlab.engine.TimeoutError:
        print('timeout')
        res = 3
        t = 'timeout'
        met = 'timeout'

    future.cancel()

    return res, t, met

def run(config) -> None:
    # Unpack configuration settings;
    locals().update(config)

    print(f'Running verification with config: dataset type={ds_type}, video length={sample_len}') 

    # make sure directory structure + results files are created and correct
    vp.prepare_filetree(config)

    # make sure matlab is started
    eng = prepare_engine(NNV_PATH, NPY_MATLAB_PATH)

    # generate indices
    indices = vp.generate_indices(config)

    # start verification
    for sample_num, index in enumerate(indices):
        print(f'Iteration {sample_num + 1}')

        # select epsilon
        for eps_index in range(1, len(epsilon) + 1):
            # get epsilon for output file naming
            # have to subtract 1 because we had to add one for matlab 1-indexing
            eps = epsilon[eps_index - 1] 

            # build the output file
            # for naming convention, we will use the
            # epsilon value for filename
            output_file = vp.build_output_filepath(config=config, filename=str(eps))

            # verify the sample with a specific epsilon value
            res, t, met = verify(ds_type, sample_len, attack_type, eng, index, eps_index)

            # write the results
            write_results(output_file, sample_num, res, t, met)

    # print verification summary
    summarize(vp.build_output_filepath(config=config, parent_only=True))

    # close matlab after experiment finishes
    eng.quit()

# TODO: implement
def run_all() -> None:
    pass

def write_results(output_file, sample_num, res, t, met):
    with open(output_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([sample_num, res, t, met])

def summarize(output_file_dir):
    for filename in os.listdir(output_file_dir):
        fp = os.path.join(output_file_dir, filename)

        # open the results csv file
        data = np.genfromtxt(
            fp, 
            delimiter=',', 
            skip_header=1, 
            dtype=None, 
            encoding=None,
            converters={1: lambda s: 3600 if s == 'timeout' else float(s)}
        )
        
        # count the number of verified samples
        total_verified = np.sum(data[:, 0] == 1)

        # calculate average time to verify
        average_time = np.mean(data[:, 1])

        # display the results
        results_header_str = f'Results of verification with {filename.split('.')[0]}'
        total_verified_str = f'Verified {total_verified} robust samples out of {100}.')
        average_time_str = f'Average running time was : {average_time}.')
        rowlength = max(len(total_verified_str), len(average_time_str), len(results_header_str))
        print('-'*rowlength)
        print(results_header_str)
        print('---')
        print(total_verified_str)
        print('---')
        print(average_time_str)
        print('-'*rowlength)


if __name__ == "__main__":
    # example config
    config = Config(
        sample_gen_type='random',
        class_size=10,
        epsilon=[1/255, 2/255, 3/255],
        ds_type='zoom_in',
        sample_len=16,
        attack_type='all_frames',
        timeout=3600,
        output_dir=''
    )

    # run verification
    run(config=config)

    # update config
    config.ds_type = 'zoom_out'
    run(config=config)




















