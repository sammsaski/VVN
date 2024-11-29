# python stdlib 
import csv
import io
import os
from typing import Tuple

# third-party packages
import matlab.engine
import numpy as np

# local modules
import vvn.prep as vp
from vvn.config import Config
import vvn.gtsrbprep as vgp
import vvn.stmnistprep as vsp

# define global variables
PARENT_PATH = os.path.dirname(os.getcwd())
NNV_PATH = os.path.join(PARENT_PATH, 'nnv')
NPY_MATLAB_PATH = os.path.join(PARENT_PATH, 'npy-matlab', 'npy-matlab')

# TODO: Write docstrings
def prepare_engine(nnv_path, npy_matlab_path):
    if not nnv_path or not npy_matlab_path:
        raise Exception('One of nnv_path or npy_matlab_path is not defined. Please ensure these have been set before running.')

    # start matlab
    eng = matlab.engine.start_matlab()
    print('started matlab engine!')

    # add nnv path and npy-matlab path
    eng.addpath(os.getcwd())
    eng.addpath(eng.genpath(nnv_path))
    eng.addpath(eng.genpath(npy_matlab_path))

    # save reference to it for calling matlab scripts to engine later
    return eng

def verify(ds_type, sample_len, ver_algorithm, eng, index, eps_index, timeout) -> Tuple[int, float | str, str]:
    # check that MATLAB engine was started correctly and is accessible
    if not eng:
        raise Exception('MATLAB Engine was not correctly started and shared. Please make sure to run `prepare_engine`.')

    # call to MATLAB script to run verification
    future = eng.verifyvideo(ds_type, sample_len, ver_algorithm, index, eps_index, nargout=3, background=True, stdout=io.StringIO())

    try:
        [res, t, met] = future.result(timeout=float(timeout))

    except matlab.engine.TimeoutError:
        print('timeout')
        res = 3
        t = 'timeout'
        met = 'timeout'

    future.cancel()

    return res, t, met

def run(config, indices) -> None:
    # Unpack configuration settings;
    epsilon = config.epsilon
    timeout = config.timeout
    
    ds_type = config.ds_type
    sample_len = config.sample_len
    ver_algorithm = config.ver_algorithm

    print(f'Running verification with config: verification algorithm={ver_algorithm}, dataset type={ds_type}, video length={sample_len}') 

    # make sure directory structure + results files are created and correct
    vp.prepare_filetree(config)

    # make sure matlab is started
    eng = prepare_engine(NNV_PATH, NPY_MATLAB_PATH)

    # start verification
    for sample_num, index in enumerate(indices):
        print(f'Iteration {sample_num + 1}')

        # select epsilon
        for eps_index in range(1, len(epsilon) + 1):
            output_file = vp.build_output_filepath(config=config, filename=f'eps={eps_index}_255')

            # verify the sample with a specific epsilon value
            res, t, met = verify(ds_type, sample_len, ver_algorithm, eng, index, eps_index, timeout)

            # write the results
            write_results(output_file, sample_num, res, t, met)

    # print verification summary
    summarize(vp.build_output_filepath(config=config, parent_only=True))

    # close matlab after experiment finishes
    eng.quit()

def verify_gtsrb(sample_len, ver_algorithm, eng, index, eps_index, timeout) -> Tuple[int, float | str, str]:
    # check that MATLAB engine was started correctly and is accessible
    if not eng:
        raise Exception('MATLAB Engine was not correctly started and shared. Please make sure to run `prepare_engine`.')

    # call to MATLAB script to run verification
    future = eng.verifygtsrb(sample_len, ver_algorithm, index, eps_index, nargout=3, background=True, stdout=io.StringIO())

    try:
        [res, t, met] = future.result(timeout=float(timeout))

    except matlab.engine.TimeoutError:
        print('timeout')
        res = 3
        t = 'timeout'
        met = 'timeout'

    future.cancel()

    return res, t, met

def run_gtsrb(config, indices) -> None:
    # Unpack configuration settings;
    epsilon = config.epsilon
    timeout = config.timeout
    
    ds_type = config.ds_type
    sample_len = config.sample_len
    ver_algorithm = config.ver_algorithm

    print(f'Running verification with config: verification algorithm={ver_algorithm}, dataset type={ds_type}, video length={sample_len}') 

    # make sure directory structure + results files are created and correct
    vgp.prepare_filetree(config)

    # make sure matlab is started
    eng = prepare_engine(NNV_PATH, NPY_MATLAB_PATH)

    # start verification
    for sample_num, index in enumerate(indices):
        print(f'Iteration {sample_num + 1}')

        if_timeout = False

        # select epsilon
        for eps_index in range(1, len(epsilon) + 1):
            output_file = vgp.build_output_filepath(config=config, filename=f'eps={eps_index}_255')

            # skip if timeout was met at any point in the previous iterations
            if if_timeout:
                res, t, met = 3, "timeout", "timeout"
                write_results(output_file, sample_num, res, t, met)
                continue

            # verify the sample with a specific epsilon value
            res, t, met = verify_gtsrb(sample_len, ver_algorithm, eng, index, eps_index, timeout)

            if res == 3:
                if_timeout = True

            # write the results
            write_results(output_file, sample_num, res, t, met)

    # print verification summary
    summarize(vgp.build_output_filepath(config=config, parent_only=True), 215)

    # close matlab after experiment finishes
    eng.quit()

def verify_stmnist(sample_len, ver_algorithm, eng, index, eps_index, timeout) -> Tuple[int, float | str, str]:
    # check that MATLAB engine was started correctly and is accessible
    if not eng:
        raise Exception('MATLAB Engine was not correctly started and shared. Please make sure to run `prepare_engine`.')

    # call to MATLAB script to run verification
    future = eng.verifystmnist(sample_len, ver_algorithm, index, eps_index, nargout=3, background=True, stdout=io.StringIO())

    try:
        [res, t, met] = future.result(timeout=float(timeout))

    except matlab.engine.TimeoutError:
        print('timeout')
        res = 3
        t = 'timeout'
        met = 'timeout'

    future.cancel()

    return res, t, met

def run_stmnist(config, indices) -> None:
    # Unpack configuration settings;
    epsilon = config.epsilon
    timeout = config.timeout
    
    ds_type = config.ds_type
    sample_len = config.sample_len
    ver_algorithm = config.ver_algorithm

    print(f'Running verification with config: verification algorithm={ver_algorithm}, dataset type={ds_type}, video length={sample_len}') 

    # make sure directory structure + results files are created and correct
    vsp.prepare_filetree(config)

    # make sure matlab is started
    eng = prepare_engine(NNV_PATH, NPY_MATLAB_PATH)

    # start verification
    for sample_num, index in enumerate(indices):
        print(f'Iteration {sample_num + 1}')

        if_timeout = False

        # select epsilon
        for eps_index in range(1, len(epsilon) + 1):
            output_file = vsp.build_output_filepath(config=config, filename=f'eps={eps_index}_255')

            # skip if timeout was met at any point in the previous iterations
            if if_timeout:
                res, t, met = 3, "timeout", "timeout"
                write_results(output_file, sample_num, res, t, met)
                continue

            # verify the sample with a specific epsilon value
            res, t, met = verify_stmnist(sample_len, ver_algorithm, eng, index, eps_index, timeout)

            if res == 3:
                if_timeout = True

            # write the results
            write_results(output_file, sample_num, res, t, met)

    # print verification summary
    summarize(vsp.build_output_filepath(config=config, parent_only=True), 100)

    # close matlab after experiment finishes
    eng.quit()

# TODO: implement
def run_all() -> None:
    pass

def write_results(output_file, sample_num, res, t, met):
    with open(output_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([sample_num, res, t, met])

def summarize(output_file_dir, data_len):
    print(f'{output_file_dir}')
    for filename in os.listdir(output_file_dir):
        if filename == '.DS_Store':
            continue

        fp = os.path.join(output_file_dir, filename)

        # open the results csv file
        with open(fp, 'r', newline='') as f:
            reader = csv.reader(f, delimiter=',')

            # skip the header
            next(reader)

            res = []
            t = []

            # read the values and build the new arrays for analysis
            for row in reader:
                res.append(row[1])
                t.append(row[2] if not row[2] == 'timeout' else 3600.0)

            # have to convert strings to valid floats before casting to int
            res = np.array(res).astype(float)
            res = np.array(res).astype(int)
            t = np.array(t).astype(float)

        # count the number of verified samples
        total_verified = np.sum(res[res == 1])

        # calculate average time to verify
        average_time = np.mean(t)

        # display the results
        results_header_str = f'Results of verification with {filename.split(".")[0]}'
        total_verified_str = f'Verified {int(total_verified)} robust samples out of {data_len}.'
        average_time_str = f'Average running time was : {average_time}.'
        rowlength = max(len(total_verified_str), len(average_time_str), len(results_header_str))
        print('='*rowlength)
        print(results_header_str)
        print('---')
        print(total_verified_str)
        print('---')
        print(average_time_str)
        print('='*rowlength)
        print('\n\n')


if __name__ == "__main__":
    # example config
    config = Config(
        class_size=10,
        epsilon=[1/255, 2/255, 3/255],
        ds_type='zoom_in',
        sample_len=16,
        ver_algorithm='relax',
        timeout=1800,
        output_dir=''
    )

    # run verification
    run(config=config)

    # update config
    config.ds_type = 'zoom_out'
    run(config=config)




















