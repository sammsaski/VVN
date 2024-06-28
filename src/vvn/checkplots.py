# python standard libray
import csv
import io
import os
import random
from typing import Tuple

# third-party packages
import matlab.engine
import numpy as np

# local modules
import vvn.prep as vp
from vvn.config import Config
import vvn.verify as vvn

# define global variables
PARENT_PATH = os.path.dirname(os.getcwd())
NNV_PATH = os.path.join(PARENT_PATH, 'nnv')
NPY_MATLAB_PATH = os.path.join(PARENT_PATH, 'npy-matlab', 'npy-matlab')
GUROBI_PATH = '/Library/gurobi1102/macos_universal2/examples/matlab' # for macos

def check_plots(ds_type, sample_len, attack_type, ver_algorithm, eng, index, eps_index, timeout) -> Tuple[int, int]:
    # check that MATLAB engine was started correctly and is accessible
    if not eng:
        raise Exception('MATLAB Engine was not correctly started and shared. Please make sure to run `prepare_engine`.')

    # call to MATLAB script to run verification + create plot
    future = eng.checkplots(ds_type, sample_len, attack_type, ver_algorithm, index, eps_index, nargout=2, background=True, stdout=io.StringIO())

    try:
        [res, label] = future.result(timeout=float(timeout))
    
    except matlab.engine.TimeoutError:
        print(f'Sample with index {index} timed out.')
        res = -1
        label = -1

    return res, int(label)
    
def run_checkplots(config, indices) -> None:
    # Unpack configuration settings
    epsilon = config.epsilon
    timeout = config.timeout

    ds_type = config.ds_type
    sample_len = config.sample_len
    attack_type = config.attack_type
    ver_algorithm = config.ver_algorithm

    # make sure matlab is started
    eng = vvn.prepare_engine(NNV_PATH, NPY_MATLAB_PATH, GUROBI_PATH)

    # start verification
    for _, index in enumerate(indices):

        # select epsilon
        for eps_index in range(1, len(epsilon) + 1):
            print(f'Generating plot for {index} with config: eps={eps_index}, ver alg={ver_algorithm}, ds type={ds_type}, vid len={sample_len}')

            # run verification + create plot
            res, label = check_plots(ds_type, sample_len, attack_type, ver_algorithm, eng, index, eps_index, timeout)

            print(f'Done creating plot. Verification returned result: {res}. True label: {label}.')

    # close matlab
    eng.quit()
















