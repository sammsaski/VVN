from enum import verify
import os

import vvn.prep as vp
import vvn.verify as vvn
from vvn.config import Config

# define global variables
PARENT_PATH = os.path.dirname(os.getcwd())
NNV_PATH = os.path.join(PARENT_PATH, 'nnv')
NPY_MATLAB_PATH = os.path.join(PARENT_PATH, 'npy-matlab', 'npy-matlab')
GUROBI_PATH = '/Library/gurobi1102/macos_universal2/examples/matlab' # for macos


if __name__ == "__main__":
    # get the results dir
    output_dir = os.path.join(os.getcwd(), 'results')

    # define the starting configuration 
    config = Config(
        sample_gen_type='random',
        class_size=10,
        epsilon=[1/255, 2/255, 3/255],
        ds_type='zoom_out',
        sample_len=8,
        attack_type='all_frames',
        ver_algorithm='relax',
        timeout=3600,
        output_dir=output_dir
    )

    # get the samples you wish to verify
    zoom_in_samples, zoom_out_samples = vp.generate_indices(config)

    # =====================================
    # ============ RELAX ==================
    # =====================================

    # run experiment #1 : dataset = zoom out, video length = 8
    eng = vvn.prepare_engine(NNV_PATH, NPY_MATLAB_PATH, GUROBI_PATH)

    ds_type = config.ds_type
    sample_len = config.sample_len
    attack_type = config.attack_type
    ver_algorithm = config.ver_algorithm
    timeout = config.timeout
    
    res, t, met = vvn.verify(ds_type, sample_len, attack_type, ver_algorithm, eng, zoom_out_samples[0], 1, timeout)
    print(res, t, met)

    # now let's see how things work for zoom in
    ds_type = 'zoom_in'
    res, t, met = vvn.verify(ds_type, sample_len, attack_type, ver_algorithm, eng, zoom_in_samples[0], 1, timeout)
    print(res, t, met)
     
    ds_type = 'zoom_out'
    sample_len = 16
    res, t, met = vvn.verify(ds_type, sample_len, attack_type, ver_algorithm, eng, zoom_out_samples[0], 1, timeout)
    print(res, t, met)

    eng.quit()

    
    # =====================================
    # ============ APPROX =================
    # =====================================

    # update config for approx method
    config.ver_algorithm = 'approx'

    # run experiment #1 : dataset = zoom out, video length = 8
    # vvn.run(config=config, indices=zoom_out_samples)
