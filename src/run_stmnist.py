import os

import vvn.stmnistprep as vsp
import vvn.verify as vvn
from vvn.config import Config


if __name__ == "__main__":
    # get the results dir
    root = os.path.dirname(os.getcwd())
    output_dir = os.path.join(root, 'results')

    # define the starting configuration 
    config = Config(
        sample_gen_type='random',
        class_size=10,
        epsilon=[1/255, 2/255, 3/255],
        ds_type='stmnist',
        sample_len=16,
        attack_type='all_frames',
        ver_algorithm='relax',
        timeout=1800,
        output_dir=output_dir
    )

    # get the samples you wish to verify
    samples = vsp.generate_indices(config)

    # =====================================
    # ============ RELAX ==================
    # =====================================

    # run experiment #1 : dataset = stmnist, video length = 16
    vvn.run_stmnist(config=config, indices=samples)

    # run experiment #2 : dataset = stmnist, video length = 32
    config.sample_len = 32
    vvn.run_stmnist(config=config, indices=samples)

    # run experiment #3 : dataset = stmnist, video length = 64
    config.sample_len = 64
    vvn.run_stmnist(config=config, indices=samples)

    # =====================================
    # ============ APPROX =================
    # =====================================

    config = Config(
        sample_gen_type='random',
        class_size=10,
        epsilon=[1/255, 2/255, 3/255],
        ds_type='stmnist',
        sample_len=16,
        attack_type='all_frames',
        ver_algorithm='approx',
        timeout=1800,
        output_dir=output_dir
    )

    # run experiment #1 : dataset = stmnist, video length = 16
    vvn.run_stmnist(config=config, indices=samples)

    # run experiment #2 : dataset = stmnist, video length = 32
    config.sample_len = 32
    vvn.run_stmnist(config=config, indices=samples)

    # run experiment #3 : dataset = stmnist, video length = 64
    config.sample_len = 64
    vvn.run_stmnist(config=config, indices=samples)