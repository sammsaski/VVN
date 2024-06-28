import os

import vvn.prep as vp
import vvn.verify as vvn
from vvn.config import Config


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
    vvn.run(config=config, indices=zoom_out_samples)

    
    # =====================================
    # ============ APPROX =================
    # =====================================

    # update config for approx method
    config.ver_algorithm = 'approx'

    # run experiment #1 : dataset = zoom out, video length = 8
    vvn.run(config=config, indices=zoom_out_samples)
