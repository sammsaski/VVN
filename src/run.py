import os

import vvn.prep as vp
import vvn.verify as vvn
from vvn.config import Config


if __name__ == "__main__":
    # get the results dir
    root = os.path.dirname(os.getcwd())
    output_dir = os.path.join(root, 'results')

    # define the starting configuration 
    config = Config(
        class_size=10,
        epsilon=[1/255, 2/255, 3/255],
        ds_type='zoom_in',
        sample_len=4,
        ver_algorithm='relax',
        timeout=1800,
        output_dir=output_dir
    )

    # get the samples you wish to verify
    zoom_in_samples, zoom_out_samples = vp.generate_indices(config)

    # =====================================
    # ============ RELAX ==================
    # =====================================

    # run experiment #1 : dataset = zoom in, video length = 4
    vvn.run(config=config, indices=zoom_in_samples)

    # run experiment #2 : dataset = zoom out, video length = 4
    config.ds_type = 'zoom_out'
    vvn.run(config=config, indices=zoom_out_samples)

    # run experiment #3 : dataset = zoom in , video length = 8
    config.ds_type = 'zoom_in'
    config.sample_len = 8
    vvn.run(config=config, indices=zoom_in_samples)

    # run experiment #4 : dataset = zoom out, video length = 8
    config.ds_type = 'zoom_out'
    vvn.run(config=config, indices=zoom_out_samples)

    # run experiment #5 : dataset = zoom in, video length = 16
    config.ds_type = 'zoom_in'
    config.sample_len = 16
    vvn.run(config=config, indices=zoom_in_samples)

    # run experiment #6 : dataset = zoom out, video length = 16
    config.ds_type = 'zoom_out'
    vvn.run(config=config, indices=zoom_out_samples)


    # =====================================
    # ============ APPROX =================
    # =====================================

    config = Config(
        class_size=10,
        epsilon=[1/255, 2/255, 3/255],
        ds_type='zoom_in',
        sample_len=4,
        ver_algorithm='approx',
        timeout=1800,
        output_dir=output_dir
    )

    # run experiment #1 : dataset = zoom in, video length = 4
    vvn.run(config=config, indices=zoom_in_samples)

    # run experiment #2 : dataset = zoom out, video length = 4
    config.ds_type = 'zoom_out'
    vvn.run(config=config, indices=zoom_out_samples)

    # run experiment #3 : dataset = zoom in , video length = 8
    config.ds_type = 'zoom_in'
    config.sample_len = 8
    vvn.run(config=config, indices=zoom_in_samples)

    # run experiment #4 : dataset = zoom out, video length = 8
    config.ds_type = 'zoom_out'
    vvn.run(config=config, indices=zoom_out_samples)

    # run experiment #5 : dataset = zoom in, video length = 16
    config.ds_type = 'zoom_in'
    config.sample_len = 16
    vvn.run(config=config, indices=zoom_in_samples)

    # run experiment #6 : dataset = zoom out, video length = 16
    config.ds_type = 'zoom_out'
    vvn.run(config=config, indices=zoom_out_samples)

