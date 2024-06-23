import os
import vvn.prep as vp
import vvn.verify as vvn
from vvn.config import Config


if __name__ == "__main__":
    # get the results dir
    output_dir = os.path.join(os.getcwd(), 'results')

    config = Config(
        sample_gen_type='random',
        class_size=10,
        epsilon=[1/255, 2/255, 3/255],
        ds_type='zoom_in',
        sample_len=4,
        attack_type='all_frames',
        timeout=3600,
        output_dir=output_dir
    )

    # run experiment #1 : dataset = zoom in, video length = 4
    vvn.run(config=config)

    # run experiment #2 : dataset = zoom out, video length = 4
    config.ds_type = 'zoom_out'
    vvn.run(config=config)

    # run experiment #3 : dataset = zoom in , video length = 8
    config.ds_type = 'zoom_in'
    config.sample_len = 8
    vvn.run(config=config)

    # run experiment #4 : dataset = zoom out, video length = 8
    config.ds_type = 'zoom_out'
    vvn.run(config=config)

    # run experiment #5 : dataset = zoom in, video length = 16
    config.ds_type = 'zoom_in'
    config.sample_len = 16
    vvn.run(config=config)

    # run experiment #6 : dataset = zoom out, video length = 16
    config.ds_type = 'zoom_out'
    vvn.run(config=config)
