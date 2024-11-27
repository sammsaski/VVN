# standard lib
import os

# local package
import vvn.prep as vp
import vvn.verify as vvn
from vvn.config import Config
import vvn.checkplots


if __name__ == "__main__":
    # get the figures dir
    root = os.path.dirname(os.getcwd())
    output_dir = os.path.join(root, 'figs')

    # define the starting configuration 
    config = Config(
        sample_gen_type='random',
        class_size=10,
        epsilon=[1/255, 2/255, 3/255],
        ds_type='zoom_in',
        sample_len=4,
        attack_type='all_frames',
        ver_algorithm='relax',
        timeout=1800,
        output_dir=output_dir
    )

    # get the original samples we verified
    zoom_in_samples, zoom_out_samples = vp.generate_indices(config)
 
    # get the samples we want
    zoom_in_ind= [10, 52, 54, 69, 83, 97]
    zoom_out_ind = [12, 27, 59, 66, 43, 52, 70, 91]

    # =====================================
    # ============ RELAX ==================
    # =====================================

    # run experiment #1 : dataset = zoom in, video length = 4
    vvn.checkplots.run_checkplots(config=config, indices=[zoom_in_samples[i] for i in zoom_in_ind])

    # run experiment #2 : dataset = zoom out, video length = 4
    config.ds_type = 'zoom_out'
    vvn.checkplots.run_checkplots(config=config, indices=[zoom_out_samples[i] for i in zoom_out_ind])

    # run experiment #3 : dataset = zoom in , video length = 8
    config.ds_type = 'zoom_in'
    config.sample_len = 8
    vvn.checkplots.run_checkplots(config=config, indices=[zoom_in_samples[i] for i in zoom_in_ind])

    # run experiment #4 : dataset = zoom out, video length = 8
    config.ds_type = 'zoom_out'
    vvn.checkplots.run_checkplots(config=config, indices=[zoom_out_samples[i] for i in zoom_out_ind])

    # run experiment #5 : dataset = zoom in, video length = 16
    config.ds_type = 'zoom_in'
    config.sample_len = 16
    vvn.checkplots.run_checkplots(config=config, indices=[zoom_in_samples[i] for i in zoom_in_ind])

    # run experiment #6 : dataset = zoom out, video length = 16
    config.ds_type = 'zoom_out'
    vvn.checkplots.run_checkplots(config=config, indices=[zoom_out_samples[i] for i in zoom_out_ind])


    # =====================================
    # ============ APPROX =================
    # =====================================
    
    # get the samples we want
    zoom_in_ind= [17, 24, 39, 86]
    zoom_out_ind = [27, 32, 58, 74]

    config = Config(
        sample_gen_type='random',
        class_size=10,
        epsilon=[1/255, 2/255, 3/255],
        ds_type='zoom_in',
        sample_len=4,
        attack_type='all_frames',
        ver_algorithm='approx',
        timeout=1800,
        output_dir=output_dir
    )

    # run experiment #1 : dataset = zoom in, video length = 4
    vvn.checkplots.run_checkplots(config=config, indices=[zoom_in_samples[i] for i in zoom_in_ind])

    # run experiment #2 : dataset = zoom out, video length = 4
    config.ds_type = 'zoom_out'
    vvn.checkplots.run_checkplots(config=config, indices=[zoom_out_samples[i] for i in zoom_out_ind])

    # run experiment #3 : dataset = zoom in , video length = 8
    config.ds_type = 'zoom_in'
    config.sample_len = 8
    vvn.checkplots.run_checkplots(config=config, indices=[zoom_in_samples[i] for i in zoom_in_ind])

    # run experiment #4 : dataset = zoom out, video length = 8
    config.ds_type = 'zoom_out'
    vvn.checkplots.run_checkplots(config=config, indices=[zoom_out_samples[i] for i in zoom_out_ind])


