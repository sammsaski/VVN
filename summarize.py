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
        ds_type='zoom_in',
        sample_len=4,
        attack_type='all_frames',
        ver_algorithm='relax',
        timeout=3600,
        output_dir=output_dir
    )
    
    # =====================================
    # ============ RELAX ==================
    # =====================================

    # run experiment #1 : dataset = zoom in, video length = 4
    # vvn.summarize(output_file_dir=vp.build_output_filepath(config, parent_only=True))

    # run experiment #2 : dataset = zoom out, video length = 4
    # config.ds_type = 'zoom_out'
    # vvn.summarize(output_file_dir=vp.build_output_filepath(config, parent_only=True))
    
    # run experiment #3: dataset = gtsrb, video_length = 4
    config.ds_type = 'gtsrb'
    vvn.summarize(output_file_dir=vp.build_output_filepath(config, parent_only=True), data_len=215)

    # run experiment #4 : dataset = zoom in , video length = 8
    # config.ds_type = 'zoom_in'
    # config.sample_len = 8
    # vvn.summarize(output_file_dir=vp.build_output_filepath(config, parent_only=True))

    # run experiment #5 : dataset = zoom out, video length = 8
    # config.ds_type = 'zoom_out'
    # vvn.summarize(output_file_dir=vp.build_output_filepath(config, parent_only=True))

    # run experiment #6 : dataset = gtsrb, video_length = 4
    config.ds_type = 'gtsrb'
    config.sample_len = 8
    vvn.summarize(output_file_dir=vp.build_output_filepath(config, parent_only=True), data_len=215)    

    # run experiment #7 : dataset = zoom in, video length = 16
    # config.ds_type = 'zoom_in'
    # config.sample_len = 16
    # vvn.summarize(output_file_dir=vp.build_output_filepath(config, parent_only=True))

    # run experiment #8 : dataset = zoom out, video length = 16
    # config.ds_type = 'zoom_out'
    # vvn.summarize(output_file_dir=vp.build_output_filepath(config, parent_only=True))

    # run experiment #9 : dataset = gtsrb, video_length = 4
    config.ds_type = 'gtsrb'
    config.sample_len = 16
    vvn.summarize(output_file_dir=vp.build_output_filepath(config, parent_only=True), data_len=215)


    # =====================================
    # ============ APPROX =================
    # =====================================

    # config = Config(
    #     sample_gen_type='random',
    #     class_size=10,
    #     epsilon=[1/255, 2/255, 3/255],
    #     ds_type='zoom_in',
    #     sample_len=4,
    #     attack_type='all_frames',
    #     ver_algorithm='approx',
    #     timeout=3600,
    #     output_dir=output_dir
    # )

    # # run experiment #1 : dataset = zoom in, video length = 4
    # vvn.summarize(output_file_dir=vp.build_output_filepath(config, parent_only=True))

    # # run experiment #2 : dataset = zoom out, video length = 4
    # config.ds_type = 'zoom_out'
    # vvn.summarize(output_file_dir=vp.build_output_filepath(config, parent_only=True))

    # # run experiment #3 : dataset = zoom in , video length = 8
    # config.ds_type = 'zoom_in'
    # config.sample_len = 8
    # vvn.summarize(output_file_dir=vp.build_output_filepath(config, parent_only=True))

    # # run experiment #4 : dataset = zoom out, video length = 8
    # config.ds_type = 'zoom_out'
    # vvn.summarize(output_file_dir=vp.build_output_filepath(config, parent_only=True))

    # # run experiment #5 : dataset = zoom in, video length = 16
    # config.ds_type = 'zoom_in'
    # config.sample_len = 16
    # vvn.summarize(output_file_dir=vp.build_output_filepath(config, parent_only=True))

    # # run experiment #6 : dataset = zoom out, video length = 16
    # config.ds_type = 'zoom_out'
    # vvn.summarize(output_file_dir=vp.build_output_filepath(config, parent_only=True))
