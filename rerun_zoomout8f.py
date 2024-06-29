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
    zoom_out_samples = [624, 882, 278, 180, 540, 439, 306, 757, 821, 654, 248, 817, 368, 949, 963, 59, 260, 34, 357, 465, 304, 69, 238, 666, 867, 356, 239, 776, 585, 460, 760, 536, 158, 301, 154, 280, 908, 659, 632, 297, 910, 687, 499, 686, 463, 418, 248, 152, 596, 578, 96, 922, 50, 117, 169, 738, 176, 989, 809, 491, 702, 67, 445, 441, 547, 616, 285, 649, 12, 809, 872, 126, 812, 630, 916, 303, 952, 758, 390, 120, 332, 507, 174, 529, 4, 873, 868, 297, 586, 933, 196, 594, 112, 736, 337, 755, 719, 223, 169, 433]

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
