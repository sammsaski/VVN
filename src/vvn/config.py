import copy
from dataclasses import dataclass, fields
from typing import List, Literal, Self


@dataclass
class Config:
    # Experimental settings
    sample_gen_type: Literal['random', 'inorder'] # TODO: This feature can be removed.
    class_size: int # number of samples to verify per class
    epsilon: List # [1/255, 2/255, 3/255] 
    timeout: int # 1800 (s)
    output_dir: str # /path/to/VVN/results 

    # Verification settings
    ds_type: Literal['zoom_in', 'zoom_out', 'gtsrb', 'stmnist'] 
    sample_len: Literal[4, 8, 16, 32, 64] # length of videos in number of frames
    attack_type: Literal['single_frame', 'all_frames'] # whether we attack all frames or a subset TODO: Like above comment, we can delete this
    ver_algorithm: Literal['relax', 'approx'] # types of verification algorithms to use

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                raise AttributeError(f'{k} is not a valid field of {self.__class__.__name__}.')

    def tostr(self) -> Self:
        # make a deepcopy of the config so that we don't
        # accidentally override the original config values
        config = copy.deepcopy(self)

        for f in fields(config):
            value = getattr(config, f.name)

            # handle case where value is list
            if isinstance(value, list):
                setattr(config, f.name, list(map(str, value)))
            
            # should be safe to handle everything else
            elif not isinstance(value, str):
                setattr(config, f.name, str(value))

        return config
