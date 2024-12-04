#!/bin/bash

# TODO: figure out what the directory this is called from is. reminder that this file itself is called from root.

# MNIST Video
python ../src/run.py

# GTSRB
python ../src/run_gtsrb.py

# STMNIST
python ../src/run_stmnist.py