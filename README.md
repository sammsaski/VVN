# (Artifact) Robustness Verification of Video Classification Neural Networks

This artifact is used to reproduce the results in _Robustness Verification of Video Classification Neural Networks_.

# Requirements

The following resources are required to run this artifact:

- MATLAB 2024a with NNV and npy-matlab installed and added to the path.
- conda environment with Python v3.11. Install rquirements from requirements.txt. Make sure to install the source files.

# Installation

1. Install NNV and npy-matlab and add them to the path.
2. Clone this repository and navigate to the `VVN` directory by running:

   ```bash
   git clone https://github.com/sammsaski/VVN.git && cd VVN
   ```

3. Download the following dataset files from here into the `VVN/data` folder:

-

4. Create a conda environment with Python v3.11 and install the requirements from requirements.txt. Additionally, install the source files to the environment. Both can be done by running the following commands from the root directory (`VVN`):
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

# Smoke Test Instructions

# Reproducing the Results

# Results

# Misc

### Using MATLAB

Navigate to `~/MATLAB/R2023a/bin/` and run `./matlab` to start up MATLAB 2023a.

Don't forget to install the NNV toolbox and the npy-matlab toolbox before running.

### requirements.txt

Numpy could not be upgraded from 1.26.4 to 2.0.0 because of some incompatability with onnxruntime.
