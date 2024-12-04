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

### Docker things

```
docker build --platform linux/amd64 -t vvni -f Dockerfile .

docker image inspect mathworks/matlab-deep-learning:r2024a

docker image inspect --format '{{.Config.User}}' <image_name>
```

Boot up the docker image (from mathworks:r2024a) and figure out what the present working directory is (`pwd`). This should be `/home/matlab/Documents/MATLAB`--even though we've set `USER root`. I'm not sure if this is intended behavior? Should we be ending up in `root/matlab/Documents/MATLAB` as the pwd instead? The `/bin/run.sh` script, which is the decided entrypoint, errors out with `USER root`. This is because, at some point, it tries to `cd /root/Documents/MATLAB`, which fails as the directory doesn't exist (nothing exists in `/root/`).

Try to diagnose these errors. Does it help if I just create a random `/root/Documents/MATLAB` with nothing in it? I can't leave out the `USER root` command because then we don't have root privileges (?) and the next commands fail.

POTENTIAL FIX: copy all of the contents of `/home/matlab/` into `/root/`.
