#
# Robustness Verification of Video Classification Neural Networks
#

FROM mathworks/matlab-deep-learning:r2024a

# Set the non-root user information
USER root

# Install mpm dependencies.
RUN export DEBIAN_FRONTEND=noninteractive \
    && apt-get update \
    && apt-get -y install git python3 pip \
    && apt-get install --no-install-recommends --yes \
    wget \
    unzip \
    tar \
    ca-certificates \
    && apt-get clean \
    && apt-get autoremove \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m pip install --upgrade pip \
    && pip install numpy scikit-learn

# ---- ---- Required nnv packages ---- --- 
# Computer Vision ------------------> already installed
# Control Systems
# Deep Learning --------------------> already installed
# Image Processing -----------------> already installed
# Optimization
# Parallel Computing ---------------> already installed
# Statistics and Machine Learning --> already installed
# Symbolic Math
# System Identification
# --- ---- ----- ----- ----- ---- ---- ---- 
RUN wget -q https://www.mathworks.com/mpm/glnxa64/mpm && \
	chmod +x mpm && \
	./mpm install --destination=/opt/matlab/R2024a/ --release=R2024a \
	--products Control_System_Toolbox Optimization_Toolbox \
    Symbolic_Math_Toolbox System_Identification_Toolbox \
	|| (echo "MPM Installation Failure. See below for more information:" && cat /tmp/mathworks_root.log && false) && \
	rm -f mpm /tmp/mathworks_root.log && \
	ln -fs /opt/matlab/R2024a/bin/matlab /usr/local/bin/matlab

# clone nnv repository and install
WORKDIR /home/user
RUN git clone --recursive https://github.com/verivital/nnv.git /home/user/nnv
RUN chmod +x install_ubuntu.sh
RUN ./install_ubuntu.sh

# clone npy-matlaba and add to MATLAB path
RUN git clone https://github.com/kwikteam/npy-matlab.git /home/user/npy-matlab
RUN matlab -nodisplay -r "cd ${CURR_DIR}; cd npy-matlab; addpath; savepath; exit()"

# Copy files and directory structure to working directory
COPY . /home/user/vvn

# CMD ["matlab", "-batch", "run('/home/user/vvn/scripts/run')"]