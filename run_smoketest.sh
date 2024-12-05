#!/bin/bash

# make sure image/container not built or in-use already
docker rm vvnc
docker rmi vvni

# unzip datasets


#
# Matlab docker for VVN
#
NNV_IMAGE_ID="vvni"
NNV_CONTAINER_ID="vvnc"
docker build -t $NNV_IMAGE_ID -f Dockerfile . # optionally add --platform linux/amd64 if running on apple silicon (?)
# docker run -it -v ${PWD}:/home/user/vvn --name $NNV_CONTAINER_ID $NNV_IMAGE_ID
docker run -dt -v ${PWD}:/home/user/vvn/ --name $NNV_CONTAINER_ID $NNV_IMAGE_ID

echo -e "Starting container...\n"

# wait until started
until [ "`docker inspect -f {{.State.Running}} vvnc`"=="true" ]; do
    sleep 0.1;
done;

echo -e "Container started...\n"


#
# Install NNV + npy-matlab
#
# echo -e "Installing NNV + npy-matlab...\n"
# docker exec ${NNV_CONTAINER_ID} matlab -batch "run('/home/user/vvn/scripts/install_tools.m')"

echo -e "Starting the smoke test."

# Run the smoketest
# docker exec ${NNV_CONTAINER_ID} cd /home/user/vvn/ && scripts/run_vvn_smoketest.sh

 docker exec -it ${NNV_CONTAINER_ID} bash -c "cd /home/user/vvn/ && chmod +x scripts/run_sample.sh && scripts/run_sample.sh"

#
# All tests passed
#
echo -e "All tests passed."

# clean up
# docker rm vvnc
# docker rmi vvni
