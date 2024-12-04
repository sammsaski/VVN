#!/bin/bash

#
# Matlab docker for VVN
#
NNV_IMAGE_ID="vvni"
NNV_CONTAINER_ID="vvnc"
docker build -t $NNV_IMAGE_ID -f Dockerfile .
docker run -it -v ${PWD}:/home/user/vvn --name $NNV_CONTAINER_ID $NNV_IMAGE_ID

