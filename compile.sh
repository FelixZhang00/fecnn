#!/bin/bash
# change you cuda path
export PATH=$PATH:/usr/local/cuda/bin

if uname | grep -q Darwin; then
  CUDA_LIB_DIR=/Developer/NVIDIA/CUDA-7.5/lib
elif uname | grep -q Linux; then
  CUDA_LIB_DIR=/usr/local/cuda/lib64
fi

nvcc -std=c++11 -O3 -o fecnn src/fecnn.cu -I/usr/local/cuda/include -L$CUDA_LIB_DIR -lcublas
