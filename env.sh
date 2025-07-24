#!/bin/bash 


export MUJOCO_PY_MUJOCO_PATH=/workspace/mujoco210
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/workspace/mujoco210/bin

export CPATH=$CONDA_PREFIX/include
export PYTHONPATH=$PWD


# cd $CONDA_PREFIX/bin
# ln -s x86_64-conda_cos7-linux-gnu-gcc gcc
# source anaconda3/etc/profile.d/conda.sh
# nohup python generate_dataset/dqnAgent.py -config config/dqn_dataset.yml  > output.log 2>&1 &