#!/bin/bash
#PBS -N std

#PBS -l select=1:ncpus=112:ngpus=4:mem=320gb:container_engine=enroot
#PBS -l walltime=16:00:00
#PBS -q normal
#PBS -P 12002486
#PBS -j oe
#PBS -M ke.guo@staff.main.ntu.edu.sg

source "/home/users/ntu/shanhelo/miniconda3/bin/activate"
cd /home/users/ntu/shanhelo/scratch/keguo_projects/sim/src
conda activate catk

python -m torch.distributed.run --nproc_per_node=4 --master_port=29502 run.py > cat.log  2>&1