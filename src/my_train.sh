#!/bin/bash
#PBS -N std

#PBS -l select=1:ncpus=112:ngpus=2:mem=320gb:container_engine=enroot
#PBS -l walltime=24:00:00
#PBS -q normal
#PBS -P 12002486
#PBS -j oe
#PBS -M ke.guo@staff.main.ntu.edu.sg

source "/home/users/ntu/shanhelo/miniconda3/bin/activate"
cd /home/users/ntu/shanhelo/scratch/keguo_projects/sim/src
conda activate catk

#python  run.py > training_map2_clean.log  2>&1
torchrun --nproc_per_node=2  -m run trainer=ddp  > training_map2_clean.log  2>&1

##python -m torch.distributed.run --nproc_per_node=4 --master_port=29502 run.py > pad064_32_noshare.log  2>&1