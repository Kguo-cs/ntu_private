#!/bin/bash
#PBS -N std

#PBS -l select=1:ncpus=112:ngpus=1:mem=320gb:container_engine=enroot
#PBS -l walltime=24:00:00
#PBS -q normal
#PBS -P 12002486
#PBS -j oe
#PBS -M ke.guo@staff.main.ntu.edu.sg

source "/home/users/ntu/shanhelo/miniconda3/bin/activate"
cd /home/users/ntu/shanhelo/scratch/keguo_projects/sim/src
conda activate catk

# torchrun --nproc_per_node=8  -m run trainer=ddp  > catk_40.log  2>&1
python  run.py > AIRL80_mean_neighhoodReward2.log  2>&1

##python -m torch.distributed.run --nproc_per_node=4 --master_port=29502 run.py > pad064_32_noshare.log  2>&1