#!/bin/bash
#PBS -N std

#PBS -l select=1:ncpus=112:ngpus=1:mem=320gb:container_engine=enroot
#PBS -l walltime=24:00:00
#PBS -q normal
#PBS -P 12002486
#PBS -j oe
#PBS -M ke.guo@staff.main.ntu.edu.sg
#PBS -l container_name=nvidia+pytorch+23.10-py3

export CUDA_HOME=/home/users/ntu/lyuchen/scratch/keguo_projects/cuda-12.2
export PATH=/home/users/ntu/lyuchen/scratch/keguo_projects/cuda12.2/bin:$PATH
export LD_LIBRARY_PATH=/home/users/ntu/lyuchen/scratch/keguo_projects/cuda12.2/lib64:$LD_LIBRARY_PATH
source "/home/users/ntu/lyuchen/miniconda3/bin/activate"
cd /home/users/ntu/lyuchen/scratch/keguo_projects/sim/src
conda activate sim

#torchrun --nproc_per_node=2  -m run trainer=ddp  > AIRL160_kl3_learnmap.log  2>&1
python  run.py > AIRL80_noedge_1m_Prevroformer_allvalid_notoken_gp.log  2>&1

##python -m torch.distributed.run --nproc_per_node=4 --master_port=29502 run.py > pad064_32_noshare.log  2>&1