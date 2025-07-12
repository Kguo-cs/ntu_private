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

python  run.py > bc64_dist03_lightmap.log  2>&1