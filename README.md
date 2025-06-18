
ssh lyuchen@aspire2pntu.nscc.sg
Lyuchen2018!
qsub -I -l select=1:ngpus=1 -l walltime=7:00:00 -P 12002486


export PBS_JOBID=53662.pbs111



rsync -avz /home/ke/code/catk/src/waymo_data/full/training_inter10_a91 shanhelo@aspire2pntu.nscc.sg:~/scratch/keguo_projects/sim/src/waymo_data/full/ 

rsync -avz /home/ke/code/catk/src/waymo_data/full/training_inter10_a91 ke@10.87.216.98:~/code/sim/src/waymo_data/full/

rsync -avz /home/ke/code/catk/src/waymo_data/full/training_inter10_a91 zhangshu@aspire2antu.nscc.sg:~/scratch/sim/src/waymo_data/full/ 

rsync -avz ke@10.87.216.98:~/code/sim/src/waymo_data/full/training_a /home/ke/code/catk/src/waymo_data/full/

rsync -avz shanhelo@aspire2pntu.nscc.sg:~/scratch/keguo_projects/sim/src/logs/bc64_pt20_map/2025-06-12_15-39-26/sim/ka6fqmjc/checkpoints/epoch=19-step=152200-val_closed_wosac=0.7689.ckpt ./

qsub -I -l select=1:ngpus=1 -l walltime=24:00:00 -P personal-ke.guo
myquota -p personal-ke.guo
ssh ke.guo@aspire2antu.nscc.sg
Gk@1402862912

source ~/miniconda3/bin/activate
cd ~/scratch/sim/src
conda activate catk

qsub -I -l select=1:ngpus=1 -l walltime=96:00:00 -P personal-zhangshu

ssh zhangshu@aspire2antu.nscc.sg
Gk@140286

ssh shanhelo@aspire2pntu.nscc.sg
Spyder1@
source "/home/users/ntu/shanhelo/miniconda3/bin/activate"
cd /home/users/ntu/shanhelo/scratch/keguo_projects/sim/src
conda activate catk
export PATH=~/scratch/keguo_projects/cuda/bin:$PATH
export LD_LIBRARY_PATH=~/scratch/keguo_projects/cuda/lib64:$LD_LIBRARY_PATH
export NVIDIA_TF32_OVERRIDE=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_LAUNCH_BLOCKING=1



conda create -y -n catk python=3.11.9
conda activate catk
conda install -y -c conda-forge ffmpeg=4.3.2
pip install -r install/requirements.txt
pip install torch_geometric
pip install torch_scatter torch_cluster -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
pip install --no-deps waymo-open-dataset-tf-2-12-0==1.6.5


nohup python run.py > 1.txt 2>&1 &




pt8 1  1.6 M
Epoch 0:   0%|          | 83/24350 [00:22<1:50:22,  3.66it/s, v_num=fktv]

share 
Epoch 0:   2%|▏         | 584/24350 [02:11<1:29:05,  4.45it/s, v_num=8-02]


share + map encoder

Epoch 0:   2%|▏         | 417/24350 [01:35<1:31:42,  4.35it/s, v_num=7-50]


share + map encoder + a30

Epoch 0:   2%|▏         | 599/24350 [02:41<1:46:45,  3.71it/s, v_num=1-41]


# bc20_pt8_share_map_tv diverge

# bc20_pt8_share_map_reward05x2 diverge

# bc20_pt8_share_map_05mixx2_nodone diverge 

# bc20_pt8_share_map_05mixx2_sa diverge 


#to do : continous action, joint distribution by copula 
