
ssh lyuchen@aspire2pntu.nscc.sg
Lyuchen2025!


export PBS_JOBID=83255.pbs111


rsync -avz /home/ke/code/catk/src/waymo_data/full/nuplan_cross2_route shanhelo@aspire2pntu.nscc.sg:~/scratch/keguo_projects/sim/src/waymo_data/full/  

rsync -avz /home/ke/code/catk/src/waymo_data/full/validation shanhelo@aspire2pntu.nscc.sg:~/scratch/keguo_projects/sim/src/waymo_data/full/  


rsync -avz ~/code/catk/src/waymo_data/full/validation_id1  ke@10.87.114.128:~/keguo/sim/src/waymo_data/full/ 

rsync -avz ~/code/catk/src/waymo_data/full/validation_id1 ke@10.87.216.98:~/code/sim/src/waymo_data/full/


rsync -avz -e "ssh -p 32884" ~/code/catk/src/waymo_data/full/validation_id1 guoke@sprl-server9.dynip.ntu.edu.sg:~/sim/src/waymo_data/full/

rsync -avz /home/ke/code/catk/src/waymo_data/full/training_inter10_raw zhangshu@aspire2antu.nscc.sg:~/scratch/sim/src/waymo_data/full/ 

rsync -avz /home/ke/code/catk/src/waymo_data/full/nuplan_cross2 shanhelo@aspire2pntu.nscc.sg:~/scratch/keguo_projects/sim/src/waymo_data/full/

rsync -avz ke@10.87.216.98:/home/ke/code/sim/src/logs/AIRL152_bc4_learnmap/2025-09-24_14-57-54/wosac_submission.tar.gz ./

rsync -avz ke@10.87.114.128:~/keguo/sim/src/logs/bc32_agent467_inter2_drop01_03/2025-10-13_22-47-35/bc/d34cjk5o/checkpoints/epoch=19-step=233320.ckpt ./



rsync -avz shanhelo@aspire2pntu.nscc.sg:~/scratch/keguo_projects/sim/src/logs/airl56_nolcf/2025-10-15_10-08-28/sim/k9p5rb2d/checkpoints/epoch=2-step=20000.ckpt ./

rsync -avz -e "ssh -p 32884" guoke@sprl-server9.dynip.ntu.edu.sg:~/sim/src/logs/AIRL160_learnmap_arn10_kl4/2025-10-11_14-10-16/sim/7a8i19q8/checkpoints/epoch=15-step=48704.ckpt ./




qsub -I -l select=1:ngpus=1 -l walltime=24:00:00 -P personal-ke.guo
myquota -p personal-ke.guo
ssh ke.guo@aspire2antu.nscc.sg
Gk@1402862912

source ~/miniconda3/bin/activate
cd ~/scratch/sim/src
conda activate catk

qsub -I -l select=1:ngpus=1 -l walltime=24:00:00 -P personal-zhangshu

ssh zhangshu@aspire2antu.nscc.sg
Gk@140286

ssh shanhelo@aspire2pntu.nscc.sg
Spyder1@

qsub -I -l select=1:ngpus=1 -l walltime=12:00:00 -P 12002486

export PATH=~/scratch/keguo_projects/cuda/bin:$PATH
source "/home/users/ntu/shanhelo/miniconda3/bin/activate"
cd /home/users/ntu/shanhelo/scratch/keguo_projects/sim/src
conda activate catk
git pull
python run.py

export LD_LIBRARY_PATH=~/scratch/keguo_projects/cuda/lib64:$LD_LIBRARY_PATH
export NVIDIA_TF32_OVERRIDE=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_LAUNCH_BLOCKING=1



conda create -y -n catk python=3.11.9
conda activate catk
conda install -y -c conda-forge ffmpeg=4.3.2
pip install -r install/requirements.txt
pip install torch_geometric
pip install torch_scatter torch_cluster -f https://data.pyg.org/whl/torch-2.4.0+cu121.html

sudo apt-get install sumo sumo-tools sumo-doc
pip install -r TrafficManager/requirements.txt

nohup python run.py >  1.log 2>&1 &

pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
pip install torch_scatter torch_cluster -f  https://data.pyg.org/whl/torch-2.7.0+cu128.html
pip install --no-cache-dir --no-deps waymo-open-dataset-tf-2-12-0==1.6.5

wsl -d Ubuntu


ssh guoke@sprl-server9.dynip.ntu.edu.sg -p 32884
140286

export PATH=/home/guoke/cuda/bin:$PATH
export LD_LIBRARY_PATH=/home/guoke/cuda/lib64:$LD_LIBRARY_PATH
ulimit -n 65535
source "/home/guoke/miniconda3/bin/activate"
cd /home/guoke/sim/src
conda activate catk
git pull
setsid  nohup torchrun --nproc_per_node=4  -m run trainer=ddp  >  1.log 2>&1 &

CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 --master_port=29501  -m run trainer=ddp
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29502  -m run trainer=ddp


CUDA_VISIBLE_DEVICES=0  python run.py >  0.log 2>&1 & 







ssh 10.87.114.128
ulimit -n 65535
source "/home/ke/miniconda3/bin/activate"
cd /home/ke/keguo/sim/src
conda activate sim
git pull

nohup python start.py >  1.log 2>&1 &








137977




# bc20_pt8_share_map_tv diverge

# bc20_pt8_share_map_reward05x2 diverge

# bc20_pt8_share_map_05mixx2_nodone diverge 

# bc20_pt8_share_map_05mixx2_sa diverge 


#to do : continous action, joint distribution by copula , continuous map point,  semi-gradient, IV-learn, interval 0.1 ,, 2048 unknown token

#pos, heading quantize 

#road edge inter 1 
#no speed bump

#use continous  
#more bin 
#no bc 
#AdamW 
#continuous map point

#discriminator sa 


#a parameter to disappear  other cannot see it ,it can see other
#discount 0.97 − 0.99
#t is best to use as large as possible replay buffers for sampling negative examples

#less agent:  bad 
#token : train
#temporal full discrimiantor
#noise to input to discriminator
#c-gail: discriminator constrain to 1/2

#not done

#dropout 0.5 input and 0.75 hidden 
#airl , other reward learning method
#other tokenize
#goal prediction and conditioned


#bc+ finetune 

#bc+iq learn
#finetune less learning rate 

# value network to reject sampling
#counterfact 
#less pt2pt
#non-edge to edge 

#running mean and std of advanatage 

# KL+REVERSE kl

#post sampling to solve goal-conditioned


#aril logpi shape

https://github.com/seolhokim/InverseRL-Pytorch/tree/main
# VAIL 

all2_clean

value use other action

centric discriminator: AIRL64_value0001_disexpertvalidcentric


AIRL64_value0001noclip_distr402060a5_expertvalid influence of range 


#to do 

#progressive diffusion discretiezed
