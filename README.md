export PBS_JOBID=118609.pbs111

rsync -avz   ke@10.87.114.128:~/keguo/sim/src/waymo_data/full/training_map2_03_light ./

rsync -avz /home/ke/code/catk/src/waymo_data/full/training_map2_03_light ke@10.87.216.98:~/code/sim/src/waymo_data/full/  

rsync -avz /home/ke/code/sim/src/waymo_data/full/training_map2_init005_light ke@10.87.114.128:~/keguo/sim/src/waymo_data/full/   

rsync -avz ke@10.87.114.128:~/keguo/sim/src/waymo_data/full/training_map2_init005_light ./

rsync -avz -e "ssh -p 32884" /home/ke/code/sim/src/waymo_data/full/training_map2_init005_light guoke@sprl-server9.dynip.ntu.edu.sg:~/sim/src/waymo_data/full/ 

rsync -avz /home/ke/code/sim/src/waymo_data/full/training_map2_init005_light lyuchen@aspire2pntu.nscc.sg:~/scratch/keguo_projects/sim/src/waymo_data/full/ 

rsync -avz ke@10.87.216.98:~/keguo/sim/src/logs/bc40_apt30_tokenexit_14679/2025-11-16_07-22-43/bc/baftp2xu/checkpoints/epoch=62-step=767025.ckpt ./

rsync -avz ke@10.87.114.128:~/keguo/sim/src/logs/bcgen128_sort08_map100_max_01loss_nopos0_nooffset/2026-01-04_23-23-57/bc/hps80kwy/checkpoints/epoch=31-step=121760.ckpt ./

rsync -avz -e "ssh -p 32884" guoke@sprl-server9.dynip.ntu.edu.sg:~/sim/src/logs/bcgen128_randsort04_map100_nooffset_max_01shape0offset/2026-01-04_04-14-39/bc/ry3j2eke/checkpoints/epoch=7-step=30440.ckpt ./



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
ssh lyuchen@aspire2pntu.nscc.sg
automan2018!!


qsub -I -l select=1:ngpus=1 -l walltime=12:00:00 -P 12002486

export CUDA_HOME=/home/users/ntu/lyuchen/scratch/keguo_projects/cuda-12.2
export PATH=/home/users/ntu/lyuchen/scratch/keguo_projects/cuda12.2/bin:$PATH
export LD_LIBRARY_PATH=/home/users/ntu/lyuchen/scratch/keguo_projects/cuda12.2/lib64:$LD_LIBRARY_PATH
cd /home/users/ntu/lyuchen/scratch/keguo_projects/sim/src
conda activate sim
git pull
python run.py


conda create -y -n catk python=3.11.9
conda activate catk
conda install -y -c conda-forge ffmpeg=4.3.2
pip install -r install/requirements.txt
pip install torch_geometric
pip install torch_scatter torch_cluster -f https://data.pyg.org/whl/torch-2.4.0+cu121.html

sudo apt-get install sumo sumo-tools sumo-doc
pip install -r TrafficManager/requirements.txt

nohup python run.py >  bc32_bothnoise_14679_random.log 2>&1 &

pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
pip install torch_scatter torch_cluster -f  https://data.pyg.org/whl/torch-2.7.0+cu128.html
pip install --no-cache-dir --no-deps waymo-open-dataset-tf-2-12-0==1.6.5

wsl -d Ubuntu


ssh guoke@sprl-server9.dynip.ntu.edu.sg -p 32884
140286

export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH
ulimit -n 65535
source "/home/guoke/miniconda3/bin/activate"
cd /home/guoke/sim/src
conda activate sim
git pull
setsid  nohup torchrun --nproc_per_node=4  -m run trainer=ddp  >  1.log 2>&1 &

CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 --master_port=29501  -m run2 trainer=ddp
CUDA_VISIBLE_DEVICES=2,3 setsid nohup torchrun --nproc_per_node=2 --master_port=29503  -m run trainer=ddp >  23.log 2>&1 & 

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29501  -m run1 trainer=ddp >  2.log 2>&1 & 

CUDA_VISIBLE_DEVICES=0 setsid nohup python run.py > 0.log 2>&1 &


#0,2,3 1,2,3  -> 0,1, 2




ssh 10.87.114.128
ulimit -n 65535
source "/home/ke/miniconda3/bin/activate"
cd /home/ke/keguo/sim/src
conda activate sim
git pull

nohup python run.py >  1.log 2>&1 &








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



1. locate the area or agent where causes  the traffic jam 
2. simulate the bird behavior

#only need keep all thus pred agent all valid , allow new entry agent and exit agent .

#do causal intervention, remove the neighbor 


# also allow high order ,weighted 

#autoregressive entry new agent. 



#diffusion initailization 

To alleviate the imbalance of these two control tokens, we set the label weights:w(<KEEP AGENT>) = 0.1 and w(<REMOVE AGENT>) = 0.9when calculating the CrossEntropy Loss


clean predict nosie v loss


#sim 16-> sim 18->gt gen 



For each
agent, it sequentially predicts (a) an agent type token,
(b) a position token, (c) tokens for size and dynamic
attributes, and (d) a sequence of trajectory tokens.
Each prediction is conditioned on the map, ego history,
agent type specification, and all previously generated
agents.


如果你正在用 LS，可以换成 MaxSup 试试。 与其惩罚 Ground-truth 的 logit，不如直接去压那个最大的 logit。



