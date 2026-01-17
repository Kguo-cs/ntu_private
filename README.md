export PBS_JOBID=118609.pbs111

rsync -avz   ke@10.87.114.128:~/keguo/sim/srcc/waymo_data/full/training_map2_03_light ./

rsync -avz /home/ke/code/catk/src/waymo_data/full/training_map2_03_light ke@10.87.216.98:~/code/sim/src/waymo_data/full/  

rsync -avz /home/ke/code/sim/src/waymo_data/gen256_val256_std_w2_no80_kl03_epoch=31-step=60896.ckpt ke@10.87.114.128:~/keguo/sim/src/waymo_data/   

rsync -avz ke@10.87.114.128:~/wifi_auto_reconnect.sh ./

rsync -avz -e "ssh -p 32884" /home/ke/code/sim/src/waymo_data/gen256_val256_std_w2_no80_kl03_epoch=31-step=60896.ckpt guoke@sprl-server9.dynip.ntu.edu.sg:~/sim/src/waymo_data/

rsync -avz /home/ke/code/sim/src/waymo_data/full/training_map2_init005_light lyuchen@aspire2pntu.nscc.sg:~/scratch/keguo_projects/sim/src/waymo_data/full/ 

rsync -avz ke@10.87.216.98:~/code/sim/src/logs/gen256_val256_std_w2_drop0_kl02/2026-01-16_16-27-20/bc/7zit9u2h/checkpoints/epoch=31-step=60896.ckpt ./

rsync -avz ke@10.87.114.128:~/keguo/sim/src/logs/gen256_val256_std_w2_no80_kl03/2026-01-16_14-04-22/bc/dsbh2sgr/checkpoints/epoch=31-step=60896.ckpt ./

rsync -avz -e "ssh -p 32884" guoke@sprl-server9.dynip.ntu.edu.sg:~/sim/src/logs/gen256_val512_auto_std_nopos_train0/2026-01-14_15-16-48/bc/f92s2zj0/checkpoints/epoch=31-step=60896.ckpt ./

rsync -avz lyuchen@aspire2pntu.nscc.sg:~/scratch/keguo_projects/sim/src/logs/gen128_gpgenr12_graphdis_shape05/2026-01-08_19-10-39/bc/7jo5ft1x/checkpoints/epoch=15-step=60880.ckpt ./


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
pip install -r install/requirements.txt

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

CUDA_VISIBLE_DEVICES=2 setsid nohup python run2.py > 2.log 2>&1 &


#0,2,3 1,2,3  -> 0,1, 2


ssh zs@10.87.225.106
source "/home/zs/miniconda3/bin/activate"
cd /home/zs/code/sim/src
conda activate sim
git pull

ssh ke@10.87.216.98
source "/home/ke/miniconda3/bin/activate"
cd /home/ke/code/sim/src
conda activate sim
git pull


ssh 10.87.114.128
ulimit -n 65535
source "/home/ke/miniconda3/bin/activate"
cd /home/ke/keguo/sim/src
conda activate sim
git pull

nohup python run.py >  1.log 2>&1 &


#to do : continous action, joint distribution by copula , continuous map point,  semi-gradient, IV-learn, interval 0.1 ,, 2048 unknown token


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



add spectral norm and softplus activation loss



sudo apt install nvidia-driver-580-open

wget https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda_12.8.0_570.86.10_linux.run

mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
source ~/miniconda3/bin/activate
conda init --all
sudo sh cuda_12.8.0_570.86.10_linux.run

git clone https://github.com/Kguo-cs/ntu_private.git


conda create -y -n sim python=3.11.9
conda activate sim
conda install -y -c conda-forge ffmpeg=4.3.2
pip install torch_geometric==2.6.1
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
pip install torch_scatter torch_cluster -f  https://data.pyg.org/whl/torch-2.7.0+cu128.html
pip install -r install/requirements.txt
pip install --no-cache-dir --no-deps waymo-open-dataset-tf-2-12-0==1.6.5
pip install shapely==2.1.1