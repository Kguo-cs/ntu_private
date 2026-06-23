export PBS_JOBID=118609.pbs111

rsync -avz -e "ssh -p 32884" guoke@sprl-server9.dynip.ntu.edu.sg:~/sim/src/waymo_data/full/validation_map2light  ./


rsync -avz /home/ke/code/sim/src/waymo_data/xflow512_val64_map100_wrap_eval5_epoch=63-valmeta=0.6463.ckpt ke@10.87.114.128:~/keguo/sim/src/waymo_data/ #full/

rsync -avz -e "ssh -p 32884" /home/ke/code/sim/src/waymo_data/xflow512_val64_map100_wrap_eval5_epoch=63-valmeta=0.6463.ckpt guoke@sprl-server9.dynip.ntu.edu.sg:~/sim/src/waymo_data/ #full/

rsync -avz /home/ke/code/sim/src/waymo_data/xflow512_wrap_300_epoch=63-valmeta=0.6458.ckpt zs@10.87.225.106:~/code/sim/src/waymo_data/ #full/

rsync -avz ke@10.87.216.98:~/code/sim/src/logs/gen256_val256_std_w2_drop0_kl02/2026-01-16_16-27-20/bc/7zit9u2h/checkpoints/epoch=63-step=121792.ckpt ./

rsync -avz ke@10.87.114.128:~/keguo/sim/src/logs/model24_t10r10_gp2_logit001_dis5e5_valuedetach/2026-06-18_22-54-59/checkpoints/epoch=4-step=202920-valmeta=0.6520.ckpt ./


rsync -avz ke@10.87.114.128:~/code/sim/src/logs/xflow256_val64/2026-06-19_17-28-16/checkpoints/epoch=57-step=110374-valmeta=0.6447.ckpt ./

rsync -avz -e "ssh -p 32884" /home/ke/miniconda3/envs/sim/lib/python3.11/site-packages/waymo_open_dataset/wdl_limited/sim_agents_metrics/map_metric_features.py  guoke@sprl-server9.dynip.ntu.edu.sg:~/miniconda3/envs/sim/lib/python3.11/site-packages/waymo_open_dataset/wdl_limited/sim_agents_metrics/

rsync -avz  /home/ke/miniconda3/envs/sim/lib/python3.11/site-packages/waymo_open_dataset/wdl_limited/sim_agents_metrics/map_metric_features.py  ke@10.87.114.128:~/miniconda3/envs/sim/lib/python3.11/site-packages/waymo_open_dataset/wdl_limited/sim_agents_metrics/

rsync -avz  /home/ke/miniconda3/envs/sim/lib/python3.11/site-packages/waymo_open_dataset/wdl_limited/sim_agents_metrics/map_metric_features.py  zs@10.87.225.106:~/miniconda3/envs/sim/lib/python3.11/site-packages/waymo_open_dataset/wdl_limited/sim_agents_metrics/

rsync -avz zs@10.87.225.106:~/code/sim/src/logs/xflow256_val64/2026-06-19_17-28-16/checkpoints/epoch=57-step=110374-valmeta=0.6447.ckpt ./


rsync -avz -e "ssh -p 32884" guoke@sprl-server9.dynip.ntu.edu.sg:~/sim/src/logs/xflow512_wrap_300/2026-06-22_14-14-06/bc/zyyema63/checkpoints/epoch=63-valmeta=0.6458.ckpt ./

rsync -avz lyuchen@aspire2pntu.nscc.sg:~/scratch/keguo_projects/sim/src/logs/gen128_gpgenr12_graphdis_shape05/2026-01-08_19-10-39/bc/7jo5ft1x/checkpoints/epoch=15-step=60880.ckpt ./


qsub -I -l select=1:ngpus=1 -l walltime=24:00:00 -P personal-ke.guo
myquota -p personal-ke.guo

Gk@1402862912

source ~/miniconda3/bin/activate
cd ~/scratch/sim/src
conda activate catk

conda create -y -n catk python=3.11.9
conda activate catk
conda install -y -c conda-forge ffmpeg=4.3.2
pip install -r install/requirements.txt
pip install torch_geometric
pip install torch_scatter torch_cluster -f https://data.pyg.org/whl/torch-2.4.0+cu121.html

sudo apt-get install sumo sumo-tools sumo-doc
pip install -r TrafficManager/requirements.txt

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

CUDA_VISIBLE_DEVICES=1 setsid nohup python -m sd.train1 > 1.log 2>&1 &


#0,2,3 1,2,3  -> 0,1, 2


ssh zs@10.87.225.106
source "/home/zs/miniconda3/bin/activate"
cd /home/zs/code/sim/src
conda activate sim
git pull

ssh 10.87.114.128
source "/home/ke/miniconda3/bin/activate"
cd /home/ke/keguo/sim/src
conda activate sim
git pull

nohup python run1.py >  1.log 2>&1 &

nohup python -m sd.train >  1.log 2>&1 &

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

python data_preprocess.py
python run.py


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

sudo bash cuda_12.8.0_570.86.10_linux.run

nano ~/.bashrc


export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH


mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
source ~/miniconda3/bin/activate
conda init --all



conda config --add channels defaults
conda config --set channel_priority flexible
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

sudo apt install git
git clone https://github.com/Kguo-cs/ntu_private.git
mv ntu_private sim
cd sim
git checkout autoenc

conda create -y -n sim python=3.11.9
conda activate sim
conda install -y -c conda-forge ffmpeg=4.3.2
pip install torch_geometric==2.6.1
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
pip install torch_scatter torch_cluster -f  https://data.pyg.org/whl/torch-2.7.0+cu128.html
pip install -r install/requirements.txt
pip install --no-cache-dir --no-deps waymo-open-dataset-tf-2-12-0==1.6.7
pip install shapely==2.1.1


crontab -e
* * * * * /home/ke/wifi_auto_reconnect.sh >> /home/ke/wifi.log 2>&1
sudo visudo
ke ALL=(ALL) NOPASSWD: /usr/bin/nmcli, /usr/bin/systemctl




decomp gail for discriminator
diffusion gail




dit256_cfg3_schedule_nohead

dit256_cfg3_schedule1_70_nohead



design t ,n scedule   previous clean

finetuning the policy encoder



dit128_airl_128pad64_lambda1_map100_t

WO53LWBDQV

121.00 usd
