
ssh lyuchen@aspire2pntu.nscc.sg
Lyuchen2025!


export PBS_JOBID=68041.pbs111


rsync -avz /home/ke/code/catk/src/waymo_data/full/training_map2_03 shanhelo@aspire2pntu.nscc.sg:~/scratch/keguo_projects/sim/src/waymo_data/full/  

rsync -avz /home/ke/code/catk/src/waymo_data/full/training_map2_03 ke@10.87.216.98:~/code/sim/src/waymo_data/full/

rsync -avz ke@10.87.216.98:~/code/sim/src/logs/bc32_l3_adamw_histdrop01_a16/2025-08-05_18-24-17/sim/iiikl7cv/checkpoints/epoch=18-step=289161-val_closed_wosac=0.7823.ckpt ./

rsync -avz /home/ke/code/catk/src/waymo_data/full/training_inter10_raw zhangshu@aspire2antu.nscc.sg:~/scratch/sim/src/waymo_data/full/ 

rsync -avz ke@10.87.216.98:~/code/sim/src/logs/bc32_l3_adamw_drop01_a16/2025-08-04_21-06-37/sim/1a7t8j2k/checkpoints/epoch=22-step=350037-val_closed_wosac=0.7807.ckpt ./

rsync -avz shanhelo@aspire2pntu.nscc.sg:~/scratch/keguo_projects/sim/src/logs/AIRL64_l3_noRnorm_value_fineall/2025-08-04_13-33-17/sim/sz8e7agn/checkpoints/epoch=5-step=45660-val_closed_wosac=0.7829.ckpt ./



rsync -avz /home/ke/PAD/exp/navsim_result/pad64_share/05.10_21.45/epoch=17-step=23922.ckpt lyuchen@aspire2pntu.nscc.sg:/home/users/ntu/lyuchen/scratch/keguo_projects/ntu/exp/ke/pad_64_share/05.12_15.32/pad/m6vultai/checkpoints/epoch=17-step=23922.ckpt

rsync -avz lyuchen@aspire2pntu.nscc.sg:~/scratch/keguo_projects/ntu/exp/ke/ke ./


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


wsl -d Ubuntu


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