
ssh lyuchen@aspire2pntu.nscc.sg
Lyuchen2018!
qsub -I -l select=1:ngpus=1 -l walltime=10:00:00 -P 12002486


source "/home/users/ntu/lyuchen/miniconda3/bin/activate"
cd /home/users/ntu/lyuchen/scratch/keguo_projects/ntu/sim/src
conda activate catk
export PBS_JOBID=10298079.pbs101


rsync -avz /home/ke/code/catk/src/waymo_data/full/training_inter10.zip lyuchen@aspire2pntu.nscc.sg:~/scratch/keguo_projects/ntu/sim/src/waymo_data/full/

rsync -avz ke@10.87.216.98:/home/ke/code/sim/src/waymo_data/full/training_a.zip /home/ke/code/catk/src/waymo_data/full/training_a.zip 

rsync -avz lyuchen@aspire2pntu.nscc.sg:~/scratch/keguo_projects/ntu/sim/logs/js1_pred_valid_token/2025-04-13_10-17-49/checkpoints/epoch_006.ckpt /home/ke/code/catk/src/logs/ 

rsync -avz lyuchen@aspire2pntu.nscc.sg:~/scratch/keguo_projects/ntu/exp/ke/B2d_064_noshare/05.11_18.22/pad/vvk987iv/checkpoints/epoch=5-step=18444.ckpt ./

rsync -avz ke@10.87.216.98:/home/ke/code/sim/src/waymo_data/full/training.zip /home/ke/code/catk/src/waymo_data/full/ 

rsync -avz /home/ke/code/catk/src/waymo_data/full/validation.zip ke@10.87.216.98:/home/ke/code/sim/src/waymo_data/full/

rsync -avz /home/ke/code/catk/src/waymo_data/full/validation.zip zhangshu@aspire2antu.nscc.sg:~/scratch/sim/src/waymo_data/full/


qsub -I -l select=1:ngpus=1 -l walltime=24:00:00 -P personal-ke.guo
myquota -p personal-ke.guo
ssh ke.guo@aspire2antu.nscc.sg

source ~/miniconda3/bin/activate
cd ~/scratch/sim/src
conda activate catk
Gk@1402862912

qsub -I -l select=1:ngpus=1 -l walltime=96:00:00 -P personal-zhangshu

ssh zhangshu@aspire2antu.nscc.sg

Gk@140286


conda create -y -n catk python=3.11.9
conda activate catk
conda install -y -c conda-forge ffmpeg=4.3.2
pip install -r install/requirements.txt
pip install torch_geometric
pip install torch_scatter torch_cluster -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
pip install --no-deps waymo-open-dataset-tf-2-12-0==1.6.5
