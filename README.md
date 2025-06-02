
ssh lyuchen@aspire2pntu.nscc.sg
Lyuchen2018!
qsub -I -l select=1:ngpus=1 -l walltime=4:00:00 -P 12002486


source "/home/users/ntu/lyuchen/miniconda3/bin/activate"
cd /home/users/ntu/lyuchen/scratch/keguo_projects/ntu/sim/src
conda activate catk
export PBS_JOBID=47811.pbs111


rsync -avz /home/ke/code/catk/src/waymo_data/full/training_light.zip lyuchen@aspire2pntu.nscc.sg:~/scratch/keguo_projects/ntu/sim/src/waymo_data/full/

rsync -avz ke@10.87.216.98:/home/ke/code/sim/src/waymo_data/full/training_a.zip /home/ke/code/catk/src/waymo_data/full/training_a.zip 

rsync -avz lyuchen@aspire2pntu.nscc.sg:~/scratch/keguo_projects/ntu/sim/logs/js1_pred_valid_token/2025-04-13_10-17-49/checkpoints/epoch_006.ckpt /home/ke/code/catk/src/logs/ 

rsync -avz lyuchen@aspire2pntu.nscc.sg:~/scratch/keguo_projects/ntu/exp/ke/B2d_share_shareagentt_nobuffer_onlyinitego/05.10_00.41/pad/upw5aleq/checkpoints/epoch=15-step=49184.ckpt ./

rsync -avz ke@10.87.216.98:/home/ke/code/sim/src/waymo_data/full/training.zip /home/ke/code/catk/src/waymo_data/full/ 

rsync -avz /home/ke/code/catk/src/waymo_data/full/training_1000_light shanhelo@aspire2pntu.nscc.sg:~/scratch/keguo_projects/sim/src/waymo_data/full/ 

rsync -avz /home/ke/code/catk/src/waymo_data/full/training_inter10_light ke@10.87.216.98:~/code/sim/src/waymo_data/full/

rsync -avz shanhelo@aspire2pntu.nscc.sg:~/scratch/keguo_projects/sim/src/logs/bc_shareaptemp100_pt10_layer24_35_fine/2025-05-23_11-48-49/wosac_submission.tar.gz ./

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

ssh shanhelo@aspire2pntu.nscc.sg
Spyder1@
source "/home/users/ntu/shanhelo/miniconda3/bin/activate"
cd /home/users/ntu/shanhelo/scratch/keguo_projects/sim/src
conda activate catk

conda create -y -n catk python=3.11.9
conda activate catk
conda install -y -c conda-forge ffmpeg=4.3.2
pip install -r install/requirements.txt
pip install torch_geometric
pip install torch_scatter torch_cluster -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
pip install --no-deps waymo-open-dataset-tf-2-12-0==1.6.5


nohup python run.py > 1.txt 2>&1 &
