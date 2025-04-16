
ssh lyuchen@aspire2pntu.nscc.sg
Lyuchen2018!
qsub -I -l select=1:ngpus=1 -l walltime=24:00:00 -P 12002486


source "/home/users/ntu/lyuchen/miniconda3/bin/activate"
cd /home/users/ntu/lyuchen/scratch/keguo_projects/ntu/sim
conda activate catk
export PBS_JOBID=38599.pbs111


rsync -avz /home/ke/code/catk/src/waymo_data/full/validation_nuplan.zip lyuchen@aspire2pntu.nscc.sg:~/scratch/keguo_projects/ntu/sim/src/waymo_data/full/

rsync -avz /home/ke/code/catk/src/waymo_data/full/training_token.zip ke@10.87.216.98:/home/ke/code/sim/src/waymo_data/full/

rsync -avz lyuchen@aspire2pntu.nscc.sg:~/scratch/keguo_projects/ntu/sim/logs/js1_pred_valid_token/2025-04-13_10-17-49/checkpoints/epoch_006.ckpt /home/ke/code/catk/src/logs/ 

rsync -avz lyuchen@aspire2pntu.nscc.sg:~/scratch/keguo_projects/ntu/sim/src/logs/lr1e3_Kl1e-3_reward1_min1e3_soft0005/2025-04-15_19-13-00/sim_agent/4nwyyl2i/checkpoints/epoch=7-step=94947-val_closed_wosac=0.7517.ckpt ./

rsync -avz ke@10.87.216.98:/home/ke/code/sim/src/waymo_data/full/training.zip /home/ke/code/catk/src/waymo_data/full/ 

rsync -avz /home/ke/code/catk/src/waymo_data/full/training_token.zip ke.guo@aspire2antu.nscc.sg:~/scratch/sim/src/waymo_data/full/

rsync -avz /home/ke/code/catk/src/waymo_data/full/validation.zip ke.guo@aspire2antu.nscc.sg:~/scratch/sim/src/waymo_data/full/

qsub -I -l select=1:ngpus=1 -l walltime=24:00:00 -P personal-ke.guo
myquota -p personal-ke.guo
ssh ke.guo@aspire2antu.nscc.sg

source ~/miniconda3/bin/activate
cd ~/scratch/sim/src
conda activate catk
Gk@1402862912