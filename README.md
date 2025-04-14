
ssh lyuchen@aspire2pntu.nscc.sg
Lyuchen2018!
qsub -I -l select=1:ngpus=1 -l walltime=12:00:00 -P 12002486


source "/home/users/ntu/lyuchen/miniconda3/bin/activate"
cd /home/users/ntu/lyuchen/scratch/keguo_projects/ntu/sim
conda activate catk
export PBS_JOBID=9975092.pbs101 


rsync -avz /home/ke/code/catk/src/waymo_data/full/training_token1.zip lyuchen@aspire2pntu.nscc.sg:~/scratch/keguo_projects/ntu/sim/waymo_data/full/

rsync -avz /home/ke/code/catk/src/waymo_data/full/training_token.zip ke@10.87.216.98:/home/ke/code/sim/src/waymo_data/full/

rsync -avz lyuchen@aspire2pntu.nscc.sg:~/scratch/keguo_projects/ntu/sim/logs/js1_pred_valid_token/2025-04-13_10-17-49/checkpoints/epoch_006.ckpt /home/ke/code/catk/src/logs/ 

rsync -avz lyuchen@aspire2pntu.nscc.sg:~/scratch/keguo_projects/ntu/sim/logs/BC/2025-04-13_21-17-43/sim_agent/l6ps9mup/checkpoints/epoch=10-step=133900-val_closed_wosac=0.7516.ckpt ./

rsync -avz ke@10.87.216.98:/home/ke/code/sim/src/waymo_data/full/training.zip /home/ke/code/catk/src/waymo_data/full/ 

rsync -avz /home/ke/code/catk/src/waymo_data/full/training_token.zip ke.guo@aspire2antu.nscc.sg:~/scratch/sim/src/waymo_data/full/

rsync -avz /home/ke/code/catk/src/waymo_data/full/validation.zip ke.guo@aspire2antu.nscc.sg:~/scratch/sim/src/waymo_data/full/

qsub -I -l select=1:ngpus=1 -l walltime=24:00:00 -P personal-ke.guo

myprojects -p personal-ke.guo -l -s 2024-12-01 -e 2025-01-25            |
myquota -p personal-ke.guo
ssh ke.guo@aspire2antu.nscc.sg

source ~/miniconda3/bin/activate
cd ~/scratch/sim/src
conda activate catk
Gk@1402862912