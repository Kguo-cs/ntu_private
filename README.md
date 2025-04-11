
ssh lyuchen@aspire2pntu.nscc.sg
Lyuchen2018!
qsub -I -l select=1:ngpus=1 -l walltime=12:00:00 -P 12002486
ssh ke.guo@aspire2antu.nscc.sg


source "/home/users/ntu/lyuchen/miniconda3/bin/activate"
cd /home/users/ntu/lyuchen/scratch/keguo_projects/ntu/sim
conda activate catk
export PBS_JOBID=37384.pbs111


rsync -avz /home/ke/code/catk/src/waymo_data/full/training_token.zip lyuchen@aspire2pntu.nscc.sg:~/scratch/keguo_projects/ntu/sim/src/waymo_data/full/

rsync -avz /home/ke/code/catk/src/waymo_data/full/training_token.zip ke@10.87.216.98:/home/ke/code/sim/src/waymo_data/full/
