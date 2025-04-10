
lyuchen@aspire2pntu.nscc.sg
Lyuchen2018!
qsub -I -l select=1:ngpus=1 -l walltime=12:00:00 -P 12002486
ssh ke.guo@aspire2antu.nscc.sg


source "/home/users/ntu/lyuchen/miniconda3/bin/activate"
cd /home/users/ntu/lyuchen/scratch/keguo_projects/ntu/sim
conda activate catk
export PBS_JOBID=37291.pbs111
