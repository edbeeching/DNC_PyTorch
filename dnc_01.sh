#!/bin/bash
LOGF="logs/logfile_jobid_$SLURM_JOBID"_"pid_$$.txt"
echo "All output goes into the logfile: $LOGF"

echo "Running on $HOSTNAME"

source /home/ebeeching/venvs/rl_env41/bin/activate

export "PATH=/storage/lib/cuda-8.0/bin:$PATH"
export "LD_LIBRARY_PATH=/storage/lib/cudnn-6.0-for-cuda-8.0/lib64:/storage/lib/cuda-8.0/lib64:$LD_LIBRARY_PATH"


echo "Running script ${0##*/}"
echo "Parameters are $PARAMS"

python3 /home/ebeeching/saved_dir/work/DNC_PyTorch/train.py > $LOGF 2>&1

echo "Script terminated."
