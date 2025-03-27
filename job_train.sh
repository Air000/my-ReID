#!/bin/bash

#SBATCH --job-name=clip_train
#SBATCH --output=logs/clip_train_%A_%a.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=5G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --tmp=8GB

ml gcc/11.3.0
ml openmpi/4.1.4
ml python/3.10.4

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
source /fred/oz090/shihan/venv_reid/bin/activate

#DATASET_TRAIN=VeRi-776

#cd $JOBFS
#unzip /fred/oz090/shihan/$DATASET_TRAIN.zip -d $JOBFS



cd /fred/oz090/shihan/my-ReID/

srun python train_hf_clip.py


