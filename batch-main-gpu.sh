#!/bin/sh

#SBATCH --job-name=train_model
#SBATCH --partition=gpu
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=4GB
#SBATCH --account=Education-EEMCS-MSc-DSAIG

module load 2023r1
module load openmpi
module load python
module load miniconda3
module load cuda/11.6

# Set conda env:
unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"

# Activate conda, run job, deactivate conda
conda activate /home/ppostelnicu/.conda/envs/env
srun python main.py train_model --config 'configs/config.ini' > train_model.log
conda deactivate
