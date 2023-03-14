#!/bin/bash

#SBATCH -A cs601_gpu
#SBATCH --partition=mig_class
#SBATCH --reservation=MIG
#SBATCH --qos=qos_mig_class
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --job-name="HW6 CS 601.471/671 homework"
#SBATCH --mem-per-cpu=6G   # memory per cpu-core

module load anaconda

# init virtual environment if needed
conda create -n new_envi python=3.8

# conda activate toy_classification_env # open the Python environment
pip install -r requirements.txt # install Python dependencies

# q5
srun python -u classification.py --experiment "overfit" --small_subset True --device cuda --model "distilbert-base-uncased" --batch_size "64" --lr 1e-4 --num_epochs 30 > test.out

#q6
srun python -u classification.py --experiment "overfit" --small_subset False --device cuda --model "distilbert-base-uncased" --batch_size "64" --lr 1e-4 --num_epochs 30 > test.out

#q7
srun python -u classification.py --experiment "overfit" --small_subset False --device cuda --model "distilbert-base-uncased" --batch_size "64" --hyperparam True > test.out

#q8
srun python -u classification.py --experiment "overfit" --small_subset False --device cuda --model "BERT-base-uncased" --batch_size "64" --hyperparam True > test.out
