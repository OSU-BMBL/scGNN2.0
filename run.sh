#!/bin/bash
#SBATCH --job-name=scGNN_v1_2_c
#SBATCH --time=6:00:00
#SBATCH --output="outputs/%j_info_log.txt"
#SBATCH --account=PCON0022
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --mail-type=BEGIN,END,FAIL

set -e

module load python/3.6-conda5.2
source activate scgnnEnv

cd /fs/ess/PCON0022/Edison/scGNN_v1_2_c/outputs
mkdir ${SLURM_JOB_ID}_${dataset_name}_use_bulk
cd ..

python -W ignore scGNN_v2.py \
--given_cell_type_labels \
--use_bulk --load_use_benchmark \
--load_dataset_dir /fs/ess/PCON0022/Edison/datasets \
--load_dataset_name ${dataset_name} \
--output_dir outputs/${SLURM_JOB_ID}_${dataset_name}_use_bulk \
--total_epoch 50

# --feature_AE_epoch 2 2 --graph_AE_epoch 2 --cluster_AE_epoch 2

# sbatch --export=dataset_name=1.Semrau run_no_bulk.sh