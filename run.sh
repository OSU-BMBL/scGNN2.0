#!/bin/bash
#SBATCH --job-name=scGNN_vf.2.0
#SBATCH --output="outputs/%j_info_log.txt"
#SBATCH --account=PCON0022
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --mail-type=BEGIN,END,FAIL

set -e

module load python/3.6-conda5.2
source activate py-scgnn

cd /fs/ess/PCON0022/Edison/scGNN2.0/outputs
mkdir ${SLURM_JOB_ID}_${dataset_name}_${dropout_prob}_dropout
cd ..

python -W ignore scGNN_v2.py \
--given_cell_type_labels \
--load_use_benchmark \
--load_dataset_dir /fs/ess/PCON0022/Edison/datasets \
--load_dataset_name ${dataset_name} \
--output_run_ID ${SLURM_JOB_ID} \
--output_dir outputs/${SLURM_JOB_ID}_${dataset_name}_${dropout_prob}_dropout \
--dropout_prob ${dropout_prob} \
--total_epoch 15 --feature_AE_epoch 500 600 \
--clustering_embed both --output_intermediate \
# --graph_AE_concat_prev_embed --graph_AE_normalize_embed # --clustering_use_flexible_k
