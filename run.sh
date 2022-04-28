#!/bin/bash
#SBATCH --job-name=scGNN_v1.3.0
#SBATCH --output="outputs/%j_info_log.txt"
#SBATCH --account=PCON0022
#SBATCH --nodes=1
#SBATCH --mem=16G
#SBATCH --mail-type=BEGIN,END,FAIL

set -e

module load python/3.6-conda5.2
source activate py-scgnn

cd /fs/ess/PCON0022/edison/scGNN2.0/outputs
mkdir ${SLURM_JOB_ID}_${dataset_name}_${dropout_prob}_dropout
cd ..

python -W ignore scGNN_v2.py --use_bulk \
--given_cell_type_labels \
--load_use_benchmark \
--load_dataset_dir /fs/ess/PCON0022/edison/datasets \
--load_dataset_name ${dataset_name} \
--output_run_ID ${SLURM_JOB_ID} \
--output_dir outputs/${SLURM_JOB_ID}_${dataset_name}_${dropout_prob}_dropout \
--dropout_prob ${dropout_prob} \
--total_epoch 31 --feature_AE_epoch 500 300 \
--output_intermediate \
--graph_AE_neighborhood_factor 0.05 --graph_AE_graph_construction v2 \
--graph_AE_normalize_embed sum1 --graph_AE_use_GAT

# --graph_AE_retain_weights #SBATCH --gpus-per-node=1
# --graph_AE_concat_prev_embed --graph_AE_normalize_embed sum1 binary # --clustering_use_flexible_k --clustering_embed both 
