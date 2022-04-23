#!/bin/bash
#SBATCH --job-name=scGNN_vf.2.5
#SBATCH --output="log/%j_info_log.txt"
#SBATCH --time=100:00:00
#SBATCH --account=PCON0022
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --mail-type=BEGIN,END,FAIL

set -x

cd /fs/ess/scratch/PCON0022/ch/scGNN2.0/gat_outputs
if  [ -d  ${SLURM_JOB_ID}_${dataset_name}_${dropout_prob}_dropout ]; then
echo ${SLURM_JOB_ID}_${dataset_name}_${dropout_prob}_dropout
echo "文件夹存在！"
rm  -rf  mkdir ${SLURM_JOB_ID}_${dataset_name}_${dropout_prob}_dropout
fi

mkdir ${SLURM_JOB_ID}_${dataset_name}_${dropout_prob}_dropout


~/.conda/envs/ch/bin/python -W ignore /users/PCON0022/haocheng/scGNN/scGNN2.0/scGNN_v2.py \
--given_cell_type_labels \
--load_use_benchmark \
--load_dataset_dir /fs/ess/PCON0022/edison/datasets \
--load_dataset_name ${dataset_name} \
--output_run_ID ${SLURM_JOB_ID} \
--output_dir ${SLURM_JOB_ID}_${dataset_name}_${dropout_prob}_dropout \
--dropout_prob ${dropout_prob} \
--total_epoch 31 --feature_AE_epoch 500 300 \
--output_intermediate \
--graph_AE_neighborhood_factor 0.1 --graph_AE_graph_construction v2 \
--graph_AE_normalize_embed sum1 --graph_AE_use_GAT

# --graph_AE_retain_weights
# --graph_AE_concat_prev_embed --graph_AE_normalize_embed sum1 binary # --clustering_use_flexible_k --clustering_embed both 
