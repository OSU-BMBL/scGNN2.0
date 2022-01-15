#!/bin/bash
#SBATCH --job-name=scGNN_v1_2_d
#SBATCH --time=2:00
#SBATCH --output="outputs/%j_info_log.txt"
#SBATCH --account=PCON0022
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --mail-type=BEGIN,END,FAIL

set -e

module load python/3.6-conda5.2
source activate scgnnEnv

cd /fs/ess/PCON0022/Edison/scGNN_v1_2_d/outputs
mkdir ${SLURM_JOB_ID}_${dataset_name}_no_bulk_${dropout_prob}_dropout
cd ..

python -W ignore scGNN_v2.py \
--given_cell_type_labels \
--load_use_benchmark \
--load_dataset_dir /fs/ess/PCON0022/Edison/datasets \
--load_dataset_name ${dataset_name} \
--output_run_ID ${SLURM_JOB_ID} \
--output_dir outputs/${SLURM_JOB_ID}_${dataset_name}_no_bulk_${dropout_prob}_dropout \
--dropout_prob ${dropout_prob} \
--total_epoch 2 --feature_AE_epoch 2 2 --graph_AE_epoch 2 --cluster_AE_epoch 2 
# --use_bulk

# sbatch --export=dataset_name=1.Semrau,dropout_prob=0 run.sh
# sbatch --export=dataset_name=2.Chu,dropout_prob=0 run.sh
# sbatch --export=dataset_name=3.Trapnell,dropout_prob=0 run.sh
# sbatch --export=dataset_name=5.Segerstolpe,dropout_prob=0 run.sh
# sbatch --export=dataset_name=9.Chung,dropout_prob=0 run.sh
# sbatch --export=dataset_name=11.Kolodziejczyk,dropout_prob=0 run.sh
# sbatch --export=dataset_name=12.Klein,dropout_prob=0 run.sh
# sbatch --export=dataset_name=13.Zeisel,dropout_prob=0 run.sh
# sbatch --export=dataset_name=1.Semrau,dropout_prob=0.1 run.sh
# sbatch --export=dataset_name=2.Chu,dropout_prob=0.1 run.sh
# sbatch --export=dataset_name=3.Trapnell,dropout_prob=0.1 run.sh
# sbatch --export=dataset_name=5.Segerstolpe,dropout_prob=0.1 run.sh
# sbatch --export=dataset_name=9.Chung,dropout_prob=0.1 run.sh
# sbatch --export=dataset_name=11.Kolodziejczyk,dropout_prob=0.1 run.sh
# sbatch --export=dataset_name=12.Klein,dropout_prob=0.1 run.sh
# sbatch --export=dataset_name=13.Zeisel,dropout_prob=0.1 run.sh
