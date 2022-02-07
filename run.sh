#!/bin/bash
#SBATCH --job-name=scGNN_v1.2.e.6.longer
#SBATCH --time=5:00:00
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
--total_epoch 31 --feature_AE_epoch 500 600
# --feature_AE_epoch 2 2 --graph_AE_epoch 2 --cluster_AE_epoch 2 --clustering_louvain_only

# sbatch --export=dataset_name=1.Semrau,dropout_prob=0.1 run.sh
# sbatch --export=dataset_name=2.Chu,dropout_prob=0.1 run.sh
# sbatch --export=dataset_name=3.Trapnell,dropout_prob=0.1 run.sh
# sbatch --export=dataset_name=5.Segerstolpe,dropout_prob=0.1 run.sh
# sbatch --export=dataset_name=9.Chung,dropout_prob=0.1 run.sh
# sbatch --export=dataset_name=11.Kolodziejczyk,dropout_prob=0.1 run.sh
# sbatch --export=dataset_name=12.Klein,dropout_prob=0.1 run.sh
# sbatch --export=dataset_name=13.Zeisel,dropout_prob=0.1 run.sh
# sbatch --export=dataset_name=1.Semrau,dropout_prob=0 run.sh
# sbatch --export=dataset_name=2.Chu,dropout_prob=0 run.sh
# sbatch --export=dataset_name=3.Trapnell,dropout_prob=0 run.sh
# sbatch --export=dataset_name=5.Segerstolpe,dropout_prob=0 run.sh
# sbatch --export=dataset_name=9.Chung,dropout_prob=0 run.sh
# sbatch --export=dataset_name=11.Kolodziejczyk,dropout_prob=0 run.sh
# sbatch --export=dataset_name=12.Klein,dropout_prob=0 run.sh
# sbatch --export=dataset_name=13.Zeisel,dropout_prob=0 run.sh

# sbatch --export=dataset_name=Biase,dropout_prob=0.1 run.sh
# sbatch --export=dataset_name=Deng,dropout_prob=0.1 run.sh
# sbatch --export=dataset_name=Goolam,dropout_prob=0.1 run.sh
# sbatch --export=dataset_name=Guo,dropout_prob=0.1 run.sh
# sbatch --export=dataset_name=Pollen,dropout_prob=0.1 run.sh
# sbatch --export=dataset_name=Treutlein,dropout_prob=0.1 run.sh
# sbatch --export=dataset_name=Usoskin,dropout_prob=0.1 run.sh
# sbatch --export=dataset_name=Yan,dropout_prob=0.1 run.sh
# sbatch --export=dataset_name=Biase,dropout_prob=0 run.sh
# sbatch --export=dataset_name=Deng,dropout_prob=0 run.sh
# sbatch --export=dataset_name=Goolam,dropout_prob=0 run.sh
# sbatch --export=dataset_name=Guo,dropout_prob=0 run.sh
# sbatch --export=dataset_name=Pollen,dropout_prob=0 run.sh
# sbatch --export=dataset_name=Treutlein,dropout_prob=0 run.sh
# sbatch --export=dataset_name=Usoskin,dropout_prob=0 run.sh
# sbatch --export=dataset_name=Yan,dropout_prob=0 run.sh


# --use_bulk
###SBATCH --gpus-per-node=1
# python -W ignore scGNN_v2.py \
# --given_cell_type_labels \
# --load_dataset_dir /fs/ess/PCON0022/Edison/datasets/raw \
# --load_dataset_name ${dataset_name} \
# --load_sc_dataset ${load_sc_dataset} \
# --load_cell_type_labels ${load_cell_type_labels} \
# --output_run_ID ${SLURM_JOB_ID} \
# --output_dir outputs/${SLURM_JOB_ID}_${dataset_name}_${dropout_prob}_dropout \
# --dropout_prob ${dropout_prob} \
# --total_epoch 2 --feature_AE_epoch 2 2 --graph_AE_epoch 2 --cluster_AE_epoch 2 

# sbatch --export=dataset_name=Biase,load_sc_dataset=Biase_expression.csv,load_cell_type_labels=Biase_cell_label.csv,dropout_prob=0.1 run.sh
# sbatch --export=dataset_name=Deng,load_sc_dataset=Deng_expression.csv,load_cell_type_labels=Deng_cell_label.csv,dropout_prob=0.1 run.sh
# sbatch --export=dataset_name=Goolam,load_sc_dataset=Goolam_expression.csv,load_cell_type_labels=Goolam_cell_label.csv,dropout_prob=0.1 run.sh
# sbatch --export=dataset_name=Guo,load_sc_dataset=Guo_expression.csv,load_cell_type_labels=Guo_cell_label.csv,dropout_prob=0.1 run.sh
# sbatch --export=dataset_name=Pollen,load_sc_dataset=Pollen_expression.csv,load_cell_type_labels=Pollen_cell_label.csv,dropout_prob=0.1 run.sh
# sbatch --export=dataset_name=Treutlein,load_sc_dataset=Treutlein_expression.csv,load_cell_type_labels=Treutlein_cell_label.csv,dropout_prob=0.1 run.sh
# sbatch --export=dataset_name=Usoskin,load_sc_dataset=Usoskin_expression.csv,load_cell_type_labels=Usoskin_cell_label.csv,dropout_prob=0.1 run.sh
# sbatch --export=dataset_name=Yan,load_sc_dataset=Yan_expression.csv,load_cell_type_labels=Yan_cell_label.csv,dropout_prob=0.1 run.sh

# sbatch --export=dataset_name=Biase,load_sc_dataset=Biase_expression.csv,load_cell_type_labels=Biase_cell_label.csv,dropout_prob=0 run.sh
# sbatch --export=dataset_name=Deng,load_sc_dataset=Deng_expression.csv,load_cell_type_labels=Deng_cell_label.csv,dropout_prob=0 run.sh
# sbatch --export=dataset_name=Goolam,load_sc_dataset=Goolam_expression.csv,load_cell_type_labels=Goolam_cell_label.csv,dropout_prob=0 run.sh
# sbatch --export=dataset_name=Guo,load_sc_dataset=Guo_expression.csv,load_cell_type_labels=Guo_cell_label.csv,dropout_prob=0 run.sh
# sbatch --export=dataset_name=Pollen,load_sc_dataset=Pollen_expression.csv,load_cell_type_labels=Pollen_cell_label.csv,dropout_prob=0 run.sh
# sbatch --export=dataset_name=Treutlein,load_sc_dataset=Treutlein_expression.csv,load_cell_type_labels=Treutlein_cell_label.csv,dropout_prob=0 run.sh
# sbatch --export=dataset_name=Usoskin,load_sc_dataset=Usoskin_expression.csv,load_cell_type_labels=Usoskin_cell_label.csv,dropout_prob=0 run.sh
# sbatch --export=dataset_name=Yan,load_sc_dataset=Yan_expression.csv,load_cell_type_labels=Yan_cell_label.csv,dropout_prob=0 run.sh

