#!/bin/bash
#SBATCH -N 1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --array=1-9
#SBATCH -D ./
#SBATCH -o log/run_fit_rf_model2_%A_%a.out
#SBATCH -e log/run_fit_rf_model2_%A_%a.err
#SBATCH -J cc_model

read sys_name common_name < <( sed -n ${SLURM_ARRAY_TASK_ID}p CCProcessed_16TFs/valid_CCxZEV_TFs.txt )
set -e 

output_dir=analysis_cc_hm_feature/
mkdir -p ${output_dir}
python -u scripts/fit_rf_model2.py -c CCProcessed_16TFs/ -d McIsaac_ZEV_DE/ -o ${output_dir} --tf_names ${sys_name},${common_name}