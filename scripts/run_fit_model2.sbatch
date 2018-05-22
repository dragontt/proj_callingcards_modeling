#!/bin/bash
#SBATCH -N 1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --array=1-9
#SBATCH -D ./
#SBATCH -o log/fit_model2_%A_%a.out
#SBATCH -e log/fit_model2_%A_%a.err
#SBATCH -J cc_model

read sys_name common_name < <( sed -n ${SLURM_ARRAY_TASK_ID}p CCProcessed_16TFs/valid_CCxZEV_TFs.txt )
set -e 

output_dir=analysis_single_cc_feature/
mkdir -p ${output_dir}
python -u scripts/fit_model2.py -a rf -c CCProcessed_16TFs/ -d McIsaac_ZEV_DE/ -o ${output_dir} --tf_names ${sys_name},${common_name}


# output_dir=analysis_cc_26hm_feature/
# mkdir -p ${output_dir}
# python -u scripts/fit_model2.py -a rf -c CCProcessed_16TFs/ -d McIsaac_ZEV_DE/ -hm histone_marks/output/hm_features.all.txt -o ${output_dir} --tf_names ${sys_name},${common_name}


# output_dir=analysis_cc_hm_wt_feature/
# mkdir -p ${output_dir}
# python -u scripts/fit_model2.py -a rf -c CCProcessed_16TFs/ -d McIsaac_ZEV_DE/ -hm histone_marks/output/hm_features.txt -wt McIsaac_ZEV_DE/all_expression/ -o ${output_dir} --tf_names ${sys_name},${common_name}