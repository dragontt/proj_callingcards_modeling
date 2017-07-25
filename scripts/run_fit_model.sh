#!/bin/bash
#SBATCH --mem-per-cpu=24G
#SBATCH -n 1
#SBATCH -o log/run_fit_model.out
#SBATCH -e log/run_fit_model.err
#SBATCH -J cc_modeling
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yiming.kang@wustl.edu

module load scipy
module load scikit-learn
module load matplotlib

python fit_model.py -m holdout_feature_variation -c ../output/ -o ../resources/optimized_cc_subset.txt -f ../output/feature_holdout_analysis.6_mimic_cc > ../log/gb_bo.out 2> ../log/gb_bo.err
