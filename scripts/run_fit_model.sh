#!/bin/bash
#SBATCH --mem-per-cpu=64G
#SBATCH -n 1
#SBATCH -o ../log/run_fit_model.out
#SBATCH -e ../log/run_fit_model.err
#SBATCH -J cc_modeling
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yiming.kang@wustl.edu

module load pandas
module load scipy
module load scikit-learn
module load matplotlib
module load biopython

python fit_model.py -m holdout_feature_regression -t binned_promoter -c ../output/ -d ../resources/ > ../log/bin_prom.GP.out 2> ../log/bin_prom.GP.err
