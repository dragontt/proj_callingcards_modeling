#!/bin/bash
#SBATCH --mem-per-cpu=24G
#SBATCH -n 10
#SBATCH -o ../log/run_fit_model.cc+ca+wt+bp.out
#SBATCH -e ../log/run_fit_model.cc+ca+wt+bp.err
#SBATCH -J cc_modeling
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yiming.kang@wustl.edu

module load pandas
module load scipy
module load scikit-learn
module load matplotlib
module load biopython

# python fit_model.py -m interactive_bp_feature_learning -t binned_promoter -c ../output/ -d ../McIsaac_ZEV_DE/ -a ../chromatin_access/output/chromatin_access_features.txt -w ../resources/
# python fit_model.py -m interactive_bp_feature_learning -t binned_promoter -c ../output/ -d ../McIsaac_ZEV_DE/

python fit_model.py -m interactive_bp_feature_learning -t binned_promoter -c ../output/ -d ../resources/ -a ../chromatin_access/output/chromatin_access_features.txt -w ../resources/ -p ../binding_potentials/output/