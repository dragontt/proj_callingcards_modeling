#!/bin/bash
#SBATCH --mem-per-cpu=20G
#SBATCH -n 1
#SBATCH -o ../log/run_fit_model.bfs.Cbf1.out
#SBATCH -e ../log/run_fit_model.bfs.Cbf1.err
#SBATCH -J cc_modeling
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yiming.kang@wustl.edu

module load pandas
module load scipy
module load scikit-learn
module load matplotlib
module load biopython

# python fit_model.py -m interactive_bp_feature_learning -t binned_promoter -c ../output/ -d ../McIsaac_ZEV_DE/matched_RNAseq_DE.20min/ -o ../output/tmp.zev_cc.matched_RNAseq_DE.20min.txt
# python fit_model.py -m interactive_bp_feature_learning -t binned_promoter -c ../output/ -d ../McIsaac_ZEV_DE/matched_RNAseq_DE.10min/ -a ../chromatin_access/output/chromatin_access_features.txt -w ../resources/ -o ../output/tmp.zev_cc+ca+wt.matched_RNAseq_DE.10min.txt

# python fit_model.py -m interactive_bp_feature_learning -t binned_promoter -c ../output/ -d ../resources/ -o ../output/tmp.RNAseq_top500.cc.txt
# python fit_model.py -m interactive_bp_feature_learning -t binned_promoter -c ../output/ -d ../resources/ -a ../chromatin_access/output/chromatin_access_features.txt -w ../resources/ -o ../output/tmp.RNAseq_top500.cc+ca+wt.txt

# python fit_hierarchical_model.py -d ../resources/ -a ../chromatin_access/output/chromatin_access_features.txt -w ../resources/ -o ../output/tmp.ca+wt.bo.txt


# python fit_model.py -m interactive_bp_feature_learning -t pvals -c ../output3/ -d ../McIsaac_ZEV_DE/ -o ../output/tmp.ZEV-15min_x_5TFs.RF.CC
# python fit_model.py -m interactive_bp_feature_learning -t pvals -c ../output3/ -d ../McIsaac_ZEV_DE/ -a ../chromatin_access/output/chromatin_access_features.txt -o ../output/tmp.ZEV-15min_x_5TFs.RF.CC+CA



python fit_model.py -m interactive_bp_feature_ranking -t pvals -c ../output3/ -a ../chromatin_access/output/chromatin_access_features.txt -d ../McIsaac_ZEV_DE/ -v YJR060W
