#!/bin/bash
#SBATCH --mem-per-cpu=8G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yiming.kang@wustl.edu

bash score_motifs.sh ../resources/TFs_of_interest.txt ../resources/scertf_pfms/ ../resources/orf_coding_all_R61-1-1_20080606.promoter_-800_-100.fasta ../output/ 
