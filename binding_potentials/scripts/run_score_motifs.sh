#!/bin/bash
#SBATCH --mem-per-cpu=8G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yiming.kang@wustl.edu

bash score_motifs.sh ../resources/TFs_of_interest.txt ../resources/scertf_pfms/ ../resources/rsat_prom_-800_-100_lite.fasta ../output/ 
