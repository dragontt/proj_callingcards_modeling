#!/bin/bash
#SBATCH -n 8  
#SBATCH -N 1 
#SBATCH --mem=30000
#SBATCH -o MGF.out # standard out goes here
#SBATCH -e MGF.err # standard error goes here
#SBATCH -J MGF
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yiming.kang@wustl.edu

module load pysam/0.8.3
module load pandas/0.17.1

python make_gnashyfile.py -p ../output_and_analysis.v2
