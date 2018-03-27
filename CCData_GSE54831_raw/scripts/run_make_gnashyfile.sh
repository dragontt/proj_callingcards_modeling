#!/bin/bash
#SBATCH -n 8  
#SBATCH -N 1 
#SBATCH --mem=30000
#SBATCH -o gnashy_mapping.out # standard out goes here
#SBATCH -e gnashy_mapping.err # standard error goes here
#SBATCH -J CC_S288C
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yiming.kang@wustl.edu

module load pysam/0.8.3
module load pandas/0.17.1

python make_gnashyfile.py -p ../output_and_analysis/
