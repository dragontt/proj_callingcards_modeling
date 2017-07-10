#!/bin/bash
#SBATCH -n 8  
#SBATCH -N 1 
#SBATCH --mem=30000
#SBATCH -o CFR.out #standard out goes here
#SBATCH -e CFR.err # standard error goes here
#SBATCH -J CFR
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yiming.kang@wustl.edu

module load biopython

python cc_filter_reads.py -r1 ../raw/Undetermined_S0_L001_R1_001.fastq  -r2 ../raw/Undetermined_S0_L001_R2_001.fastq -b ../raw/barcodes.txt --hammp 0 --hammt 0 -o ../output_and_analysis


