#!/bin/bash
#SBATCH -n 1  
#SBATCH -N 1 
#SBATCH --mem=20G
#SBATCH -D ./
#SBATCH -o novoalign.out # standard out goes here
#SBATCH -e novoalign.err # standard error goes here
#SBATCH -J CC_S288C
#SBATCH --array=1-28%10  # Adding this means that you now have an enivornment variable called SLURM_ARRAY_TASK_ID
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yiming.kang@wustl.edu

module load novoalign/3.07.00
module load samtools


filename=$( cut -f 1 ../SRA_files/cc_filelist.txt | sed -n ${SLURM_ARRAY_TASK_ID}p )  # extracts line $SLURM_ARRAY_TASK_ID from file

# srun python map_reads_novo.py -f ${filename} -g S288C_Plasmids -l 40 -p False -o output_and_analysis
srun python -u map_reads_novo.py -f ../SRA_files/${filename} -g ../S288C/S288C_reference_sequence_R61-1-1_20080605.nix -l 100 -p False -o ../output_and_analysis