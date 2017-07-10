#!/bin/bash
#SBATCH -n 1  
#SBATCH -N 1 
#SBATCH --mem=30000
#SBATCH -o NOVO.out # standard out goes here
#SBATCH -e NOVO.err # standard error goes here
#SBATCH -J NOVO
#SBATCH --array=1-9   # Adding this means that you now have an enivornment variable called SLURM_ARRAY_TASK_ID
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yiming.kang@wustl.edu

module load novoalign/3.07.00
module load samtools


filename=$( sed -n ${SLURM_ARRAY_TASK_ID}p ../output_and_analysis.v2/cc_filelist.txt)  # extracts line $SLURM_ARRAY_TASK_ID from file

srun python map_reads_novo.py -f ${filename} -g S288C_Plasmids -l 40 -p False -o ../output_and_analysis.v2

