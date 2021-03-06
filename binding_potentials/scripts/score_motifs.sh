#!/bin/bash
#SBATCH --mem=8G
#SBATCH --ntasks-per-node=1

# Input variables
IN_TF_LIST=$1 	# list of tf names
IN_TF_PWM=$2	# directory of tf pwm
IN_PROMOTERS=$3	# promoter sequence file (e.g. yeast_promoter_seq/s_cerevisiae.promoters.fasta, fly_promoter_seq/rsat_dmel_upstream_-2000_+200.filtered.fasta)
OUT_FIMO=$4		# directory of fimo alignment output 

counter=0

while read -a line
do
	# process fimo scan
	counter=$[$counter +1]
	motif=${line[0]}
	echo  "*** Processing $motif ... $counter"
	
	if [ -f $IN_TF_PWM/$motif ]; then
		fimo -o $OUT_FIMO/$motif --thresh 5e-3 $IN_TF_PWM/$motif $IN_PROMOTERS
		sed ' 1d ' $OUT_FIMO/$motif/fimo.txt | cut -f 1,2,7 > $OUT_FIMO/$motif/temp.txt
		ruby estimate_affinity.rb -i $OUT_FIMO/$motif/temp.txt > $OUT_FIMO/${motif}.summary
		echo "*** Done"
	else
		echo "*** No motif exists"
	fi

done < $IN_TF_LIST

echo "*** ALL DONE! ***"
