#!/usr/bin/python
import sys
import os
import re
import glob
import numpy as np
import argparse

"""
Example usage: 
python compute_orf_hops.py -r ../resources/ -u -1000 -d 100 -o ../output/
"""


def parse_args(argv):
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-r","--resources_dir")
    parser.add_argument("-f","--orf_fasta", default="../resources/orf_coding_all_R61-1-1_20080606.fasta")
    parser.add_argument("-b","--background_gnashy", default="../resources/NOTF_Minus_Adh1_2015_17_combined.gnashy")
    parser.add_argument("-u","--promoter_upstream", type=int)
    parser.add_argument("-d","--promoter_downstream", type=int)
    parser.add_argument("-o","--output_dir")
    parsed = parser.parse_args(argv[1:])
    return parsed


def convert_orf_to_bed(file_in, file_out):
	CH_DICT = {'Chr I':'chr1', 'Chr II':'chr2', 'Chr III':'chr3', 
				'Chr IV':'chr4', 'Chr V':'chr5', 'Chr VI':'chr6', 
				'Chr VII':'chr7', 'Chr VIII':'chr8', 'Chr IX':'chr9', 
				'Chr X':'chr10', 'Chr XI':'chr11', 'Chr XII':'chr12', 
				'Chr XIII':'chr13', 'Chr XIV':'chr14', 'Chr XV':'chr15', 
				'Chr XVI':'chr16', 'Chr Mito':'chrm'}

	## load fasta file
	f = open(file_in, 'r')
	lines = f.readlines()
	f.close()
	bed_out = []
	for i in range(len(lines)):
		## parse orf header 
		if lines[i].startswith('>'):
			x = lines[i].strip().strip('>').split(', ')
			g = x[0].split(' ')[0]
			s = '+' if g.split('-')[0].endswith('W') else '-'

			if x[1].startswith('2-micron plasmid'): ## deal with plasmid 
				l = re.split('-|,| from ',x[1].strip('2-micron '))
				ch = l[0]
				pos = np.array(l[1:], dtype=int)
				atg = np.min(pos) if s == '+' else np.max(pos)
			else: ## deal with other chromsomes
				l = re.split('-|,| from ',x[1])
				ch = CH_DICT[l[0]]
				pos = np.array(l[1:], dtype=int)
				atg = np.min(pos) if s == '+' else np.max(pos)
			## update bed structure
			bed_out.append([ch, atg, atg, g, '.', s])
	## save bed file
	np.savetxt(file_out, np.array(bed_out), fmt='%s', delimiter='\t')

	## sort bed file by chromosome and then by start position
	os.system('mv '+ file_out +' '+ file_out +'_tmp')
	os.system('sort -k1,1 -k2,2n '+ file_out +'_tmp > '+ file_out)
	os.system('rm '+ file_out +'_tmp')


def convert_gnashy_to_bed(file_in, file_out):
	## load gnashy file
	nashy = np.loadtxt(file_in, dtype=int, delimiter='\t')
	out = []
	for i in range(len(nashy)):
		## parse lines iteratively
		ch = 'chr'+str(nashy[i,0])
		pos = nashy[i,1]
		num = nashy[i,2]
		name = 'hop'+str(i+1)
		## update bed structure
		out.append([ch, pos, pos, name, num, '+'])
	## save bed file
	np.savetxt(file_out, np.array(out), fmt='%s', delimiter='\t')

	## sort bed files by chromosome and then by start position
	os.system('mv '+ file_out +' '+ file_out +'_tmp')
	os.system('sort -k1,1 -k2,2n '+ file_out +'_tmp > '+ file_out)
	os.system('rm '+ file_out +'_tmp')


def map_hops_to_orf(file_orf, file_in, file_out, promoter_range=False):
	## load bed files
	orfs = np.loadtxt(file_orf, dtype=str, delimiter='\t')
	hops = np.loadtxt(file_in, dtype=str, delimiter='\t')
	
	M = []
	for i in range(len(orfs)):
		## get info of each orf and find the matching chromosome in hops data
		orf_ch, orf_pos, orf_strand = orfs[i, [0,1,5]]
		hops_subset = hops[hops[:,0]==orf_ch, :] 

		## iterate thru hops
		for j in range(len(hops_subset)):
			hop_pos = hops_subset[j,1]
			## calculate hop to pos distance: '-' for upstream, '+' for downstream
			dist = int(hop_pos)-int(orf_pos) if orf_strand == '+' else int(orf_pos)-int(hop_pos)
			## check if hops within promoter range
			if promoter_range and (dist < promoter_range[0] or dist > promoter_range[1]):
				continue
			## update
			M.append(list(hops_subset[j,:]) + list(orfs[i,:]) + [dist])

	## save output 
	M = np.array(M)
	np.savetxt(file_out, M, fmt='%s', delimiter='\t')



def count_total_hops_and_reads(res_dir, file_tbl):
	writer = open(file_tbl, "w")
	writer.write("#file_basename\thop_cnt\tread_cnt\n")
	for file_gnashy in glob.glob(res_dir +"/*.gnashy"):
		sample_basename = os.path.splitext(os.path.basename(file_gnashy))[0]
		reads = np.loadtxt(file_gnashy, usecols=[2])
		hops_cnt = len(reads)
		reads_cnt = sum(reads)
		writer.write("%s\t%d\t%d\n" % (sample_basename, hops_cnt, reads_cnt))
	writer.close()


def main(argv):
	parsed = parse_args(argv)
	if parsed.output_dir.endswith('/'):
		parsed.output_dir = parsed.output_dir[:-1]
	## prepare promoter parameter
	promoter_range = [parsed.promoter_upstream, parsed.promoter_downstream]
	if promoter_range[0] > 0:
		promoter_range[0] = -1*promoter_range[0]

	## convert orf fasta to bed format
	print "Converting orf to bed"
	file_orf_fasta = parsed.orf_fasta
	file_orf_bed = parsed.output_dir +'/'+ os.path.basename(file_orf_fasta).strip("fasta")+"ATG.bed"
	convert_orf_to_bed(file_orf_fasta, file_orf_bed)

	## convert gnashy files to bed format
	print "Converting gnashy to bed"
	for file_gnashy in glob.glob(parsed.resources_dir +"/*.gnashy"):
		print "... working on", os.path.basename(file_gnashy)
		file_basename = os.path.splitext(os.path.basename(file_gnashy))[0]
		file_experiment_bed = parsed.output_dir +'/'+ file_basename +'.bed'
		convert_gnashy_to_bed(file_gnashy, file_experiment_bed)

	## count total hops for normalization
	print "Counting total hops"
	file_totals = parsed.output_dir +"/total_hops_and_reads.tbl"
	count_total_hops_and_reads(parsed.resources_dir, file_totals)

	## map hops to each orf, and each hop is allowed to multiple orfs
	print "Converting gnashy to bed"
	files_experiment_bed = glob.glob(parsed.output_dir +"/*.bed")
	files_experiment_bed.remove(file_orf_bed)
	for file_experiment_bed in files_experiment_bed:
		print "... working on", os.path.basename(file_experiment_bed)
		file_orf_hops = parsed.output_dir +'/'+ os.path.basename(file_experiment_bed).strip('bed') +'orf_hops'
		map_hops_to_orf(file_orf_bed, file_experiment_bed, file_orf_hops, promoter_range)

	## clean up
	os.system("rm "+ parsed.output_dir +"/*.bed")


if __name__ == "__main__":
    main(sys.argv)
