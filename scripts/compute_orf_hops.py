#!/usr/bin/python
import sys
import os
import re
import glob
import numpy as np
import argparse

"""
Example usage: 
python compute_orf_hops.py -r ../resources/ -o ../output/ -pu -1000 -pd 100
"""


def parse_args(argv):
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-r","--resources_dir")
    parser.add_argument("-f","--orf_fasta", default="../resources/orf_coding_all_R61-1-1_20080606.fasta")
    parser.add_argument("-b","--background_gnashy", default="../resources/NOTF_Minus_Adh1_2015_17_combined.gnashy")
    parser.add_argument("-pu","--promoter_upstream", type=int)
    parser.add_argument("-pd","--promoter_downstream", type=int)
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
				g_start = np.min(pos) if s == '+' else np.max(pos)
				g_end = np.max(pos) if s == '+' else np.min(pos)
			else: ## deal with other chromsomes
				l = re.split('-|,| from ',x[1])
				ch = CH_DICT[l[0]]
				pos = np.array(l[1:], dtype=int)
				g_start = np.min(pos) if s == '+' else np.max(pos)
				g_end = np.max(pos) if s == '+' else np.min(pos)
			## update bed structure
			bed_out.append([ch, g_start, g_end, g, '.', s])
	## save bed file
	np.savetxt(file_out, np.array(bed_out), fmt='%s', delimiter='\t')

	## sort bed file by chromosome and then by start position
	os.system('mv '+ file_out +' '+ file_out +'_tmp')
	os.system('sort -k1,1 -k2,2n '+ file_out +'_tmp > '+ file_out)
	os.system('rm '+ file_out +'_tmp')


def convert_orf_to_promoter(file_orf, file_prom, promoter_range):
	## get gene body
	gene_body_dict = {}
	orf = np.loadtxt(file_orf, dtype=str)
	for i in range(orf.shape[0]):
		ch = orf[i,0]
		pos = sorted(np.array(orf[i,1:3], dtype=int))
		try:
			gene_body_dict[ch].append(pos)
		except KeyError:
			gene_body_dict[ch] = [pos]
	## get promoter by substracting gene body
	prom_tbl = []
	for i in range(orf.shape[0]):
		ch = orf[i,0]
		atg = int(orf[i,1])
		name = orf[i,3]
		strand = orf[i,5]
		if strand == "+":
			p_start = max(0, atg + promoter_range[0])
			p_end = atg + promoter_range[1]
			## check if the promoter is in the gene body of neighbors
			for pos in gene_body_dict[ch]:
				if p_start > pos[0] and p_start < pos[1]:
					p_start = pos[1]
		else:
			p_start = atg - promoter_range[0]
			p_end = max(0, atg - promoter_range[1])
			## check if the promoter is in the gene body of neighbors
			for pos in gene_body_dict[ch]:
				if p_start > pos[0] and p_start < pos[1]:
					p_start = pos[0]
		## store promoter data
		if (strand == "+" and p_start <= atg) or (strand == "-" and p_start >= atg):
			prom_tbl.append([ch, p_start, p_end, name, ".", strand])
	np.savetxt(file_prom, np.array(prom_tbl), fmt="%s", delimiter="\t")


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


def map_hops_to_orf(file_orfs, file_prom, file_hops, file_out):
	## load bed files
	orfs = np.loadtxt(file_orfs, dtype=str, delimiter='\t')
	promoters = np.loadtxt(file_prom, dtype=str, delimiter='\t')
	hops = np.loadtxt(file_hops, dtype=str, delimiter='\t')
	## get orfs dict
	orfs_dict = {}
	for i in range(len(orfs)):
		orfs_dict[orfs[i,3]] = orfs[i,:]
	M = []
	for i in range(len(promoters)):
		## get info of each orf and find the matching chromosome in hops data
		orf_ch, prom_start, prom_stop, orf_name, score, orf_strand = promoters[i,:]
		prom_pos = sorted([int(prom_start), int(prom_stop)])
		orf_atg = int(orfs_dict[orf_name][1])
		hops_subset = hops[hops[:,0]==orf_ch, :] 
		## iterate thru hops
		for j in range(len(hops_subset)):
			hop_pos = int(hops_subset[j,1])
			## check if hops within promoter range
			if hop_pos >= prom_pos[0] and hop_pos <= prom_pos[1]:
				## calculate hop to pos distance: '-' for upstream, '+' for downstream
				dist = hop_pos-orf_atg if orf_strand == '+' else orf_atg-hop_pos
				## update
				M.append(list(hops_subset[j,:]) + list(orfs_dict[orf_name]) + [dist])
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
	file_orf_fasta = parsed.orf_fasta
	file_orf_atg = parsed.output_dir +'/'+ os.path.basename(file_orf_fasta).strip("fasta")+"ATG.bed"
	file_orf_prom = parsed.output_dir +'/'+ os.path.basename(file_orf_fasta).strip("fasta")+"promoter.bed"
	print "Converting orf to ATG"
	convert_orf_to_bed(file_orf_fasta, file_orf_atg)
	print "Converting orf to promoter"
	convert_orf_to_promoter(file_orf_atg, file_orf_prom, promoter_range)

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
	print "Matching hops to orf promoters"
	files_experiment_bed = glob.glob(parsed.output_dir +"/*.bed")
	files_experiment_bed.remove(file_orf_atg)
	for file_experiment_bed in files_experiment_bed:
		file_basename_experiment_bed = os.path.basename(file_experiment_bed)
		if not file_basename_experiment_bed.startswith('orf'):
			print "... working on", file_experiment_bed
			file_orf_hops = parsed.output_dir +'/'+ file_basename_experiment_bed.strip('bed') +'orf_hops'
			map_hops_to_orf(file_orf_atg, file_orf_prom, file_experiment_bed, file_orf_hops)
			## clean up
			# os.system("rm "+ file_experiment_bed)


if __name__ == "__main__":
    main(sys.argv)
