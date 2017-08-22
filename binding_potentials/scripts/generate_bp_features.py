#!/usr/bin/python
import numpy as np
import sys
import glob
import os.path


def parse_promoter_data(filename):
	prom_bed = np.loadtxt(filename, dtype=str, delimiter='\t')
	prom_dict = {}
	for i in range(len(prom_bed)):
		prom_dict[prom_bed[i,3]] = {'length':int(prom_bed[i,2])-int(prom_bed[i,1]), 
									'strand':prom_bed[i,5]}
	return prom_dict


def parse_fimo_data(filename, prom_dict, atg_dist=100):
	fimo_data = np.loadtxt(filename, dtype=str, delimiter='\t')
	fimo_dict = {}
	for i in range(len(fimo_data)):
		g = fimo_data[i,1]
		p = (int(fimo_data[i,2]) + int(fimo_data[i,3])) /2
		if prom_dict[g]['strand'] == '+':
			d = prom_dict[g]['length'] + atg_dist - p
		else:
			d = atg_dist + p
		try:
			fimo_dict[g].append(d)
		except KeyError:
			fimo_dict[g] = [d]
	return fimo_dict


def calculate_hit_dist(genes, fimo_dict):
	dist_mtx = np.ones((len(genes),4))*1000
	for i in range(len(genes)):
		if genes[i] in fimo_dict.keys():
			ds = sorted(fimo_dict[genes[i]])

			if len(ds) > 1:
				pdists = []
				for j in range(len(ds)-1):
					for k in range(j+1, len(ds)):
						pdists.append(abs(ds[j]-ds[k]))
				dist_mtx[i,0] = min(pdists)

			ds = [0] + ds
			d_diffs = [ds[i+1]-ds[i] for i in range(len(ds)-1)]
			dist_mtx[i,1:min(4,len(d_diffs)+1)] = d_diffs[:min(3,len(d_diffs))]
	return dist_mtx


## get promoter data
prom_dict = parse_promoter_data('../resources/orf_coding_all_R61-1-1_20080606.promoter_-800_-100.bed')

## generate features
out_header = ['#sequence_name','sum_score','max_score','count',
				'dist_closest','dist_atg_1st','dist_atg_2nd','dist_atg_3rd']
for file in glob.glob('../output/*.summary'):
	tf = os.path.basename(file).split('.')[0]
	print '... working on', tf
	## get processed hit scores
	scores = np.loadtxt(file, dtype=str, delimiter='\t', usecols=[1,2,4,6])
	genes = scores[:,0]
	## get fimo hits
	fimo_dict = parse_fimo_data('../output/'+ tf +'/fimo.txt', prom_dict)
	dists = calculate_hit_dist(genes, fimo_dict)
	## compile results
	out = np.hstack(( scores, dists ))
	out = np.vstack(( np.array(out_header)[np.newaxis], out ))
	np.savetxt('../output/'+ tf +'.scores.txt', out, fmt='%s', delimiter='\t')



