#!/usr/bin/python
import sys
import numpy as np
from scipy.stats.mstats import gmean

histone_marks = sorted(["H3K4me3", "H3K36me3", "H4K16ac", "H3K27ac", "H3K79me"])
print histone_marks
num_hms = len(histone_marks)
gene_pos_types = [-3,-2,-1,0]

## parse nucleosome data
nuc_data = np.loadtxt("../resources/molcel_5341_mmc4", delimiter=',', dtype=str)
nuc_header = nuc_data[0,]
nuc_dict = {}
for i in range(2,len(nuc_data)):
	nuc_id = nuc_data[i,0]
	nuc_dict[nuc_id] = {}
	for hm in histone_marks:
		j = np.where(nuc_header == hm)[0]
		if len(j) > 0:
			# nuc_dict[nuc_data[i,0]][hm] = gmean(np.array(nuc_data[i,j], dtype=float))
			nuc_dict[nuc_id][hm] = float(nuc_data[i,j[0]])

## parse mapped modifications
gene_data = np.loadtxt("../resources/mmc3.csv", delimiter=',', dtype=str, skiprows=1)
genes = np.unique(gene_data[:,5])
genes = np.delete(genes, np.argwhere(genes == ''))
feat_mtx = np.empty((0,num_hms*4+1))
for gene in genes:
	i = np.where(gene_data[:,5] == gene)[0]
	gene_pos = np.array(gene_data[i,6], dtype=float)
	nuc_ids = gene_data[i,0]
	## each row has features: gene_pos x histone marks 
	prom3_feats, prom2_feats, prom1_feats = [], [], []
	genebody_feats = [[] for j in range(num_hms)]
	for j in range(len(gene_pos)):
		if gene_pos[j] == -3: 
			for k in range(num_hms):
				prom3_feats.append(nuc_dict[nuc_ids[j]][histone_marks[k]]) 
		elif gene_pos[j] == -2: 
			for k in range(num_hms):
				prom2_feats.append(nuc_dict[nuc_ids[j]][histone_marks[k]])
		elif gene_pos[j] == -1: 
			for k in range(num_hms):
				prom1_feats.append(nuc_dict[nuc_ids[j]][histone_marks[k]]) 
		elif gene_pos[j] > 0:
			for k in range(num_hms):
				genebody_feats[k].append(nuc_dict[nuc_ids[j]][histone_marks[k]])
	## sum nonpositive gene body features
	for k in range(num_hms):
		tmp = np.array(genebody_feats[k])
		genebody_feats[k] = np.mean(tmp[tmp < 0]) if len(tmp[tmp < 0]) > 0 else 0
	## append data
	prom3_feats = [0]*num_hms if len(prom3_feats) == 0 else prom3_feats
	prom2_feats = [0]*num_hms if len(prom2_feats) == 0 else prom2_feats
	prom1_feats = [0]*num_hms if len(prom1_feats) == 0 else prom1_feats
	feat_row = np.concatenate(([gene], prom3_feats, prom2_feats, prom1_feats, genebody_feats))
	feat_mtx = np.vstack((feat_mtx, feat_row))

mtx_header = np.concatenate((['orf'], 
							[hm + '_prom_-3' for hm in histone_marks], 
							[hm + '_prom_-2' for hm in histone_marks], 
							[hm + '_prom_-1' for hm in histone_marks], 
							[hm + '_body' for hm in histone_marks]))
feat_mtx = np.vstack((mtx_header, feat_mtx))
## save feature matrix
np.savetxt('../output/chromatin_access_features.txt', feat_mtx, fmt='%s', delimiter='\t')




