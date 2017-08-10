#!/usr/bin/python
import os.path
import sys
import numpy as np
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

resources_dir = '../resources/'
label_filename = '../resources/optimized_cc_subset.txt'
output_dir = '../output/'

# resources_dir = '../resources2/'
# label_filename = '../resources2/optimized_cc_kemmeren_subset.txt'
# output_dir = '../output2/'


label_dict = {}
label_data = np.loadtxt(label_filename, dtype=str, skiprows=1, delimiter='\t')
for i in range(len(label_data)):
	sample = label_data[i,0]
	intersection = [x.strip('"') for x in label_data[i,1].strip('{').strip('}').split(', ')]
	bound = [x.strip('"') for x in label_data[i,2].strip('{').strip('}').split(', ')]
	label_dict[sample] = {'intersection': intersection, 'bound': bound}


for filename in glob.glob(''.join([resources_dir, "*.DE.tsv"])):
	sample = os.path.basename(filename).split('.')[0]
	print sample

	cc_data = np.loadtxt(''.join([output_dir, sample, '.sig_prom.gnashy']), dtype=str, usecols=[0,7], delimiter='\t')
	de_data = np.loadtxt(filename, dtype=str, delimiter='\t')
	cc_dict = {}
	for i in range(len(cc_data)):
		cc_dict[cc_data[i,0]] = float(cc_data[i,1])
	de_dict = {}
	for i in range(len(de_data)):
		de_dict[de_data[i,0]] = np.array(de_data[i,1:], dtype=float)
	
	orfs = np.array([orf for orf in np.intersect1d(cc_data[:,0], de_data[:,0]) if orf.startswith('Y') and orf.split('-')[0][-1] in ['W','C']])
	cc_pvals = np.array([cc_dict[orf] for orf in orfs])
	cc_pvals[-np.log10(cc_pvals) > 15] = 10**(-15)
	de_pvals = np.array([de_dict[orf][0] for orf in orfs])
	de_lfcs = np.array([de_dict[orf][1] for orf in orfs])
	de_lfcs[de_lfcs > 15] = 15
	de_lfcs[de_lfcs < -15] = -15

	plt.figure(num=None, figsize=(10, 7), dpi=300)
	plt.scatter(-np.log10(cc_pvals)[de_pvals > .05], np.abs(de_lfcs)[de_pvals > .05], c='r', s=10, alpha=.2, label='p > .05 (DE)')
	plt.scatter(-np.log10(cc_pvals)[de_pvals <= .05], np.abs(de_lfcs)[de_pvals <= .05], c='b', s=10, alpha=.2, label='p <= .05 (DE)')
	if sample in label_dict.keys():
		indx = [np.where(orfs == orf)[0][0] for orf in label_dict[sample]['bound'] if orf in orfs] 
		plt.scatter(-np.log10(cc_pvals)[indx], np.abs(de_lfcs)[indx], facecolors='none', s=30, linewidths=1, edgecolors='g', label='Optimized bound')
		indx = [np.where(orfs == orf)[0][0] for orf in label_dict[sample]['intersection'] if orf in orfs] 
		plt.scatter(-np.log10(cc_pvals)[indx], np.abs(de_lfcs)[indx], facecolors='none', s=30, linewidths=1, edgecolors='k', label='Optimized intersection')
	plt.xlabel('- log10 p (CallingCards)')
	plt.ylabel('| log2 FC | (DE)')
	plt.ylim([0,3])
	plt.legend()
	plt.savefig(''.join([output_dir, 'fig_cc_vs_de.', sample, '.pdf']), fmt='pdf')	

