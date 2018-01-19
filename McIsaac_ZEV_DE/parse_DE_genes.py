#!/usr/bin/python
import numpy as np
import sys

def process_DE(tf, tf_sys, tf_indx, lys_cond, filename):
	genes = data[1:,0]
	lfc = np.array(data[1:,tf_indx], dtype=float)
	cutoff = sorted(np.abs(lfc))[::-1][matched_DE_count[tf][lys_cond]]
	indx = np.where(np.abs(lfc) > cutoff)[0]
	de = np.ones(len(lfc))
	de[indx] = 0
	header = ['##gene','de(0)|not_de(1)','log2_fold_change']
	out = np.concatenate((genes.reshape(-1,1), de.reshape(-1,1), lfc.reshape(-1,1)), axis=1)
	out = np.vstack((np.array(header)[np.newaxis], out))
	np.savetxt(filename, out, fmt='%s', delimiter='\t')


data = np.loadtxt('cleaned_data.txt', dtype=str)
## 10 min
matched_DE_count = {'LYS14': {'minusLys': 1884, 'plusLys': 447}, 
					'RGT1': {'minusLys': 2193, 'plusLys': 2686}}
process_DE('LYS14', 'YDR034C', 4, 'minusLys', 'YDR034C-10min.match_minusLys.DE.txt')
process_DE('LYS14', 'YDR034C', 4, 'plusLys', 'YDR034C-10min.match_plusLys.DE.txt')
process_DE('RGT1', 'YKL038W', 12, 'minusLys', 'YKL038W-10min.match_minusLys.DE.txt')
process_DE('RGT1', 'YKL038W', 12, 'plusLys', 'YKL038W-10min.match_plusLys.DE.txt')

matched_DE_count = {'LYS14': {'minusLys': 500}, 
					'RGT1': {'minusLys': 500}}
process_DE('LYS14', 'YDR034C', 4, 'minusLys', 'YDR034C-10min.top_500.DE.txt')
process_DE('RGT1', 'YKL038W', 12, 'minusLys', 'YKL038W-10min.top_500.DE.txt')


## 15 min
matched_DE_count = {'LYS14': {'minusLys': 1884, 'plusLys': 447}, 
					'RGT1': {'minusLys': 2193, 'plusLys': 2686}}
process_DE('LYS14', 'YDR034C', 5, 'minusLys', 'YDR034C-15min.match_minusLys.DE.txt')
process_DE('LYS14', 'YDR034C', 5, 'plusLys', 'YDR034C-15min.match_plusLys.DE.txt')
process_DE('RGT1', 'YKL038W', 13, 'minusLys', 'YKL038W-15min.match_minusLys.DE.txt')
process_DE('RGT1', 'YKL038W', 13, 'plusLys', 'YKL038W-15min.match_plusLys.DE.txt')

matched_DE_count = {'LYS14': {'minusLys': 500}, 
					'RGT1': {'minusLys': 500}}
process_DE('LYS14', 'YDR034C', 5, 'minusLys', 'YDR034C-15min.top_500.DE.txt')
process_DE('RGT1', 'YKL038W', 13, 'minusLys', 'YKL038W-15min.top_500.DE.txt')

## 20 min
matched_DE_count = {'LYS14': {'minusLys': 1884, 'plusLys': 447}, 
					'RGT1': {'minusLys': 2193, 'plusLys': 2686}}
process_DE('LYS14', 'YDR034C', 6, 'minusLys', 'YDR034C-20min.match_minusLys.DE.txt')
process_DE('LYS14', 'YDR034C', 6, 'plusLys', 'YDR034C-20min.match_plusLys.DE.txt')
process_DE('RGT1', 'YKL038W', 14, 'minusLys', 'YKL038W-20min.match_minusLys.DE.txt')
process_DE('RGT1', 'YKL038W', 14, 'plusLys', 'YKL038W-20min.match_plusLys.DE.txt')

matched_DE_count = {'LYS14': {'minusLys': 500}, 
					'RGT1': {'minusLys': 500}}
process_DE('LYS14', 'YDR034C', 6, 'minusLys', 'YDR034C-20min.top_500.DE.txt')
process_DE('RGT1', 'YKL038W', 14, 'minusLys', 'YKL038W-20min.top_500.DE.txt')
