#!/usr/bin/python
import sys
import numpy as np
import glob
import os.path
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def load_data(file):
	data = np.loadtxt(file)
	# data_sorted = data[np.argsort(data[:,1])[::-1],]
	data_sorted = np.array(sorted(data, key=lambda x: (x[1],x[0])))[::-1]
	return data_sorted


def combine_data(file_cc, file_zev):
	cc = np.loadtxt(file_cc, dtype=str)
	zev = np.loadtxt(file_zev, dtype=str, usecols=[0,2])

	zev_de = np.abs(np.array(zev[:,1], dtype=float))
	five_percentile = np.sort(zev_de)[::-1][int(np.floor(len(zev)*.05))]
	if five_percentile == 0:
		indx_pos = np.where(zev_de != 0)[0]	
	else:
		indx_pos = np.where(zev_de >= five_percentile)[0]
	zev[:,1] = -1
	zev[indx_pos,1] = 1
	orfs = np.intersect1d(cc[:,0], zev[:,0])
	data = []
	for orf in orfs:
		indx_cc = np.where(cc[:,0] == orf)[0][0]
		indx_zev = np.where(zev[:,0] == orf)[0][0]
		row = [zev[indx_zev,1]] + list(cc[indx_cc,1:])
		data.append(row)
	return np.array(data, dtype=float)


def cal_support_rates(data, step, bin, set_tie_rank=False):
	if set_tie_rank:
		data = data[:step*bin,]
		xpts, supp_rates = [], []
		rankings = rankdata(-data[:,1], method='max')
		for r in np.unique(rankings):
			i = np.where(rankings == r)[0][-1]
			x = data[:(i+1),0]
			xpts.append(i)
			supp_rates.append(float(len(x[x==1]))/len(x))
	else:
		xpts = range(bin)
		supp_rates = []
		for i in range(1,bin+1):
			x = data[:i*step,0]
			r = float(len(x[x==1])) / len(x)
			supp_rates.append(r)
	supp_rates = [r*100 for r in supp_rates]
	return (np.array(xpts), np.array(supp_rates))


def cal_auprc(file):
	data = np.loadtxt(file)
	return average_precision_score(data[:,0], data[:,1])


def plot_support_rate3(file_zev_cc, file_hu_cc, file_kemmeren_cc, file_zev_chip, file_hu_chip, file_kemmeren_chip, fig_filename, step=5, bin=60):
	xpts_zev_cc, rates_zev_cc = cal_support_rates(file_zev_cc, step, bin, True) if os.path.isfile(file_zev_cc) else (range(bin), None)
	xpts_hu_cc, rates_hu_cc = cal_support_rates(file_hu_cc, step, bin, True) if os.path.isfile(file_hu_cc) else (range(bin), None)
	xpts_kemmeren_cc, rates_kemmeren_cc = cal_support_rates(file_kemmeren_cc, step, bin, True) if os.path.isfile(file_kemmeren_cc) else (range(bin), None)
	xpts_zev_chip, rates_zev_chip = cal_support_rates(file_zev_chip, step, bin, True) if os.path.isfile(file_zev_chip) else (range(bin), None)
	xpts_hu_chip, rates_hu_chip = cal_support_rates(file_hu_chip, step, bin, True) if os.path.isfile(file_hu_chip) else (range(bin), None)
	xpts_kemmeren_chip, rates_kemmeren_chip = cal_support_rates(file_kemmeren_chip, step, bin, True) if os.path.isfile(file_kemmeren_chip) else (range(bin), None)

	##print precision at rank cutoff
	rank_cutoff = 50
	# print "%d\t%d\t%d\t%d\t%d\t%d" % (
	# 		xpts_zev_cc[xpts_zev_cc < rank_cutoff][-1] if rates_zev_cc is not None else 0, 
	# 		xpts_kemmeren_cc[xpts_kemmeren_cc < rank_cutoff][-1] if rates_kemmeren_cc is not None else 0, 
	# 		xpts_hu_cc[xpts_hu_cc < rank_cutoff][-1] if rates_hu_cc is not None else 0, 
	# 		xpts_zev_chip[xpts_zev_chip < rank_cutoff][-1] if rates_zev_chip is not None else 0,
	# 		xpts_kemmeren_chip[xpts_kemmeren_chip < rank_cutoff][-1] if rates_kemmeren_chip is not None else 0,
	# 		xpts_hu_chip[xpts_hu_chip < rank_cutoff][-1] if rates_hu_chip is not None else 0) 
	print "%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f" % ( 
			rates_zev_cc[xpts_zev_cc < rank_cutoff][-1] if rates_zev_cc is not None else 0, 
			rates_kemmeren_cc[xpts_kemmeren_cc < rank_cutoff][-1] if rates_kemmeren_cc is not None else 0,
			rates_hu_cc[xpts_hu_cc < rank_cutoff][-1] if rates_hu_cc is not None else 0,  
			rates_zev_chip[xpts_zev_chip < rank_cutoff][-1] if rates_zev_chip is not None else 0, 
			rates_kemmeren_chip[xpts_kemmeren_chip < rank_cutoff][-1] if rates_kemmeren_chip is not None else 0,
			rates_hu_chip[xpts_hu_chip < rank_cutoff][-1] if rates_hu_chip is not None else 0)

	## plot precision rank curve
	fig = plt.figure(num=None, figsize=(4,4), dpi=300)
	if rates_zev_cc is not None:
		plt.plot(xpts_zev_cc, rates_zev_cc, color="#ef3e5b", label="CC vs ZEV")
		xpts_diff = xpts_zev_cc[0]
		plt.plot(range(xpts_diff), [rates_zev_cc[0]]*xpts_diff, color="#ef3e5b")
	if rates_hu_cc is not None:
		plt.plot(xpts_hu_cc, rates_hu_cc, color="#95d47a", label="CC vs Hu")
		xpts_diff = xpts_hu_cc[0]
		plt.plot(range(xpts_diff), [rates_hu_cc[0]]*xpts_diff, color="#95d47a")
	if rates_kemmeren_cc is not None:
		plt.plot(xpts_kemmeren_cc, rates_kemmeren_cc, color="#6f5495", label="CC vs Kemmeren")
		xpts_diff = xpts_kemmeren_cc[0]
		plt.plot(range(xpts_diff), [rates_kemmeren_cc[0]]*xpts_diff, color="#6f5495")
	if rates_zev_chip is not None:
		plt.plot(xpts_zev_chip, rates_zev_chip, color="#ef3e5b", linestyle="--", label="ChIP vs ZEV")
		xpts_diff = xpts_zev_chip[0]
		plt.plot(range(xpts_diff), [rates_zev_chip[0]]*xpts_diff, color="#ef3e5b")
	if rates_hu_chip is not None:
		plt.plot(xpts_hu_chip, rates_hu_chip, color="#95d47a", linestyle="--", label="ChIP vs Hu")
		xpts_diff = xpts_hu_chip[0]
		plt.plot(range(xpts_diff), [rates_hu_chip[0]]*xpts_diff, color="#95d47a")
	if rates_kemmeren_chip is not None:
		plt.plot(xpts_kemmeren_chip, rates_kemmeren_chip, color="#6f5495", linestyle="--", label="ChIP vs Kemmeren")
		xpts_diff = xpts_kemmeren_chip[0]
		plt.plot(range(xpts_diff), [rates_kemmeren_chip[0]]*xpts_diff, color="#6f5495")
	random = 2.7 if 'leu3' in fig_filename else 5
	plt.plot(np.arange(step*bin), random*np.ones(step*bin), color="#777777", linestyle=":", label="Random")
	plt.xlabel("Ranking by binding signal", fontsize=14); plt.ylabel("% responsive", fontsize=14)
	space = 25
	plt.xticks(np.arange(space-1,step*bin,space), 
				np.arange(space,step*(bin+1),space), rotation=60)
	plt.ylim([0,100])
	plt.legend(loc="best", frameon=True)
	plt.tight_layout()
	plt.savefig(fig_filename, fmt="pdf")
	plt.close()



def plot_support_rate(data, header, fig_filename, step=5, bin=60):
	colors = ['r','g','b','y','m','c','k']
	fig = plt.figure(num=None, figsize=(4,4), dpi=300)
	for i in range(1,len(header)):
		data_sorted = np.array(sorted(data[:,[0,i]], key=lambda x: (x[1],x[0])))[::-1]
		xpts, rates = cal_support_rates(data_sorted, step, bin, True)
		plt.plot(xpts, rates, color=colors[i-1], label=header[i])
	random = 2.7 if 'Leu3' in fig_filename else 5
	plt.plot(np.arange(step*bin), random*np.ones(step*bin), color="#777777", linestyle=":", label="Random")
	plt.xlabel("Ranking by binding signal", fontsize=14); plt.ylabel("% responsive", fontsize=14)
	space = 25
	plt.xticks(np.arange(space-1,step*bin,space), 
				np.arange(space,step*(bin+1),space), rotation=60)
	plt.ylim([0,100])
	plt.legend(loc="best", frameon=True)
	plt.tight_layout()
	plt.savefig(fig_filename, fmt="pdf")
	plt.close()


## main
# tf_names = {'YLR451W':'Leu3'}
tf_names = {'YLR451W':'Leu3',
			'YDR034C':'Lys14',
			'YKL038W':'Rgt1',
			'YOL067C':'Rtg1',
			'YHL020C':'Opi1',
			'YFR034C':'Pho4',
			'YLR403W':'Sfp1',
			'YJL056C':'Zap1'} ## excluded 'YJR060W':'Cbf1'
for sys_name, common_name in tf_names.iteritems():
	print '... working on %s\t' % common_name
	file_cc = "CCProcess_16TFs/" + sys_name + ".cc_single_feature_matrix.txt"
	file_zev = "McIsaac_ZEV_DE/" + sys_name + "-15min.DE.txt"
	data = combine_data(file_cc, file_zev)
	header = ['label', 'tph_total', 'rph_total', 'logrph_total', 
				'tph_bs_total', 'rph_bs_total', 'logrph_bs_total', '-log_p']
	fig_filename = "analysis_single_cc_feature/precision_ranking."+ common_name +".pdf"
	plot_support_rate(data, header, fig_filename)



