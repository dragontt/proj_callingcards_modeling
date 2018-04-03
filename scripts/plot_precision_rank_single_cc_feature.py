#!/usr/bin/python
import sys
import numpy as np
import pandas as pd
import glob
import os.path
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


global color_theme
color_theme = {"black":(0,0,0), 
                "blue":(31, 119, 180), "blue_L":(174, 199, 232), 
                "orange":(255, 127, 14), "orange_L":(255, 187, 120),
                "green":(44, 160, 44), "green_L":(152, 223, 138), 
                "red":(214, 39, 40), "red_L":(255, 152, 150),
                "magenta":(148, 103, 189), "magenta_L":(197, 176, 213),
                "brown":(140, 86, 75), "brown_L":(196, 156, 148),
                "pink":(227, 119, 194), "pink_L":(247, 182, 210), 
                "grey":(127, 127, 127), "grey_L":(199, 199, 199),
                "yellow":(255, 215, 0), "yellow_L":(219, 219, 141), 
                "cyan":(23, 190, 207), "cyan_L":(158, 218, 229)}
for c in color_theme.keys():
    (r, g, b) = color_theme[c]
    color_theme[c] = (r/255., g/255., b/255.)


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
		data = data[data[:,1] >= data[step*bin,1] ,]
		xpts, supp_rates = [], []
		rankings = rankdata(-data[:,1], method='max')
		for r in np.unique(rankings):
			i = np.where(rankings == r)[0][-1]
			x = data[:(i+1),0]
			xpts.append(i)
			supp_rates.append(float(len(x[x==1]))/len(x))
	else:
		xpts = np.arange(1,bin+1)*step
		supp_rates = []
		for i in range(1,bin+1):
			x = data[data[:,1] >= data[i*step,1] ,0]
			r = float(len(x[x==1])) / len(x)
			supp_rates.append(r)
	supp_rates = [r*100 for r in supp_rates]
	return (np.array(xpts), np.array(supp_rates))


def cal_support_rates2(data, header, top_target):
	support_rates = []
	for i in range(1,len(header)):
		data2 = np.array(sorted(data[:,[0,i]], key=lambda k: (k[1],k[0])))[::-1]
		x = data2[data2[:,1] >= data2[top_target,1] ,0]
		r = float(len(x[x==1])) / len(x)
		support_rates.append(r)
	return support_rates


def cal_auprc(file):
	data = np.loadtxt(file)
	return average_precision_score(data[:,0], data[:,1])


def plot_support_rate(data, header, fig_filename, min_rank=300, set_tie_rank=False, step=5):
	bin = min_rank/step
	line_colors = [color_theme['blue'], color_theme['orange'], 
					color_theme['green'], color_theme['blue_L'], 
					color_theme['orange_L'], color_theme['green_L'],
					color_theme['black']]
	fig = plt.figure(num=None, figsize=(4,4), dpi=300)
	for i in range(1,len(header)):
		data_sorted = np.array(sorted(data[:,[0,i]], key=lambda x: (x[1],x[0])))[::-1]
		xpts, rates = cal_support_rates(data_sorted, step, bin, set_tie_rank)
		plt.plot(xpts, rates, color=line_colors[i-1], label=header[i])
	random = 2.7 if 'Leu3' in fig_filename else 5
	plt.plot(np.arange(min_rank), random*np.ones(min_rank), color="#777777", linestyle=':', label="Random")
	plt.xlabel("Ranking by binding signal", fontsize=14)
	plt.ylabel("% responsive", fontsize=14)
	space = 25
	plt.xticks(np.arange(space-1,min_rank,space), 
				np.arange(space,step*(bin+1),space), rotation=60)
	plt.xlim([0,min_rank])
	# plt.ylim([0,100])
	plt.legend(loc="best", frameon=True)
	plt.tight_layout()
	plt.savefig(fig_filename, fmt="pdf")
	plt.close()


def calculate_stats(support_rates_dict, top_targets, detailed_header):
	for top_target in top_targets:
		output = pd.DataFrame(support_rates_dict[top_target], index=detailed_header)
		num_predictors, num_tfs = output.shape
		rank_mtx = np.zeros((num_predictors, num_tfs))
		for i in range(num_tfs):
			rank_mtx[:,i] = num_predictors+1 - rankdata(output.iloc[:,i], method='max')
		avg_rank = np.mean(rank_mtx, axis=1)
		med_rank = np.median(rank_mtx, axis=1)
		output = pd.concat([output, pd.DataFrame({'Average rank': avg_rank, 'Median rank': med_rank}, index=detailed_header)], axis=1)
		support_rates_dict[top_target]['output'] = output
	return support_rates_dict


def save_excel(support_rates_dict, filename):
	writer = pd.ExcelWriter(filename, engine='xlsxwriter')
	for top_target in sorted(support_rates_dict.keys()):
		sheet = 'Top%d' % top_target
		support_rates_dict[top_target]['output'].to_excel(writer, sheet_name=sheet)
	writer.save()


def main():
	# ## excluded 'YJR060W':'Cbf1'
	# # tf_names = {'YLR451W':'Leu3'}
	# tf_names = {'YOR032C':'Hms1',
	# 			'YLR451W':'Leu3',
	# 			'YDR034C':'Lys14',
	# 			'YKL038W':'Rgt1',
	# 			'YOL067C':'Rtg1',
	# 			'YHL020C':'Opi1',
	# 			'YFR034C':'Pho4',
	# 			'YLR403W':'Sfp1',
	# 			'YJL056C':'Zap1'} 

	# header = ['label', 'tph_total', 'rph_total', 'logrph_total', 
	# 			'tph_bs_total', 'rph_bs_total', 'logrph_bs_total', '-log_p']
	# top_targets = [10, 25, 50, 100, 300]
	# support_rates_dict = {}

	# for sys_name, common_name in tf_names.iteritems():
	# 	print '... working on %s\t' % common_name
	# 	file_cc = "CCProcessed_16TFs/" + sys_name + ".cc_single_feature_matrix.txt"
	# 	file_zev = "McIsaac_ZEV_DE/" + sys_name + "-15min.DE.txt"
	# 	data = combine_data(file_cc, file_zev)
	# 	# fig_filename = "analysis_single_cc_feature/simple_ranking."+ common_name +".pdf"
	# 	# plot_support_rate(data, header, fig_filename, set_tie_rank=False)

	# 	## store specific support rates
	# 	for top_target in top_targets:
	# 		support_rate = cal_support_rates2(data, header, top_target)
	# 		if top_target not in support_rates_dict.keys():
	# 			support_rates_dict[top_target] = {}
	# 		support_rates_dict[top_target][common_name] = support_rate

	# ## add average of each predictor
	# detailed_header = ['Transpositions per 100k', 
	# 					'Reads per 100k', 
	# 					'Transpositions weighted by log(reads) per 100k', 
	# 					'Background subtracted transpositions per 100k', 
	# 					'Background subtracted reads per 100k', 
	# 					'Background subtracted transpositions weighted by log(reads) per 100k',
	# 					'Poisson score, -log(p)']
	# support_rates_dict = calculate_stats(support_rates_dict, top_targets, detailed_header)

	# filename = "analysis_single_cc_feature/single_predictor_comparison.xlsx"
	# save_excel(support_rates_dict, filename)


	tf_names = np.loadtxt("CCProcessed_16TFs/valid_CCxZEV_TFs.txt", dtype=str, usecols=[1])
	algos = ["simple", "rf_cv10", "rf_cv100", "gb_cv10", "gb_cv100", "lr_cv10", "lr_cv100"]
	detailed_header = ["Simple Ranking", "Random Forest CV10", "Random Forest CV100", "Gradient Boosting CV10", "Gradient Boosting CV100", "Logistic Regression CV10", "Logistic Regression CV100"]
	top_targets = [10, 25, 50, 100, 300]
	support_rates_dict = {}
	## loop thru TFs then thru ml algos
	for common_name in tf_names:
		print '... working on %s\t' % common_name
		for top_target in top_targets:
			support_rates = []
			for algo in algos:
				data = np.loadtxt("analysis_single_cc_feature/"+ algo +"."+ common_name +".txt")
				## store specific support rates
				support_rate = cal_support_rates2(data, ['label','pred'], top_target)
				support_rates += support_rate
				if top_target not in support_rates_dict.keys():
					support_rates_dict[top_target] = {}
			support_rates_dict[top_target][common_name] = support_rates
	## save output
	support_rates_dict = calculate_stats(support_rates_dict, top_targets, detailed_header)
	filename = "analysis_single_cc_feature/ml_predictor_comparison.xlsx"
	save_excel(support_rates_dict, filename)


if __name__ == "__main__":
	main()

