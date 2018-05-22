#!/usr/bin/python
import sys
import argparse
import numpy as np
import pandas as pd
import glob
import os.path
import operator
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


def parse_args(argv):
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--cc", help="Calling Cards data filename")
    parser.add_argument("--chip", help="ChIP data filename")
    parser.add_argument("--zev", help="ZEV DE data filename")
    parser.add_argument("--kemmeren", help="Kemmeren Holstege DE data filename")
    parser.add_argument("--hu", help="Hu Iyer DE data filename")
    parser.add_argument("-o", "--output_filename")
    parsed = parser.parse_args(argv[1:])
    return parsed


def load_data(file):
	data = np.loadtxt(file)
	# data_sorted = data[np.argsort(data[:,1])[::-1],]
	data_sorted = np.array(sorted(data, key=lambda x: (x[1],x[0])))[::-1]
	return data_sorted


def combine_data(file_bound, file_expr, tie_breaker_col=None):
	if (file_bound is None) or (file_expr is None):
		return None

	bound = np.loadtxt(file_bound, dtype=str)
	expr = np.loadtxt(file_expr, dtype=str, usecols=[0,2])

	expr_de = np.abs(np.array(expr[:,1], dtype=float))
	five_percentile = np.sort(expr_de)[::-1][int(np.floor(len(expr)*.05))]
	if five_percentile == 0:
		indx_pos = np.where(expr_de != 0)[0]	
	else:
		indx_pos = np.where(expr_de >= five_percentile)[0]
	expr[:,1] = -1
	expr[indx_pos,1] = 1
	orfs = np.intersect1d(bound[:,0], expr[:,0])
	data = []
	for orf in orfs:
		indx_bound = np.where(bound[:,0] == orf)[0][0]
		indx_expr = np.where(expr[:,0] == orf)[0][0]
		row = [expr[indx_expr,1]] + list(bound[indx_bound,1:])
		## only get DE label and binding strength + tie breaker if avail
		if tie_breaker_col:
			row = [row[0], row[-1], row[tie_breaker_col]]
		else:
			row = [row[0], row[-1]] 
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


def plot_support_rate(data_dict, cmp_group, fig_filename, min_rank=300, set_tie_rank=False, step=5):
	bin = min_rank/step
	line_colors = ["#ef3e5b", "#95d47a", "#6f5495", "#ef3e5b", "#95d47a", "#6f5495"]
	fig = plt.figure(num=None, figsize=(4,4), dpi=300)
	for i in range(len(cmp_group)):
		group = cmp_group[i]
		## sort with or without tiebreaker
		if data_dict[group].shape[1] == 2:
			data_sorted = np.array(sorted(data_dict[group], key=lambda x: (x[1],x[0])))[::-1]
		elif data_dict[group].shape[1] == 3:
			data_sorted = np.array(sorted(data_dict[group], key=operator.itemgetter(1, 2))[::-1])
			data_sorted[:,1] += data_sorted[:,2]*(10**(-10))
		## calculate points to plot
		xpts, rates = cal_support_rates(data_sorted, step, bin, set_tie_rank)
		linestyle = "-" if i < 3 else "--"
		plt.plot(xpts, rates, color=line_colors[i], linestyle=linestyle, label=cmp_group[i])
	random = 2.7 if 'Leu3' in fig_filename else 5
	plt.plot(np.arange(min_rank), random*np.ones(min_rank), color="#777777", linestyle=':', label="Random")
	plt.xlabel("Ranking by binding signal", fontsize=14)
	plt.ylabel("% responsive", fontsize=14)
	space = 25
	plt.xticks(np.arange(space-1,min_rank,space), 
				np.arange(space,step*(bin+1),space), rotation=60)
	plt.xlim([0,min_rank])
	plt.ylim([0,100])
	plt.legend(loc="best", frameon=True)
	plt.tight_layout()
	plt.savefig(fig_filename, fmt="pdf")
	plt.close()


def main(argv):
	parsed = parse_args(argv)
	cmp_group = ["CC vs ZEV", "CC vs Hu", "CC vs Kemmeren",
				"ChIP vs ZEV", "ChIP vs Hu", "ChIP vs Kemmeren"]
	data_dict = {}

	tbc = 4

	data_dict["CC vs ZEV"] = combine_data(parsed.cc, parsed.zev, tie_breaker_col=tbc)
	data_dict["CC vs Hu"] = combine_data(parsed.cc, parsed.hu, tie_breaker_col=tbc)
	data_dict["CC vs Kemmeren"] = combine_data(parsed.cc, parsed.kemmeren, tie_breaker_col=tbc)
	data_dict["ChIP vs ZEV"] = combine_data(parsed.chip, parsed.zev)
	data_dict["ChIP vs Hu"] = combine_data(parsed.chip, parsed.hu)
	data_dict["ChIP vs Kemmeren"] = combine_data(parsed.chip, parsed.kemmeren)
	
	plot_support_rate(data_dict, cmp_group, parsed.output_filename, set_tie_rank=True)




if __name__ == "__main__":
	main(sys.argv)

