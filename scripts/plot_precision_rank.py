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


def cal_support_rates(file, step, bin, set_tie_rank=False):
	data = load_data(file)
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


def plot_support_rate(file_simple_cc, file_rf_cc, file_rf_cc_ca, fig_filename, step=5, bin=60):
	print "%.5f\t%.5f\t%.5f" % (cal_auprc(file_simple_cc), cal_auprc(file_rf_cc), cal_auprc(file_rf_cc_ca))
	xpt_simple_cc, rates_simple_cc = cal_support_rates(file_simple_cc, step, bin)
	xpt_rf_cc, rates_rf_cc = cal_support_rates(file_rf_cc, step, bin)
	xpt_rf_cc_ca, rates_rf_cc_ca = cal_support_rates(file_rf_cc_ca, step, bin)

	fig = plt.figure(num=None, figsize=(4,4), dpi=300)
	plt.plot(xpt_simple_cc, rates_simple_cc, color="#ef3e5b", label="Simple CC")
	plt.plot(xpt_rf_cc, rates_rf_cc, color="#95d47a", label="RF CC")
	plt.plot(xpt_rf_cc_ca, rates_rf_cc_ca, color="#6f5495", label="RF CC+CA")
	random = 2.7 if 'leu3' in fig_filename else 5
	plt.plot(np.arange(bin), random*np.ones(bin), color="#777777", linestyle=":", label="Random")
	plt.xlabel("Ranking", fontsize=14); plt.ylabel("% responsive", fontsize=14)
	space = 5
	plt.xticks(np.arange(space-1,bin+1,space), 
				np.arange(step*space,step*(bin+1),step*space), rotation=60)
	plt.ylim([0,100])
	plt.legend(loc="best", frameon=True)
	plt.tight_layout()
	plt.savefig(fig_filename, fmt="pdf")


def plot_support_rate1(file_simple_cc, file_rf_cc, file_rf_cc_ca, fig_filename, step=5, bin=60):
	print "%.5f\t%.5f\t%.5f" % (cal_auprc(file_simple_cc), cal_auprc(file_rf_cc), cal_auprc(file_rf_cc_ca))
	xpt_simple_cc, rates_simple_cc = cal_support_rates(file_simple_cc, step, bin, True)
	xpt_rf_cc, rates_rf_cc = cal_support_rates(file_rf_cc, step, bin, True)
	xpt_rf_cc_ca, rates_rf_cc_ca = cal_support_rates(file_rf_cc_ca, step, bin, True)

	fig = plt.figure(num=None, figsize=(4,4), dpi=300)
	plt.plot(xpt_simple_cc, rates_simple_cc, color="#ef3e5b", label="Simple CC")
	plt.plot(xpt_rf_cc, rates_rf_cc, color="#95d47a", label="RF CC")
	plt.plot(xpt_rf_cc_ca, rates_rf_cc_ca, color="#6f5495", label="RF CC+CA")
	random = 2.7 if 'leu3' in fig_filename else 5
	plt.plot(np.arange(step*bin), random*np.ones(step*bin), color="#777777", linestyle=":", label="Random")
	plt.xlabel("Ranking", fontsize=14); plt.ylabel("% responsive", fontsize=14)
	space = 25
	plt.xticks(np.arange(space-1,step*bin,space), 
				np.arange(space,step*(bin+1),space), rotation=60)
	plt.ylim([0,100])
	plt.legend(loc="best", frameon=True)
	plt.tight_layout()
	plt.savefig(fig_filename, fmt="pdf")


def plot_support_rate2(file_zev_cc, file_hu_cc, file_kemmeren_cc, file_zev_chip, file_hu_chip, file_kemmeren_chip, fig_filename, step=5, bin=60):
	_, rates_zev_cc = cal_support_rates(file_zev_cc, step, bin) if os.path.isfile(file_zev_cc) else (range(bin), None)
	_, rates_hu_cc = cal_support_rates(file_hu_cc, step, bin) if os.path.isfile(file_hu_cc) else (range(bin), None)
	_, rates_kemmeren_cc = cal_support_rates(file_kemmeren_cc, step, bin) if os.path.isfile(file_kemmeren_cc) else (range(bin), None)
	_, rates_zev_chip = cal_support_rates(file_zev_chip, step, bin) if os.path.isfile(file_zev_chip) else (range(bin), None)
	_, rates_hu_chip = cal_support_rates(file_hu_chip, step, bin) if os.path.isfile(file_hu_chip) else (range(bin), None)
	_, rates_kemmeren_chip = cal_support_rates(file_kemmeren_chip, step, bin) if os.path.isfile(file_kemmeren_chip) else (range(bin), None)

	##print precision at rank 50
	print "%.3f\t%.3f\t%.3f\n%.3f\t%.3f\t%.3f\n" % (
			rates_hu_cc[5] if rates_hu_cc else 0, 
			rates_kemmeren_cc[5] if rates_kemmeren_cc else 0, 
			rates_zev_cc[5] if rates_zev_cc else 0, 
			rates_hu_chip[5] if rates_hu_chip else 0, 
			rates_kemmeren_chip[5] if rates_kemmeren_chip else 0, 
			rates_zev_chip[5] if rates_zev_chip else 0)


	## plot precision rank curve
	fig = plt.figure(num=None, figsize=(4,4), dpi=300)
	if rates_zev_cc:
		plt.plot(np.arange(bin), rates_zev_cc, color="#ef3e5b", label="Calling cards vs ZEV")
	if rates_hu_cc:
		plt.plot(np.arange(bin), rates_hu_cc, color="#95d47a", label="Calling cards vs Hu")
	if rates_kemmeren_cc:
		plt.plot(np.arange(bin), rates_kemmeren_cc, color="#6f5495", label="Calling cards vs Kemmeren")
	# if rates_zev_chip:
	# 	plt.plot(np.arange(bin), rates_zev_chip, color="#ef3e5b", linestyle="--", label="CC ChIP")
	# if rates_hu_chip:
	# 	plt.plot(np.arange(bin), rates_hu_chip, color="#95d47a", linestyle="--", label="Hu ChIP")
	# if rates_kemmeren_chip:
	# 	plt.plot(np.arange(bin), rates_kemmeren_chip, color="#6f5495", linestyle="--", label="Kemmeren ChIP")
	random = 2.7 if 'leu3' in fig_filename else 5
	plt.plot(np.arange(bin), random*np.ones(bin), color="#777777", linestyle=":", label="Random")
	plt.xlabel("Ranking", fontsize=14); plt.ylabel("% responsive", fontsize=14)
	space = 5
	plt.xticks(np.arange(space-1,bin+1,space), 
				np.arange(step*space,step*(bin+1),step*space), rotation=60)
	plt.legend(loc="best", frameon=True)
	plt.tight_layout()
	plt.savefig(fig_filename, fmt="pdf")



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


"""
tf_names = {'YJR060W':'Cbf1',
			'YLR451W':'Leu3',
			'YDR034C':'Lys14',
			'YKL038W':'Rgt1'}

for sys_name, common_name in tf_names.iteritems():
	print sys_name, common_name
	file_simple_cc = "../output/tmp.ZEV-15min_x_5TFs.simple.CC."+ sys_name +".txt"
	file_rf_cc = "../output/tmp.ZEV-15min_x_5TFs.RF.CC."+ sys_name +".txt"
	file_rf_cc_ca = "../output/tmp.ZEV-15min_x_5TFs.RF.CC_CA."+ sys_name +".txt"
	fig_filename = "../output/fig_ranking_comparison."+ common_name +".pdf"
	plot_support_rate1(file_simple_cc, file_rf_cc, file_rf_cc_ca, fig_filename)
"""

# """
tf_names = {'YLR451W':'Leu3',
			'YDR034C':'Lys14',
			'YKL038W':'Rgt1',
			'YOL067C':'Rtg1',
			'YHL020C':'Opi1',
			'YFR034C':'Pho4',
			'YLR403W':'Sfp1',
			'YJL056C':'Zap1'} ## excluded 'YJR060W':'Cbf1'
for sys_name, common_name in tf_names.iteritems():
	sys.stdout.write('%s\t' % common_name)
	# file_zev_cc = "../output/tmp.ZEV-15min_x_5TFs.simple.CC."+ sys_name +".txt"
	# file_hu_cc = "../output/tmp.Hu_x_5TFs.simple.CC."+ sys_name +".DE.tsv.txt"
	# file_kemmeren_cc = "../output/tmp.Kemmeren_x_5TFs.simple.CC."+ sys_name +".DE.tsv.txt"
	# file_zev_chip = "../output/tmp.ZEV-15min_x_5TFs.simple.ChIP."+ sys_name +".txt"
	# file_hu_chip = "../output/tmp.Hu_x_5TFs.simple.ChIP."+ sys_name +".DE.tsv.txt"
	# file_kemmeren_chip = "../output/tmp.Kemmeren_x_5TFs.simple.ChIP."+ sys_name +".DE.tsv.txt"

	file_zev_cc = "../output4/tmp.iterRF_finalTest_CCHM_cvrank/tmp.CC_v_ZEV.RF."+ sys_name +".txt"
	file_hu_cc = "../output4/tmp.CC_v_Hu.RF."+ sys_name +".txt"
	file_kemmeren_cc = "../output4/tmp.CC_v_Hols.RF."+ sys_name +".txt"
	file_zev_chip = "../output/tmp.ZEV-15min_x_5TFs.simple.ChIP."+ sys_name +".txt"
	file_hu_chip = "../output/tmp.Hu_x_5TFs.simple.ChIP."+ sys_name +".DE.tsv.txt"
	file_kemmeren_chip = "../output/tmp.Kemmeren_x_5TFs.simple.ChIP."+ sys_name +".DE.tsv.txt"

	fig_filename = "../output/fig_ranking."+ common_name +".pdf"
	plot_support_rate3(file_zev_cc, file_hu_cc, file_kemmeren_cc, file_zev_chip, file_hu_chip, file_kemmeren_chip, fig_filename)
# """
"""
tfs_zev_cc = [f.split('.')[-2] for f in glob.glob("../output5/precision_rank_simple/tmp.CC_vs_ZEV.*.txt")]
tfs_hu_cc = [f.split('.')[-2] for f in glob.glob("../output5/precision_rank_simple/tmp.CC_vs_Hu.*.txt")]
tfs_kemmeren_cc = [f.split('.')[-2] for f in glob.glob("../output5/precision_rank_simple/tmp.CC_vs_Hols.*.txt")]
tfs_zev_chip = [f.split('.')[-2] for f in glob.glob("../output5/precision_rank_simple/tmp.ChIP_v_ZEV.*.txt")]
tfs_hu_chip = [f.split('.')[-2] for f in glob.glob("../output5/precision_rank_simple/tmp.ChIP_v_Hu.*.txt")]
tfs_kemmeren_chip = [f.split('.')[-2] for f in glob.glob("../output5/precision_rank_simple/tmp.ChIP_v_Hols.*.txt")]
tfs_common = sorted(list(set(tfs_zev_cc) & set(tfs_hu_cc) & set(tfs_kemmeren_cc) & set(tfs_zev_chip) & set(tfs_hu_chip) & set(tfs_kemmeren_chip)))
for tf in tfs_common:
	sys.stdout.write('%s\t' % tf)
	file_zev_cc = "../output5/precision_rank_simple/tmp.CC_vs_ZEV."+ tf +".txt"
	file_hu_cc = "../output5/precision_rank_simple/tmp.CC_vs_Hu."+ tf +".txt"
	file_kemmeren_cc = "../output5/precision_rank_simple/tmp.CC_vs_Hols."+ tf +".txt"
	file_zev_chip = "../output5/precision_rank_simple/tmp.ChIP_v_ZEV."+ tf +".txt"
	file_hu_chip = "../output5/precision_rank_simple/tmp.ChIP_v_Hu."+ tf +".txt"
	file_kemmeren_chip = "../output5/precision_rank_simple/tmp.ChIP_v_Hols."+ tf +".txt"
	fig_filename = "../output/fig_ranking."+ tf +".pdf"
	plot_support_rate3(file_zev_cc, file_hu_cc, file_kemmeren_cc, file_zev_chip, file_hu_chip, file_kemmeren_chip, fig_filename)
"""


"""
## plot directional DE 
sys_name, common_name, timepoint = 'YJL056C', 'Zap1', '10min'
file_zev_cc = "../output/tmp.major_DE_direction.ZEV-"+timepoint+"_x_5TFs.simple.CC."+ sys_name +".txt"
file_hu_cc = "../output/tmp.major_DE_direction.Hu_x_5TFs.simple.CC."+ sys_name +".DE.tsv.txt"
file_kemmeren_cc = "../output/tmp.major_DE_direction.Kemmeren_x_5TFs.simple.CC."+ sys_name +".DE.tsv.txt"
file_zev_chip = "../output/tmp.major_DE_direction.ZEV-"+timepoint+"_x_5TFs.simple.ChIP."+ sys_name +".txt"
file_hu_chip = "../output/tmp.major_DE_direction.Hu_x_5TFs.simple.ChIP."+ sys_name +".DE.tsv.txt"
file_kemmeren_chip = "../output/tmp.major_DE_direction.Kemmeren_x_5TFs.simple.ChIP."+ sys_name +".DE.tsv.txt"
fig_filename = "../output/fig_ranking.major_DE_direction."+ common_name +".ZEV_"+timepoint+".pdf"
# fig_filename = "../output/fig_ranking.minor_DE_direction."+ common_name +".ZEV_"+timepoint+".pdf"
plot_support_rate3(file_zev_cc, file_hu_cc, file_kemmeren_cc, file_zev_chip, file_hu_chip, file_kemmeren_chip, fig_filename)
"""

def plot_support_rate4(file_tf1, file_tf2, file_tfs, label1, label2, label3, fig_filename, step=20, bin=60):
	print "%.5f\t%.5f\t%.5f" % (cal_auprc(file_tf1), cal_auprc(file_tf2), cal_auprc(file_tfs))

	xpts_tf1, rates_tf1 = cal_support_rates(file_tf1, step, bin)
	xpts_tf2, rates_tf2 = cal_support_rates(file_tf2, step, bin)
	xpts_tfs, rates_tfs = cal_support_rates(file_tfs, step, bin)

	fig = plt.figure(num=None, figsize=(4.5,4), dpi=300)
	plt.plot(xpts_tf1, rates_tfs, color="#ef3e5b", label=label3)
	plt.plot(xpts_tf2, rates_tf2, color="#95d47a", label=label2)
	plt.plot(xpts_tfs, rates_tf1, color="#6f5495", label=label1)
	plt.plot(np.arange(bin), 5*np.ones(bin), color="#777777", linestyle=":", label="Random")
	plt.xlabel("Ranking by random forest model", fontsize=14); plt.ylabel("% responsive to "+label1+" induction", fontsize=14)
	space = 5
	plt.xticks(np.arange(space-1,bin+1,space), 
				np.arange(step*space,step*(bin+1),step*space), rotation=60)
	plt.ylim([0,25])
	plt.legend(loc="best", frameon=True)
	plt.tight_layout()
	plt.savefig(fig_filename, fmt="pdf")

"""
file_Cbf1 = "../output/tmp.2_interactive_TFs.CC.YJR060W.txt"
file_Pho4_on_Cbf1 = "../output/tmp.2_interactive_TFs.CC.YFR034C_only_on_predict_YJR060W.txt" 
file_Cbf1_Pho4 = "../output/tmp.2_interactive_TFs.CC.YJR060W_YFR034C.txt"
fig_filename = "../output/fig_ranking.interactive_Cbf1_Pho4.pdf"
plot_support_rate4(file_Cbf1, file_Pho4_on_Cbf1, file_Cbf1_Pho4, "Cbf1", "Pho4", "Cbf1 and Pho4", fig_filename)

file_Pho4 = "../output/tmp.2_interactive_TFs.CC.YFR034C.txt"
file_Cbf1_on_Pho4 = "../output/tmp.2_interactive_TFs.CC.YJR060W_only_on_predict_YFR034C.txt" 
file_Pho4_Cbf1 = "../output/tmp.2_interactive_TFs.CC.YFR034C_YJR060W.txt"
fig_filename = "../output/fig_ranking.interactive_Pho4_Cbf1.pdf"
plot_support_rate4(file_Pho4, file_Cbf1_on_Pho4, file_Pho4_Cbf1, "Pho4", "Cbf1", "Pho4 and Cbf1", fig_filename)

file_Cbf1 = "../output/tmp.2_interactive_TFs.CC.YJR060W.txt"
file_Tye7_on_Cbf1 = "../output/tmp.2_interactive_TFs.CC.YOR344C_only_on_predict_YJR060W.txt" 
file_Cbf1_Tye7 = "../output/tmp.2_interactive_TFs.CC.YJR060W_YOR344C.txt"
fig_filename = "../output/fig_ranking.interactive_Cbf1_Tye7.pdf"
plot_support_rate4(file_Cbf1, file_Tye7_on_Cbf1, file_Cbf1_Tye7, "Cbf1", "Tye7", "Cbf1 and Tye7", fig_filename)
"""


