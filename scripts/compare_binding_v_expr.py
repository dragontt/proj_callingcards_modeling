#/usr/bin/python
import sys
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve, roc_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def load_cc(file, transform=False):
	cc = np.loadtxt(file, dtype=str, usecols=[0,7]) ## promoter p-val
	pval = np.array(cc[:,1], dtype=float) + 10**-25
	cc[:,1] = -1*np.log(pval)
	return cc


def load_chip(file):
	chip = np.loadtxt(file, dtype=str, usecols=[0,2])
	chip[(chip[:,1] == 'nan'),1] = 0
	return chip


def load_zev(file):
	return np.loadtxt(file, dtype=str, usecols=[0,2])


def load_microarray(file, use_lfc=False):
	if use_lfc:
		data = np.loadtxt(file, dtype=str, usecols=[0,2])
	else:
		data = np.loadtxt(file, dtype=str, usecols=[0,1])
		pval = np.array(data[:,1], dtype=float) + 10**-25
		data[:,1] = np.round(-1*np.log(pval), decimals=8)
	return data


def match_binding_de(file_binding, file_de, cutoff, type_binding, type_de, type_cutoff="percentage"):
	if type_binding.lower() == "cc":
		binding = load_cc(file_binding)
	elif type_binding.lower() == "chip":
		binding = load_chip(file_binding)
	if type_de.lower() == "zev":
		de = load_zev(file_de)
	elif type_de.lower() in ["microarray", "rnaseq"]:
		de = load_microarray(file_de)

	common_tgts = np.intersect1d(binding[:,0], de[:,0])
	indx_binding = [np.where(binding[:,0] == common_tgts[i])[0][0] for i in range(len(common_tgts))]
	indx_de = [np.where(de[:,0] == common_tgts[i])[0][0] for i in range(len(common_tgts))]

	data_binding = np.array(binding[indx_binding,1], dtype=float)
	data_de = np.abs(np.array(de[indx_de,1], dtype=float))

	if type_cutoff == "percentage":
		indx_sort = np.argsort(data_de)[::-1]
		top = int(cutoff*len(data_de))
		cnt_nonzero_tgts = len(data_de[data_de != 0])
		if top > cnt_nonzero_tgts:
			top = cnt_nonzero_tgts
			print "WARNING: Non-zero LFC targets <", cutoff, "x total genes!"
			## update for LEU3: keep postive percentage the same
			global de_cutoff
			de_cutoff = float(cnt_nonzero_tgts)/len(data_de)
			## update for LEU3: keep positive count the same
			# global de_cutoff
			# global type_de_cutoff
			# de_cutoff = float(cnt_nonzero_tgts) # float(cnt_nonzero_tgts)/len(data_de)
			# type_de_cutoff = "fixed_rank"
		data_de[:] = -1
		data_de[indx_sort[:top]] = 1
	elif type_cutoff == "fixed_rank":
		indx_sort = np.argsort(data_de)[::-1]
		top = int(de_cutoff)
		data_de[:] = -1
		data_de[indx_sort[:top]] = 1
	elif type_cutoff == "lfc_cutoff":
		data_de[data_de < cutoff] = -1
		data_de[data_de >= 0] = 1
	data_de = np.array(data_de, dtype=int)
	# print np.unique(data_de, return_counts=True)
	return (data_de, data_binding)


def calculate_auc(file_binding, file_de, cutoff, type_binding, type_de, type_cutoff="percentage"):
	if (not file_binding) or (not file_de):
		return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)

	data_de, data_binding = match_binding_de(file_binding, file_de, cutoff, type_binding, type_de, type_cutoff)

	# out = np.hstack((data_de.reshape(-1,1), data_binding.reshape(-1,1)))
	# filename = '../output/tmp.Kemmeren_x_5TFs.simple.ChIP.'+file_de.split('/')[2].split('-')[0]+'.txt'
	# np.savetxt(filename, out, fmt='%s', delimiter='\t')

	auprc = average_precision_score(data_de, data_binding) 
	auroc = roc_auc_score(data_de, data_binding)

	auprc_rnd = []
	auroc_rnd = []
	for i in range(200):
		rnd_binding = np.random.permutation(data_binding)
		auprc_rnd.append(average_precision_score(data_de, rnd_binding))
		auroc_rnd.append(roc_auc_score(data_de, rnd_binding))
	return (np.median(auprc_rnd), np.percentile(auprc_rnd, 97.5)-np.median(auprc_rnd), auprc, 
			np.median(auroc_rnd), np.percentile(auroc_rnd, 97.5)-np.median(auroc_rnd), auroc)


def batch_calculate_aucs(file_cc, file_chip, file_zev, file_rnaseq, file_kemmeren, file_hu):
	cc_zev = calculate_auc(file_cc, file_zev, de_cutoff, "CC", "ZEV", type_de_cutoff)
	cc_rnaseq = calculate_auc(file_cc, file_rnaseq, de_cutoff, "CC", "microarray", type_de_cutoff)
	cc_hu = calculate_auc(file_cc, file_hu, de_cutoff, "CC", "microarray", type_de_cutoff)
	cc_kemmeren = calculate_auc(file_cc, file_kemmeren, de_cutoff, "CC", "RNAseq", type_de_cutoff)
	
	chip_zev = calculate_auc(file_chip, file_zev, de_cutoff, "ChIP", "ZEV", type_de_cutoff)
	chip_rnaseq = calculate_auc(file_chip, file_rnaseq, de_cutoff, "ChIP", "RNAseq", type_de_cutoff)
	chip_hu = calculate_auc(file_chip, file_hu, de_cutoff, "ChIP", "microarray", type_de_cutoff)
	chip_kemmeren = calculate_auc(file_chip, file_kemmeren, de_cutoff, "ChIP", "microarray", type_de_cutoff)
	
	print "\tHu\tKemmeren\tBrent\tMcIssac"
	print "Calling Card\t%.3f\t%.3f\t%.3f\t%.3f" % (cc_hu[2], cc_kemmeren[2], cc_rnaseq[2], cc_zev[2])
	print "Harbison\t%.3f\t%.3f\t%.3f\t%.3f" % (chip_hu[2], chip_kemmeren[2], chip_rnaseq[2], chip_zev[2])
	print "PWM"
	print "Random, median\t%.3f\t%.3f\t%.3f\t%.3f" % (chip_hu[0], chip_kemmeren[0], chip_rnaseq[0], chip_zev[0])
	print "Random, 97.5%% upper bound\t%.3f\t%.3f\t%.3f\t%.3f" % (chip_hu[1], chip_kemmeren[1], chip_rnaseq[1], chip_zev[1])


def calculate_curve_data(file_binding, file_de, cutoff, type_binding, type_de, type_cutoff="percentage"):
	data_de, data_binding = match_binding_de(file_binding, file_de, cutoff, type_binding, type_de, type_cutoff)
	pr, re, _ = precision_recall_curve(data_de, data_binding)
	fp, tp, _ = roc_curve(data_de, data_binding)
	rnd_binding = np.random.permutation(data_binding)
	rnd_pr, rnd_re, _ = precision_recall_curve(data_de,rnd_binding)
	rnd_fp, rnd_tp, _ = roc_curve(data_de, rnd_binding)
	return ((pr,re), (fp,tp), (rnd_pr,rnd_re), (rnd_fp,rnd_tp))


def batch_plot_curve(file_cc, file_chip, file_zev, file_rnaseq, file_kemmeren, file_hu, figname):
	fig = plt.figure(num=None, figsize=(4.5,4), dpi=300)
	## CC vs ZEV
	print de_cutoff, type_de_cutoff
	((pr,re), (fp,tp), _, _) = calculate_curve_data(file_cc, file_zev, de_cutoff, "CC", "ZEV", type_de_cutoff)
	plt.step(re, pr, color="#ef3e5b", label="Calling cards vs ZEV")
	## CC vs Kemmeren
	print de_cutoff, type_de_cutoff
	((pr,re), (fp,tp), _, _) = calculate_curve_data(file_cc, file_kemmeren, de_cutoff, "CC", "microarray", type_de_cutoff)
	plt.step(re, pr, color="#95d47a", linestyle="-", label="Calling cards vs Kemmeren")
	## CC vs Hu
	print de_cutoff, type_de_cutoff
	((pr,re), (fp,tp), (rnd_pr,rnd_re), _) = calculate_curve_data(file_chip, file_hu, de_cutoff, "ChIP", "microarray", type_de_cutoff)
	plt.step(re, pr, color="#6f5495", linestyle="-", label="Calling cards vs Hu")
	## random 
	plt.step(rnd_re, rnd_pr, color="#c9c9c9", linestyle="-", label="Random")
	## figure attributes
	matplotlib.rcParams.update({'font.size': 12})
	matplotlib.rcParams.update({'font.family': 'DejaVu Sans'})
	plt.xlabel("Recall", fontsize=14); plt.ylabel("Precision", fontsize=14)
	plt.ylim([0.0, 1.01]); plt.xlim([0, 1.0])
	plt.grid(linewidth=.5)
	plt.legend(loc="best", frameon=True)
	plt.savefig(figname, format="pdf")


def calculate_direction_of_change(file, reverse=False, usecol=2, lfc_cutoff=1):
	if not file:
		return (np.nan, np.nan, np.nan, np.nan, np.nan)
	lfc = np.loadtxt(file, usecols=[usecol])
	pos_lfc = len(lfc[lfc > lfc_cutoff])
	neg_lfc = len(lfc[lfc < -1*lfc_cutoff])
	if reverse:
		return (float(neg_lfc)/(pos_lfc+neg_lfc), float(pos_lfc)/(pos_lfc+neg_lfc), 
				pos_lfc+neg_lfc, neg_lfc, pos_lfc)
	else:
		return (float(pos_lfc)/(pos_lfc+neg_lfc), float(neg_lfc)/(pos_lfc+neg_lfc), 
				pos_lfc+neg_lfc, pos_lfc, neg_lfc)


def batch_calculate_direction_of_change(tf, files_zev, file_rnaseq, file_kemmeren, file_hu):
	timepoint_dict = {0: 'early', 1: 'mid', 2: 'late'}
	results = np.empty((0,7))
	for i in range(len(files_zev)):
		zev = calculate_direction_of_change(files_zev[i])
		zev = ['ZEV-'+timepoint_dict[i], tf] + list(zev)
		results = np.vstack(( results, zev ))
	rnaseq = calculate_direction_of_change(file_rnaseq)
	rnaseq = ['RNAseq', tf] + list(rnaseq)
	results = np.vstack(( results, rnaseq ))
	kemmeren = calculate_direction_of_change(file_kemmeren, True)
	kemmeren = ['Kemmeren', tf] + list(kemmeren)
	results = np.vstack(( results, kemmeren ))
	hu = calculate_direction_of_change(file_hu, True)
	hu = ['Hu', tf] + list(hu)
	results = np.vstack(( results, hu ))
	for i in range(len(results)):
		print "%s\t%s\t%s\t%s\t%s\t%s\t%s" % tuple(results[i])


global de_cutoff
global type_de_cutoff
de_cutoff = 0.05 ## 5% of top genes as positive class
type_de_cutoff = "percentage"
# de_cutoff = .5 ## |LFC| > .5 as positive class

"""
timepoint = '15min'
print "\nRGT1"
file_cc = "../CCDataProcessed/NULL_model_results.RGT1-Tagin-Lys_filtered.gnashy"
file_chip = "../Harbison_ChIP/YKL038W.cc_feature_matrix.binned_promoter.txt"
file_zev = "../McIsaac_ZEV_DE/YKL038W-"+timepoint+".match_minusLys.DE.txt"
file_rnaseq = "../resources/YKL038W-minusLys.DE.tsv"
file_kemmeren = "../Holstege_DE/YKL038W.DE.tsv"
file_hu = "../Hu_DE/YKL038W.DE.tsv"
batch_calculate_aucs(file_cc, file_chip, file_zev, file_rnaseq, file_kemmeren, file_hu)

print "\nLYS14"
file_cc = "../CCDataProcessed/NULL_model_results.Lys14-Tagin-Lys_filtered.gnashy"
file_chip = None
file_zev = "../McIsaac_ZEV_DE/YDR034C-"+timepoint+".match_minusLys.DE.txt"
file_rnaseq = "../resources/YDR034C-minusLys.DE.tsv"
file_kemmeren = "../Holstege_DE/YDR034C.DE.tsv"
file_hu = None
batch_calculate_aucs(file_cc, file_chip, file_zev, file_rnaseq, file_kemmeren, file_hu)

print "\nRTG1"
file_cc = "../CCDataProcessed/NULL_model_results.RTG1_68A_filtered.gnashy"
file_chip = "../Harbison_ChIP/YOL067C.cc_feature_matrix.binned_promoter.txt"
file_zev = "../McIsaac_ZEV_DE/YOL067C-"+timepoint+".DE.txt"
file_rnaseq = None
file_kemmeren = "../Holstege_DE/YOL067C.DE.tsv"
file_hu	= "../Hu_DE/YOL067C.DE.tsv"
batch_calculate_aucs(file_cc, file_chip, file_zev, file_rnaseq, file_kemmeren, file_hu)

print "\nCBF1"
file_cc = "../CCDataProcessed/NULL_model_results.CBF1_FLALL.gnashy"
file_chip = "../Harbison_ChIP/YJR060W.cc_feature_matrix.binned_promoter.txt"
file_zev = "../McIsaac_ZEV_DE/YJR060W-90min.DE.txt"
file_rnaseq = None
file_kemmeren = "../Holstege_DE/YJR060W.DE.tsv"
file_hu	= "../Hu_DE/YJR060W.DE.tsv"
batch_calculate_aucs(file_cc, file_chip, file_zev, file_rnaseq, file_kemmeren, file_hu)

print "\nZAP1"
file_cc = "../CCDataProcessed/NULL_model_results.ZAP1_52A_filtered.gnashy"
file_chip = "../Harbison_ChIP/YJL056C.cc_feature_matrix.binned_promoter.txt"
file_zev = "../McIsaac_ZEV_DE/YJL056C-"+timepoint+".DE.txt"
file_rnaseq = None
file_kemmeren = "../Holstege_DE/YJL056C.DE.tsv"
file_hu	= "../Hu_DE/YJL056C.DE.tsv"
batch_calculate_aucs(file_cc, file_chip, file_zev, file_rnaseq, file_kemmeren, file_hu)

print "\nSFP1, CC-48A"
file_cc = "../CCDataProcessed/NULL_model_results.SFP1_48A_filtered.gnashy"
# file_cc = "../CCDataProcessed/NULL_model_results.SFP1_62A_filtered.gnashy"
file_chip = "../Harbison_ChIP/YLR403W.cc_feature_matrix.binned_promoter.txt"
file_zev = "../McIsaac_ZEV_DE/YLR403W-"+timepoint+".DE.txt"
file_rnaseq = None
file_kemmeren = "../Holstege_DE/YLR403W.DE.tsv"
file_hu	= "../Hu_DE/YLR403W.DE.tsv"
batch_calculate_aucs(file_cc, file_chip, file_zev, file_rnaseq, file_kemmeren, file_hu)

# print "\nSFP1, CC-62A"
# # file_cc = "../CCDataProcessed/NULL_model_results.SFP1_48A_filtered.gnashy"
# file_cc = "../CCDataProcessed/NULL_model_results.SFP1_62A_filtered.gnashy"
# file_chip = "../Harbison_ChIP/YLR403W.cc_feature_matrix.binned_promoter.txt"
# file_zev = "../McIsaac_ZEV_DE/YLR403W-"+timepoint+".DE.txt"
# file_rnaseq = None
# file_kemmeren = "../Holstege_DE/YLR403W.DE.tsv"
# file_hu	= "../Hu_DE/YLR403W.DE.tsv"
# batch_calculate_aucs(file_cc, file_chip, file_zev, file_rnaseq, file_kemmeren, file_hu)

print "\nOPI1"
file_cc = "../CCDataProcessed/NULL_model_results.OPI1_51A_filtered.gnashy"
file_chip = "../Harbison_ChIP/YHL020C.cc_feature_matrix.binned_promoter.txt"
file_zev = "../McIsaac_ZEV_DE/YHL020C-"+timepoint+".DE.txt"
file_rnaseq = None
file_kemmeren = "../Holstege_DE/YHL020C.DE.tsv"
file_hu	= "../Hu_DE/YHL020C.DE.tsv"
batch_calculate_aucs(file_cc, file_chip, file_zev, file_rnaseq, file_kemmeren, file_hu)

print "\nPHO4"
file_cc = "../CCDataProcessed/NULL_model_results.PHO4_71A_filtered.gnashy"
file_chip = "../Harbison_ChIP/YFR034C.cc_feature_matrix.binned_promoter.txt"
file_zev = "../McIsaac_ZEV_DE/YFR034C-"+timepoint+".DE.txt"
file_rnaseq = None
file_kemmeren = "../Holstege_DE/YFR034C.DE.tsv"
file_hu	= "../Hu_DE/YFR034C.DE.tsv"
batch_calculate_aucs(file_cc, file_chip, file_zev, file_rnaseq, file_kemmeren, file_hu)

print "\nLEU3"
file_cc = "../CCDataProcessed/NULL_model_results.Leu3-Tagin-Trp_filtered.gnashy"
file_chip = "../Harbison_ChIP/YLR451W.cc_feature_matrix.binned_promoter.txt"
file_zev = "../McIsaac_ZEV_DE/YLR451W-"+timepoint+".DE.txt"
file_rnaseq = None
file_kemmeren = "../Holstege_DE/YLR451W.DE.tsv"
file_hu	= "../Hu_DE/YLR451W.DE.tsv"
batch_calculate_aucs(file_cc, file_chip, file_zev, file_rnaseq, file_kemmeren, file_hu)
"""

"""
print "\nLEU3"
file_cc = "../CCDataProcessed/NULL_model_results.Leu3-Tagin-Trp_filtered.gnashy"
file_chip = "../Harbison_ChIP/YLR451W.cc_feature_matrix.binned_promoter.txt"
file_zev = "../McIsaac_ZEV_DE/YLR451W-15min.DE.txt"
file_rnaseq = None
file_kemmeren = "../Holstege_DE/YLR451W.DE.tsv"
file_hu	= "../Hu_DE/YLR451W.DE.tsv"
batch_calculate_aucs(file_cc, file_chip, file_zev, file_rnaseq, file_kemmeren, file_hu)
"""

"""
print "\nCBF1-5min"
file_cc = "../CCDataProcessed/NULL_model_results.CBF1_FLALL.gnashy"
file_chip = "../Harbison_ChIP/YJR060W.cc_feature_matrix.binned_promoter.txt"
file_zev = "../McIsaac_ZEV_DE/YJR060W-5min.DE.txt"
file_rnaseq = None
file_kemmeren = "../Holstege_DE/YJR060W.DE.tsv"
file_hu	= "../Hu_DE/YJR060W.DE.tsv"
batch_calculate_aucs(file_cc, file_chip, file_zev, file_rnaseq, file_kemmeren, file_hu)

print "\nCBF1-15min"
file_cc = "../CCDataProcessed/NULL_model_results.CBF1_FLALL.gnashy"
file_chip = "../Harbison_ChIP/YJR060W.cc_feature_matrix.binned_promoter.txt"
file_zev = "../McIsaac_ZEV_DE/YJR060W-15min.DE.txt"
file_rnaseq = None
file_kemmeren = "../Holstege_DE/YJR060W.DE.tsv"
file_hu	= "../Hu_DE/YJR060W.DE.tsv"
batch_calculate_aucs(file_cc, file_chip, file_zev, file_rnaseq, file_kemmeren, file_hu)

print "\nCBF1-30min"
file_cc = "../CCDataProcessed/NULL_model_results.CBF1_FLALL.gnashy"
file_chip = "../Harbison_ChIP/YJR060W.cc_feature_matrix.binned_promoter.txt"
file_zev = "../McIsaac_ZEV_DE/YJR060W-30min.DE.txt"
file_rnaseq = None
file_kemmeren = "../Holstege_DE/YJR060W.DE.tsv"
file_hu	= "../Hu_DE/YJR060W.DE.tsv"
batch_calculate_aucs(file_cc, file_chip, file_zev, file_rnaseq, file_kemmeren, file_hu)

print "\nCBF1-45min"
file_cc = "../CCDataProcessed/NULL_model_results.CBF1_FLALL.gnashy"
file_chip = "../Harbison_ChIP/YJR060W.cc_feature_matrix.binned_promoter.txt"
file_zev = "../McIsaac_ZEV_DE/YJR060W-45min.DE.txt"
file_rnaseq = None
file_kemmeren = "../Holstege_DE/YJR060W.DE.tsv"
file_hu	= "../Hu_DE/YJR060W.DE.tsv"
batch_calculate_aucs(file_cc, file_chip, file_zev, file_rnaseq, file_kemmeren, file_hu)

print "\nCBF1-60min"
file_cc = "../CCDataProcessed/NULL_model_results.CBF1_FLALL.gnashy"
file_chip = "../Harbison_ChIP/YJR060W.cc_feature_matrix.binned_promoter.txt"
file_zev = "../McIsaac_ZEV_DE/YJR060W-60min.DE.txt"
file_rnaseq = None
file_kemmeren = "../Holstege_DE/YJR060W.DE.tsv"
file_hu	= "../Hu_DE/YJR060W.DE.tsv"
batch_calculate_aucs(file_cc, file_chip, file_zev, file_rnaseq, file_kemmeren, file_hu)

print "\nCBF1-90min"
file_cc = "../CCDataProcessed/NULL_model_results.CBF1_FLALL.gnashy"
file_chip = "../Harbison_ChIP/YJR060W.cc_feature_matrix.binned_promoter.txt"
file_zev = "../McIsaac_ZEV_DE/YJR060W-90min.DE.txt"
file_rnaseq = None
file_kemmeren = "../Holstege_DE/YJR060W.DE.tsv"
file_hu	= "../Hu_DE/YJR060W.DE.tsv"
batch_calculate_aucs(file_cc, file_chip, file_zev, file_rnaseq, file_kemmeren, file_hu)
"""

"""
file_cc = "../CCDataProcessed/NULL_model_results.Leu3-Tagin-Trp_filtered.gnashy"
file_chip = "../Harbison_ChIP/YLR451W.cc_feature_matrix.binned_promoter.txt"
file_zev = "../McIsaac_ZEV_DE/YLR451W-15min.DE.txt"
file_rnaseq = "../resources/YLR451W.DE.tsv"
file_kemmeren = "../Holstege_DE/YLR451W.DE.tsv"
file_hu = "../Hu_DE/YLR451W.DE.tsv"
batch_plot_curve(file_cc, file_chip, file_zev, file_rnaseq, file_kemmeren, file_hu, "../output/LEU3.PR.pdf")

"""


# """
tf = "RGT1"
files_zev = ["../McIsaac_ZEV_DE/YKL038W-10min.match_minusLys.DE.txt",
			"../McIsaac_ZEV_DE/YKL038W-15min.match_minusLys.DE.txt",
			"../McIsaac_ZEV_DE/YKL038W-20min.match_minusLys.DE.txt"]
file_rnaseq = "../resources/YKL038W-minusLys.DE.tsv"
file_kemmeren = "../Holstege_DE/YKL038W.DE.tsv"
file_hu = "../Hu_DE/YKL038W.DE.tsv"
batch_calculate_direction_of_change(tf, files_zev, file_rnaseq, file_kemmeren, file_hu)

tf = "LYS14"
files_zev = ["../McIsaac_ZEV_DE/YDR034C-10min.match_minusLys.DE.txt",
			"../McIsaac_ZEV_DE/YDR034C-15min.match_minusLys.DE.txt",
			"../McIsaac_ZEV_DE/YDR034C-20min.match_minusLys.DE.txt"]
file_rnaseq = "../resources/YDR034C-minusLys.DE.tsv"
file_kemmeren = "../Holstege_DE/YDR034C.DE.tsv"
file_hu = None
batch_calculate_direction_of_change(tf, files_zev, file_rnaseq, file_kemmeren, file_hu)

tf = "RTG1"
files_zev = ["../McIsaac_ZEV_DE/YOL067C-10min.DE.txt",
			"../McIsaac_ZEV_DE/YOL067C-15min.DE.txt",
			"../McIsaac_ZEV_DE/YOL067C-20min.DE.txt"]
file_rnaseq = None
file_kemmeren = "../Holstege_DE/YOL067C.DE.tsv"
file_hu	= "../Hu_DE/YOL067C.DE.tsv"
batch_calculate_direction_of_change(tf, files_zev, file_rnaseq, file_kemmeren, file_hu)

tf = "CBF1"
files_zev = ["../McIsaac_ZEV_DE/YJR060W-5min.DE.txt",
			"../McIsaac_ZEV_DE/YJR060W-15min.DE.txt",
			"../McIsaac_ZEV_DE/YJR060W-30min.DE.txt"]
file_rnaseq = None
file_kemmeren = "../Holstege_DE/YJR060W.DE.tsv"
file_hu	= "../Hu_DE/YJR060W.DE.tsv"
batch_calculate_direction_of_change(tf, files_zev, file_rnaseq, file_kemmeren, file_hu)

tf = "LEU3"
files_zev = ["../McIsaac_ZEV_DE/YLR451W-10min.DE.txt",
			"../McIsaac_ZEV_DE/YLR451W-15min.DE.txt",
			"../McIsaac_ZEV_DE/YLR451W-20min.DE.txt"]
file_rnaseq = None
file_kemmeren = "../Holstege_DE/YLR451W.DE.tsv"
file_hu	= "../Hu_DE/YLR451W.DE.tsv"
batch_calculate_direction_of_change(tf, files_zev, file_rnaseq, file_kemmeren, file_hu)

tf = "OPI1"
files_zev = ["../McIsaac_ZEV_DE/YHL020C-10min.DE.txt",
			"../McIsaac_ZEV_DE/YHL020C-15min.DE.txt",
			"../McIsaac_ZEV_DE/YHL020C-20min.DE.txt"]
file_rnaseq = None
file_kemmeren = "../Holstege_DE/YHL020C.DE.tsv"
file_hu	= "../Hu_DE/YHL020C.DE.tsv"
batch_calculate_direction_of_change(tf, files_zev, file_rnaseq, file_kemmeren, file_hu)

tf = "PHO4"
files_zev = ["../McIsaac_ZEV_DE/YFR034C-10min.DE.txt",
			"../McIsaac_ZEV_DE/YFR034C-15min.DE.txt",
			"../McIsaac_ZEV_DE/YFR034C-20min.DE.txt"]
file_rnaseq = None
file_kemmeren = "../Holstege_DE/YFR034C.DE.tsv"
file_hu	= "../Hu_DE/YFR034C.DE.tsv"
batch_calculate_direction_of_change(tf, files_zev, file_rnaseq, file_kemmeren, file_hu)

tf = "SFP1"
files_zev = ["../McIsaac_ZEV_DE/YLR403W-10min.DE.txt",
			"../McIsaac_ZEV_DE/YLR403W-15min.DE.txt",
			"../McIsaac_ZEV_DE/YLR403W-20min.DE.txt"]
file_rnaseq = None
file_kemmeren = "../Holstege_DE/YLR403W.DE.tsv"
file_hu	= "../Hu_DE/YLR403W.DE.tsv"
batch_calculate_direction_of_change(tf, files_zev, file_rnaseq, file_kemmeren, file_hu)

tf = "ZAP1"
files_zev = ["../McIsaac_ZEV_DE/YJL056C-10min.DE.txt",
			"../McIsaac_ZEV_DE/YJL056C-15min.DE.txt",
			"../McIsaac_ZEV_DE/YJL056C-20min.DE.txt"]
file_rnaseq = None
file_kemmeren = "../Holstege_DE/YJL056C.DE.tsv"
file_hu	= "../Hu_DE/YJL056C.DE.tsv"
batch_calculate_direction_of_change(tf, files_zev, file_rnaseq, file_kemmeren, file_hu)
# """