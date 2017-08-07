#!/usr/bin/python
import sys
import os
import glob
import numpy as np
import pandas as pd
import argparse
import json
import operator

"""
Example usage:
module load pandas 
python generate_features.py -m highest_peaks -i ../output/ -o ../output/ -c 200
"""

def parse_args(argv):
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-i","--input_dir")
    parser.add_argument("-o","--output_dir")
    parser.add_argument("-m","--feature_model", help="Choose from ['binned_promoter','highest_peaks','num_peaks,'linked_peaks','summarized_peaks']")
    parser.add_argument("-pu","--promoter_upstream", type=int)
    parser.add_argument("-pd","--promoter_downstream", type=int)
    parser.add_argument("-w","--bin_width", type=int, default=200)
    parser.add_argument("-t","--file_total_hops_reads", default="../output/total_hops_and_reads.tbl")
    parser.add_argument("-b","--file_background", default="../output/NOTF_Minus_Adh1_2015_17_combined.orf_hops")
    parser.add_argument("-c","--dist_cutoff", default=200)
    parser.add_argument("-bd","--binding_data", default="calling_cards", help="Choose from ['calling_cards',binding_potential']")
    parsed = parser.parse_args(argv[1:])
    return parsed


def load_total_hops_and_reads(file):
	## load total hops and total reads for each sample
	cnt_dict = {}
	data = np.loadtxt(file, dtype=str)
	print data
	for i in range(len(data)):
		cnt_dict[data[i,0]] = {}
		cnt_dict[data[i,0]]["hops"] = float(data[i,1])
		cnt_dict[data[i,0]]["reads"] = float(data[i,2])
	return cnt_dict


def load_orf_hops(file):
	## load as data frame
	return pd.read_csv(file, delimiter='\t', usecols=[1,3,4,7,9,11,12], header=None, 
						names=['Hop_pos','Hop_ID','Reads','Orf_pos','Orf','Strand','Dist'])


def load_orf_peaks(file):
	## load as data frame
	return pd.read_csv(file, delimiter="\t", usecols=[0,9,10,11,12], header=None, skiprows=1,
						names=["Orf","TPH","RPH","Strand","Dist"])


def generate_binned_hop_features(binding_data, bin_width, prom_range, expt, bkgd = "", expt_totals = "", bkgd_totals = ""):
	## get number of bins
	#shift = bin_width/2
	# bins = (prom_range[1]-prom_range[0]) *2 / bin_width - 1
	shift = bin_width 
	bins = int(np.ceil((prom_range[1]-prom_range[0]) / float(bin_width))) 

	bin_dict = {}
	feature_header = []
	for k in range(bins):
		bin_left = shift*k + prom_range[0]
		bin_right = min(bin_left + bin_width, prom_range[1])
		bin_dict[k] = [bin_left, bin_right]
		feature_header.append('_'.join(['tph', str(bin_left), str(bin_right)]))
		feature_header.append('_'.join(['rph', str(bin_left), str(bin_right)]))
		feature_header.append('_'.join(['logrph', str(bin_left), str(bin_right)]))
	feature_header += ['tph_total', 'rph_total', 'logrph_total']

	## initialize feature matrix
	orfs = np.unique(expt["Orf"])
	feature_mtx = np.zeros((len(orfs), bins*3+3))

	## iteratre thru orfs to parse out the hops and reads
	orf_counter = 0
	for i in range(len(orfs)):
		orf = orfs[i]
		orf_counter += 1
		if orf_counter % 1000 == 0:
			print "Analyzing "+str(orf_counter)+"th orf"
		## assign normalized experiment hops and reads to bins
		for j, row in expt.loc[expt["Orf"] == orf].iterrows():
			expt_dist = row["Dist"]

			for k in bin_dict.keys():
				if expt_dist >= bin_dict[k][0] and expt_dist < bin_dict[k][1]:
					if binding_data=="calling_cards":
						curr_tph = 100000/expt_totals["hops"]
						curr_rph = row["Reads"] * 100000/expt_totals["reads"]
					else:
						curr_tph = 1
						curr_rph = row["Reads"]
					feature_mtx[i, k*3] += curr_tph
					feature_mtx[i, k*3+1] += curr_rph
					feature_mtx[i, k*3+2] += np.log2(curr_rph)

		## subtract normalized background hops and reads in bins
		if binding_data=="calling_cards":
			if orf in list(bkgd["Orf"]):
				for j, row in bkgd.loc[bkgd["Orf"] == orf].iterrows():
					bkgd_dist = row["Dist"]

					for k in bin_dict.keys():
						if bkgd_dist >= bin_dict[k][0] and bkgd_dist < bin_dict[k][1]:
							curr_tph = 100000/bkgd_totals["hops"]
							curr_rph = row["Reads"] * 100000/bkgd_totals["reads"]
							feature_mtx[i, k*3] -= curr_tph
							feature_mtx[i, k*3+1] -= curr_rph
							feature_mtx[i, k*3+2] -= np.log2(curr_rph)
		## set negative entries to zero
		feature_mtx[feature_mtx < 0] = 0

		## sum tph, rph, logrph (every 3 columns)
		for i in range(3):
			feature_mtx[:, bins*3+i] = np.sum(feature_mtx[:, np.arange(0,bins*3,3)+i], axis=1)

	feature_mtx = np.hstack(( orfs[np.newaxis].T, feature_mtx ))
	feature_mtx = np.vstack(( np.array(['#orf']+feature_header)[np.newaxis], feature_mtx ))

	return feature_mtx


def generate_highest_peaks_features(peaks_df, sort_by="TPH", num_peaks=1, max_dist=-1200):
	## create peak dictionary
	feature_dict = generate_linked_peaks_features(peaks_df, sort_by)
	# peak_feature_names = ['Dist', 'RPH', 'TPH']
	peak_feature_names = sorted(feature_dict[feature_dict.keys()[0]]["1"].keys())
	feature_mtx = []
	sorted_orfs = sorted(feature_dict.keys())
	for orf in sorted_orfs:
		sorted_peaks = feature_dict[orf]["Sorted_peaks"]
		feature_row = []
		## iteratively append peaks, sorted by height
		for i in range(num_peaks): 
			if i >= len(sorted_peaks): ## append null peaks
				feature_row += [max_dist,0,0]
			else: ## append valid peaks
				peak_dict = feature_dict[orf][sorted_peaks[i]]
				feature_row += [peak_dict[x] for x in peak_feature_names]
		feature_mtx.append(feature_row)
	## add rownames and header
	feature_mtx = np.hstack((np.array(sorted_orfs)[np.newaxis].T, np.array(feature_mtx)))
	header = ['#orf']
	for i in range(num_peaks): 
		header += ["".join(['p',str(i+1),'_',x]) for x in peak_feature_names]
	feature_mtx = np.vstack((np.array(header)[np.newaxis], feature_mtx))
	return feature_mtx

def generate_num_peaks_features(peaks_df, sort_by="TPH", max_dist=-1200):
	## create peak dictionary
	feature_dict = generate_linked_peaks_features(peaks_df, sort_by)
	# peak_feature_names = ['Dist', 'RPH', 'TPH']
	peak_feature_names = sorted(feature_dict[feature_dict.keys()[0]]["1"].keys())
	feature_mtx = []
	sorted_orfs = sorted(feature_dict.keys())
	for orf in sorted_orfs:
		sorted_peaks = feature_dict[orf]["Sorted_peaks"]
		feature_row = []
		## iteratively append peaks, sorted by height
		if len(sorted_peaks)==0:
			feature_row += [max_dist,0,0,0]
		else:
			peak_dict = feature_dict[orf][sorted_peaks[0]]
			feature_row += [peak_dict[x] for x in peak_feature_names]
			feature_row.append(int(len(sorted_peaks)))
		feature_mtx.append(feature_row)
	## add rownames and header
	feature_mtx = np.hstack((np.array(sorted_orfs)[np.newaxis].T, np.array(feature_mtx)))
	header = ['#orf']
	header += ["".join(['p',str(1),'_',x]) for x in peak_feature_names]
	header += ["num_peaks"]
	feature_mtx = np.vstack((np.array(header)[np.newaxis], feature_mtx))
	return feature_mtx

def generate_linked_peaks_features(peaks_df, sort_by="TPH"):
	feature_dict = {}
	## iteratre thru dataframe to load everything into feature dictionary
	for i, row in peaks_df.iterrows():
		orf = row["Orf"]
		if not orf in feature_dict.keys():
			feature_dict[orf] = {}
			peak_counter = 1
		feature_dict[orf][str(peak_counter)] = {"TPH":row["TPH"], 
											"RPH":row["RPH"], 
											"Dist":row["Dist"]}
		# feature_dict[orf][str(peak_counter)] = {sort_by:row[sort_by], 
		# 									"Dist":row["Dist"]}
		peak_counter += 1
	## iterature thru peaks of each orf, sort peaks by height and add pointer
	for orf, orf_peaks in feature_dict.iteritems():
		if len(orf_peaks) > 1:
			peaks_sorted = sorted(orf_peaks.iteritems(), 
								key=lambda (x,y):float(y[sort_by]))[::-1]
			feature_dict[orf]["Sorted_peaks"] = [str(peaks_sorted[i][0]) for i in range(len(peaks_sorted))]
			for i in range(len(peaks_sorted)-1):
				curr_peak = peaks_sorted[i][0]
				next_peak = peaks_sorted[i+1][0]
		else:
			feature_dict[orf]["Sorted_peaks"] = [str(1)]
			
	return feature_dict


def generate_summarized_peak_features(file, promoter_range=None):
	## create feature dictionary
	feature_dict = {}
	data = np.loadtxt(file, dtype=str, delimiter='\t', skiprows=1)

	for i in range(len(data)):
		orf, strand = data[i,[0,8]]
		tph, rph = np.array(data[i,range(4,6)])
		tph_bs, rph_bs = np.array(data[i,range(8,10)], dtype=float)
		dist = float(data[i,11])

		if not orf in feature_dict.keys():
			feature_dict[orf] = {'unique_peaks': 0,
								'tph': [],
								'rph': [],
								'tph_bs': [],
								'rph_bs': [],
								'dists_to_atg': []}

		feature_dict[orf]['unique_peaks'] += 1
		feature_dict[orf]['tph'].append(tph)
		feature_dict[orf]['rph'].append(rph)
		feature_dict[orf]['tph_bs'].append(tph_bs)
		feature_dict[orf]['rph_bs'].append(rph_bs)
		feature_dict[orf]['dists_to_atg'].append(dist)

	## convert feature dictionary to feature matrix
	large_dist = 5000
	feature_header = ['unique_peaks',
					'total_tph_bs',
					'total_rph_bs',
					'median_tph_bs',
					'median_rph_bs',
					'median_abs_dist',
					'max_tph_bs',
					'max_rph_bs',
					'abs_dist_max_tph_bs',
					'abs_dist_max_rph_bs']
	feature_indx_dict = {}
	for i in range(len(feature_header)):
		feature_indx_dict[feature_header[i]] = i

	## initialize feature matrix
	bound_orfs = np.sort(feature_dict.keys())
	feature_mtx = np.zeros((len(bound_orfs),len(feature_header)))
	
	# for f in ['median_abs_dist', 'median_us_dist', 'median_ds_dist', 'abs_dist_max_hop_cnt', 'abs_dist_max_read_cnt', 'dist_max_us_hop_cnt', 'dist_max_us_read_cnt', 'dist_max_ds_hop_cnt', 'dist_max_ds_read_cnt']:
	for f in ['median_abs_dist', 'abs_dist_max_tph_bs', 'abs_dist_max_rph_bs']:
		feature_mtx[:,feature_indx_dict[f]] = large_dist

	for i in range(len(bound_orfs)):
		orf = bound_orfs[i]
		unique_peaks = feature_dict[orf]['unique_peaks']
		tph = np.array(feature_dict[orf]['tph_bs'])
		rph = np.array(feature_dict[orf]['rph_bs'])
		dists_to_atg = np.array(feature_dict[orf]['dists_to_atg'])

		## parse orfs with non-empty hops
		if unique_peaks > 0:
			feature_mtx[i, feature_indx_dict['unique_peaks']] = unique_peaks
			feature_mtx[i, feature_indx_dict['total_tph_bs']] = sum(tph)
			feature_mtx[i, feature_indx_dict['total_rph_bs']] = sum(rph)
			feature_mtx[i, feature_indx_dict['median_tph_bs']] = np.median(tph)
			feature_mtx[i, feature_indx_dict['median_rph_bs']] = np.median(rph)
			feature_mtx[i, feature_indx_dict['median_abs_dist']] = np.median(np.abs(dists_to_atg))
			indx_max_tph = np.argmax(tph)
			feature_mtx[i, feature_indx_dict['max_tph_bs']] = tph[indx_max_tph]
			feature_mtx[i, feature_indx_dict['abs_dist_max_tph_bs']] = np.abs(dists_to_atg[indx_max_tph])
			indx_max_rph = np.argmax(rph)
			feature_mtx[i, feature_indx_dict['max_rph_bs']] = tph[indx_max_rph]
			feature_mtx[i, feature_indx_dict['abs_dist_max_rph_bs']] = np.abs(dists_to_atg[indx_max_rph])

	feature_mtx = np.hstack(( bound_orfs[np.newaxis].T, feature_mtx ))
	feature_mtx = np.vstack(( np.array(['#orf']+feature_header)[np.newaxis], feature_mtx ))
	return feature_mtx


def write_feature_matrix(file, features):
	np.savetxt(file, features, fmt="%s", delimiter="\t")


def write_feature_dict(file, features):
	with open(file, "w") as fp:
		json.dump(features, fp)


def main(argv):
	parsed = parse_args(argv)
	## prepare promoter parameter
	promoter_range = [parsed.promoter_upstream, parsed.promoter_downstream]
	if promoter_range[0] > 0:
		promoter_range[0] = -1*promoter_range[0]

	if parsed.binding_data == "binding_potential":
		writer = open(parsed.output_dir +"/total_hops_and_reads.tbl", "w")
		writer.write("#file_basename\thop_cnt\tread_cnt\n")
		for file_orfhops in glob.glob(parsed.output_dir +"/*.orf_hops"):
			sample_basename = os.path.splitext(os.path.basename(file_orfhops))[0]
			writer.write("%s\t%d\t%d\n" % (sample_basename, 0, 0))
		writer.close()

	if parsed.feature_model == "binned_promoter":
		
		if parsed.binding_data == "binding_potential":
			files_experiment = glob.glob(parsed.input_dir +'/*.orf_hops')
			for file_in in files_experiment:
				file_in_basename = os.path.basename(file_in).split(".")[0]
				print "... working on", file_in_basename
				experiment = load_orf_hops(file_in)
				feature_matrix = generate_binned_hop_features(parsed.binding_data, parsed.bin_width, promoter_range, experiment)
				file_output = parsed.output_dir +"/"+ file_in_basename +".cc_feature_matrix.binned_promoter.txt"
				write_feature_matrix(file_output, feature_matrix)

		elif parsed.binding_data == "calling_cards":
			## get total hops and reads, and background data
			totals_dict = load_total_hops_and_reads(parsed.file_total_hops_reads)
			file_basename_background = os.path.splitext(os.path.basename(parsed.file_background))[0]
			background_totals = totals_dict[file_basename_background]
			background = load_orf_hops(parsed.file_background)

			## generate features in binned promoter regions
			files_experiment = glob.glob(parsed.input_dir +'/*.orf_hops')
			files_experiment.remove(parsed.file_background)
			for file_in in files_experiment:
				file_in_basename = os.path.basename(file_in).split(".")[0]
				print "... working on", file_in_basename
				experiment_totals = totals_dict[file_in_basename]
				experiment = load_orf_hops(file_in)
				feature_matrix = generate_binned_hop_features(parsed.binding_data, parsed.bin_width, promoter_range,
											experiment, background,
											experiment_totals, background_totals)
				file_output = parsed.output_dir +"/"+ file_in_basename +".cc_feature_matrix.binned_promoter.txt"
				write_feature_matrix(file_output, feature_matrix)

	elif parsed.feature_model == "highest_peaks":
		## generate features in a linked list (json)
		files_experiment = glob.glob(parsed.input_dir +'/*.orf_peaks.'+ parsed.dist_cutoff +'bp')
		for file_in in files_experiment:
			file_in_basename = os.path.basename(file_in).split(".")[0]
			print "... working on", file_in_basename
			peaks_dataframe = load_orf_peaks(file_in)
			feature_matrix = generate_highest_peaks_features(peaks_dataframe, "RPH", 2)
			file_output = parsed.output_dir +"/"+ file_in_basename +".cc_feature_matrix.highest_peaks.txt"
			write_feature_matrix(file_output, feature_matrix)

	elif parsed.feature_model == "num_peaks":
		## generate features in a linked list (json)
		files_experiment = glob.glob(parsed.input_dir +'/*.orf_peaks.200bp')
		for file_in in files_experiment:
			file_in_basename = os.path.basename(file_in).split(".")[0]
			print "... working on", file_in_basename
			peaks_dataframe = load_orf_peaks(file_in)
			feature_matrix = generate_num_peaks_features(peaks_dataframe, "RPH")
			file_output = parsed.output_dir +"/"+ file_in_basename +".cc_feature_matrix.num_peaks.txt"
			write_feature_matrix(file_output, feature_matrix)

	elif parsed.feature_model == "linked_peaks":
		## generate features in a linked list (json)
		files_experiment = glob.glob(parsed.input_dir +'/*.orf_peaks.'+ parsed.dist_cutoff +'bp')
		for file_in in files_experiment:
			file_in_basename = os.path.basename(file_in).split(".")[0]
			print "... working on", file_in_basename
			peaks_dataframe = load_orf_peaks(file_in)
			feature_dict = generate_linked_peaks_features(peaks_dataframe, "TPH")
			file_output = parsed.output_dir +"/"+ file_in_basename +".cc_feature_matrix.linked_peaks.json"
			write_feature_dict(file_output, feature_dict)  	

	elif parsed.feature_model == "summarized_peaks":
		# generate features by summarizing the attributes of peaks
		for file_in in glob.glob(dir_data+'/*.orf_peaks.'+ parsed.dist_cutoff +'bp'):
			file_prefix = file_in.strip('orf_peaks.'+ parsed.dist_cutoff +'bp')
			file_out = file_prefix+'cc_feature_matrix.summarized_orf_peaks.txt'
			print '... working on', file_prefix.strip('./')
			## generate calling cards feature matrix from clustered peak data
			feature_matrix = generate_summarized_peak_features(file_in, promoter_range)
			write_feature_matrix(file_output, feature_matrix)

	else:
		sys.exit("Wrong feature model")


if __name__ == "__main__":
	main(sys.argv)
