#!/usr/bin/python
import sys
import os
import glob
import numpy as np
import pandas as pd
from peak_calling_util import cluster_gnashyframe, cluster_orf2hops

""" Required modules
load module scipy
load module pandas
python call_peaks.py -d ../output/
"""

def parse_args(argv):
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-d","--data_dir")
    parser.add_argument("-c","--dist_cutoff", type=int, default=200)
    parsed = parser.parse_args(argv[1:])
    return parsed


def load_total_hops_and_reads(file):
	## load total hops and total reads for each sample
	cnt_dict = {}
	data = np.loadtxt(file, dtype=str)
	for i in range(len(data)):
		cnt_dict[data[i,0]] = {}
		cnt_dict[data[i,0]]["hops"] = float(data[i,1])
		cnt_dict[data[i,0]]["reads"] = float(data[i,2])
	return cnt_dict


def main():
	parsed = parse_args(argv)

	# X = pd.read_csv(file, delimiter='\t', header=None, names=['Chr','Hop Pos','Reads'])
	# Y = cluster_gnashyframe(X, dist_cutoff)
	# print Y

	file_total_hops_reads = parsed.data_dir +'/total_hops_and_reads.tbl'
	file_background = parsed.data_dir +'/NOTF_Minus_Adh1_2015_17_combined.orf_hops'

	T = load_total_hops_and_reads(file_total_hops_reads)

	file_basename_background = os.path.splitext(os.path.basename(file_background))[0]
	BG = pd.read_csv(file_background, delimiter='\t', 
					usecols=[1,3,4,7,9,11], header=None, 
					names=['Hop_pos','Hop_ID','Reads','Orf_pos','Orf','Strand'])

	files_experiment = glob.glob(parsed.data_dir +'/*.orf_hops')
	files_experiment.remove(file_background)
	
	for file_in in files_experiment:
		file_in_basename = os.path.splitext(os.path.basename(file_in))[0]
		print "... working on", file_in_basename
		X = pd.read_csv(parsed.data_dir +'/'+ file_in_basename +'.orf_hops', 
					delimiter='\t', usecols=[1,3,4,7,9,11], header=None, 
					names=['Hop_pos','Hop_ID','Reads','Orf_pos','Orf','Strand'])
		Y = cluster_orf2hops(X, BG, T[file_in_basename], T[file_basename_background], parsed.dist_cutoff)
		Y.to_csv(file_in_basename +'.orf_peaks.'+ str(parsed.dist_cutoff) +'bp', sep='\t', index=False)


if __name__ == "__main__":
	main()

