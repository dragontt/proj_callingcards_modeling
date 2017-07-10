"""find_sig_promoters.py 

written 2/12/16 by RDM
modified 4/11/16 by RDM to include TPH field
modified 4/13/16 by RDM to include background subtrated TPH
and also to have TPH Raw which is not background subtracted.
USAGE

python find_sig_promoters
	required fields:
	-b, --bfile  background filename and full path
	-rp --refpath  path to reference files (notORF file and orf coding filename)
	-g -gnashyfile gnashy filename 
	-op --outpath  output path


The output file contains the following fields:

	Intergenic Region
	Left Feature
	Right Feature
	Chr
	Start
	End
	Background Hops
	Experiment Hops
	Fraction Background
 	Fraction Experiment
 	EB ratio
	CHG pvalue
	Poisson pvalue
	Background Hops Lax
	Experiment Hops Lax
	CHG pvalue Lax
	Poisson pvalue Lax
	Left Common Name
	Right Common Name
"""

import sys
import os
import numpy as np 
import pandas as pd 
import math 
import re
import Bio
import Bio.SeqIO
import scipy.stats as scistat
import math
import argparse

def find_significant_IGRs(background_gnashy_filename,experiment_gnashy_filename,refpath):
	#Locations of important files.  Change this for different Accounts!!
	print experiment_gnashy_filename
	NotFeatures_filename = "notORF_RDM_SCG2008_filtered.txt"
	orfcoding_filename = "orf_coding_all_R61-1-1_20080606.fasta"

	lax_value = 200
	

	#read in intergenic regions and populate columns
	IGR_frame = readin_NotFeatures(refpath+NotFeatures_filename)

	#read in background and experiment hops gnashy files
	#and populate expected and observed hops
	[IGR_frame,bg_hops,exp_hops] = readin_hops(IGR_frame,background_gnashy_filename,experiment_gnashy_filename,lax_value)
	
	#read in orf table and populate common names
	IGR_frame = populate_common_names(IGR_frame,refpath+orfcoding_filename)

	#compute cumulative hypergeometric
	IGR_frame = compute_cumulative_hypergeometric(IGR_frame,bg_hops,exp_hops)

	#compute poisson
	IGR_frame = compute_cumulative_poisson(IGR_frame,bg_hops,exp_hops)
	
	#remove His3 false positive
	IGR_frame = IGR_frame[IGR_frame['Right Common Name'] != "HIS3"]

	#output frame
	IGR_frame = IGR_frame.sort_values(["TPH"],ascending = [False])
	if re.search(r'/',experiment_gnashy_filename):
		m = re.search(r'/([\w\.]+)$',experiment_gnashy_filename)
		base_experiment_gnashy_filename = m.group(1)
	else:
		base_experiment_gnashy_filename = experiment_gnashy_filename
	output_filename = "sig_prom_"+base_experiment_gnashy_filename+'.txt'
	IGR_frame.to_csv(output_filename,sep = '\t',index = None)

def readin_NotFeatures(filename):
	#initialize IGR Dataframe
	IGR_frame = pd.DataFrame(columns = ["Intergenic Region","Chr","Start","End","Left Feature","Right Feature","Left Common Name","Right Common Name","TPH","Log TPH",
		"TPH BS","Log TPH BS","CHG pvalue","Poisson pvalue","Background Hops","Experiment Hops","Fraction Background",
		"Fraction Experiment","EB ratio","Background Hops Lax","Experiment Hops Lax","TPH Lax","TPH Lax BS",
		"CHG pvalue Lax","Poisson pvalue Lax"
		])

	notORF_frame = pd.read_csv(filename,delimiter = '\t')

	IGR_frame["Intergenic Region"] = notORF_frame["Name"]
	IGR_frame["Chr"] = notORF_frame["Chr"]
	IGR_frame["Left Feature"] = notORF_frame["Up"]
	IGR_frame["Right Feature"] = notORF_frame["Down"]
	IGR_frame["Start"] = notORF_frame["Start"]
	IGR_frame["End"] = notORF_frame["Stop"]

	return IGR_frame

def readin_hops(IGR_frame,background_gnashy_filename,experiment_gnashy_filename,lax_value):
	background_frame = pd.read_csv(background_gnashy_filename, delimiter = "\t",header = None)
	background_frame.columns = ['Chr','Pos','Reads']
	bg_hops = len(background_frame)
	#print experiment_gnashy_filename
	experiment_frame = pd.read_csv(experiment_gnashy_filename, delimiter = "\t",header = None)
	experiment_frame.columns = ['Chr','Pos','Reads']
	exp_hops = len(experiment_frame)
	multiplication_factor = float(exp_hops) / float(bg_hops)
	print "There were "+str(exp_hops)+" hops in the experiment and "+str(bg_hops)+" in the background file"
	print "The multiplication factor is "+str(multiplication_factor)
	for indexvar in IGR_frame.index:
		IGR_frame.ix[indexvar,"Background Hops"] = len(background_frame[(background_frame["Chr"]==IGR_frame.ix[indexvar,"Chr"]) & (background_frame["Pos"] <= IGR_frame.ix[indexvar,"End"]) &(background_frame["Pos"] >= IGR_frame.ix[indexvar,"Start"])])
		IGR_frame.ix[indexvar,"Experiment Hops"] = len(experiment_frame[(experiment_frame["Chr"]==IGR_frame.ix[indexvar,"Chr"]) & (experiment_frame["Pos"] <= IGR_frame.ix[indexvar,"End"]) &(experiment_frame["Pos"] >= IGR_frame.ix[indexvar,"Start"])])
		IGR_frame.ix[indexvar,"Fraction Background"] = float(IGR_frame.ix[indexvar,"Background Hops"])/float(bg_hops)
		IGR_frame.ix[indexvar,"Fraction Experiment"] = float(IGR_frame.ix[indexvar,"Experiment Hops"])/float(exp_hops)
		#compute transpositions per hundred thousand 
		IGR_frame.ix[indexvar,'TPH']= float(IGR_frame.ix[indexvar,"Experiment Hops"])* (100000/float(exp_hops))
		IGR_frame.ix[indexvar,'TPH BS']= float(IGR_frame.ix[indexvar,"Experiment Hops"])* (100000/float(exp_hops)) - float(IGR_frame.ix[indexvar,"Background Hops"])* (100000/float(bg_hops))
		if IGR_frame.ix[indexvar,'TPH BS'] < 0:
			IGR_frame.ix[indexvar,'TPH BS'] = 0
		if IGR_frame.ix[indexvar,"Fraction Background"] > 0:
			IGR_frame.ix[indexvar,"EB ratio"] = IGR_frame.ix[indexvar,"Fraction Experiment"]/IGR_frame.ix[indexvar,"Fraction Background"]
		else:
			IGR_frame.ix[indexvar,"EB ratio"] = np.nan
		IGR_frame.ix[indexvar,"Background Hops Lax"] = len(background_frame[(background_frame["Chr"]==IGR_frame.ix[indexvar,"Chr"]) & (background_frame["Pos"] <= (IGR_frame.ix[indexvar,"End"] + lax_value)) &(background_frame["Pos"] >= (IGR_frame.ix[indexvar,"Start"] - lax_value))])
		IGR_frame.ix[indexvar,"Experiment Hops Lax"] = len(experiment_frame[(experiment_frame["Chr"]==IGR_frame.ix[indexvar,"Chr"]) & (experiment_frame["Pos"] <= (IGR_frame.ix[indexvar,"End"] + lax_value)) &(experiment_frame["Pos"] >= (IGR_frame.ix[indexvar,"Start"] - lax_value))])
		IGR_frame.ix[indexvar,'TPH Lax']= float(IGR_frame.ix[indexvar,"Experiment Hops Lax"])* (100000/float(exp_hops))
		IGR_frame.ix[indexvar,'TPH Lax BS']= float(IGR_frame.ix[indexvar,"Experiment Hops Lax"])* (100000/float(exp_hops)) - float(IGR_frame.ix[indexvar,"Background Hops Lax"])* (100000/float(bg_hops))
		if IGR_frame.ix[indexvar,'TPH Lax BS'] < 0:
			IGR_frame.ix[indexvar,'TPH Lax BS'] = 0
		IGR_frame.ix[indexvar,"Log TPH"] = np.log2(IGR_frame.ix[indexvar,"TPH"])
		if np.isneginf(IGR_frame.ix[indexvar,"Log TPH"]):
			IGR_frame.ix[indexvar,"Log TPH"] = np.nan
		IGR_frame.ix[indexvar,"Log TPH BS"] = np.log2(IGR_frame.ix[indexvar,"TPH BS"])
		if np.isneginf(IGR_frame.ix[indexvar,"Log TPH BS"]):
			IGR_frame.ix[indexvar,"Log TPH BS"] = np.nan
	return IGR_frame,bg_hops,exp_hops

def populate_common_names(IGR_frame,orfcoding_filename):
	orf_dict = {}
	for x in Bio.SeqIO.parse(orfcoding_filename,"fasta"):
		pattern = '(?<=Y)\S+(?= )'
		matchobj = re.search(pattern,x.description)
		if matchobj:
			y_name = "Y"+matchobj.group(0)
			pattern = '\S+(?= SGDID)'
			matchobj = re.search(pattern,x.description)
			if matchobj:
				orf_dict[y_name] = matchobj.group(0)
	for idx,row in IGR_frame.iterrows():
		if IGR_frame.ix[idx,"Left Feature"] in orf_dict:
			IGR_frame.ix[idx,"Left Common Name"] = orf_dict[row["Left Feature"]]
		else:
			IGR_frame.ix[idx,"Left Common Name"] = row["Left Feature"]
		if IGR_frame.ix[idx,"Right Feature"] in orf_dict:
			IGR_frame.ix[idx,"Right Common Name"] = orf_dict[row["Right Feature"]]
		else:
			IGR_frame.ix[idx,"Right Common Name"] = row["Right Feature"]
	return IGR_frame

def compute_cumulative_hypergeometric(IGR_frame,bg_hops,exp_hops):
	#usage
	#scistat.hypergeom.cdf(x,M,n,N)
	#where x is observed number of type I events (white balls in draw) (experiment hops at locus)
	#M is total number of balls (total number of hops)
	#n is total number of white balls (total number of expeirment hops)
	#N is the number of balls drawn (total hops at a locus)
	for idx,row in IGR_frame.iterrows():
		IGR_frame.ix[idx,"CHG pvalue"] = 1-scistat.hypergeom.cdf(IGR_frame.ix[idx,"Experiment Hops"]-1,(bg_hops + exp_hops),exp_hops,(IGR_frame.ix[idx,"Experiment Hops"]+IGR_frame.ix[idx,"Background Hops"]))
		IGR_frame.ix[idx,"CHG pvalue Lax"] = 1-scistat.hypergeom.cdf(IGR_frame.ix[idx,"Experiment Hops Lax"]-1,(bg_hops + exp_hops),exp_hops,(IGR_frame.ix[idx,"Experiment Hops Lax"]+IGR_frame.ix[idx,"Background Hops Lax"]))
	return IGR_frame

def compute_cumulative_poisson(IGR_frame,bg_hops,exp_hops):
	#usage
	#scistat.poisson.cdf(x,mu)
	pseudocount = 0.2
	for idx,row in IGR_frame.iterrows():
		IGR_frame.ix[idx,"Poisson pvalue"] = 1-scistat.poisson.cdf((IGR_frame.ix[idx,"Experiment Hops"]+pseudocount),(IGR_frame.ix[idx,"Background Hops"] * (float(exp_hops)/float(bg_hops)) + pseudocount))
		IGR_frame.ix[idx,"Poisson pvalue Lax"] = 1-scistat.poisson.cdf((IGR_frame.ix[idx,"Experiment Hops Lax"]+pseudocount),(IGR_frame.ix[idx,"Background Hops Lax"] * (float(exp_hops)/float(bg_hops)) + pseudocount))
	return IGR_frame


if __name__ == '__main__': 
	parser = argparse.ArgumentParser(description='This is the find_sig_promoters module for yeast.')
	parser.add_argument('-op','--outputpath',help='output path',required=True)
	parser.add_argument('-g','--gnashyfile',help='gnashy filename',required=True)
	parser.add_argument('-b','--bfile',help='background hop distribution path and filename',required=False,default="/scratch/rmlab/ref/calling_card_ref/yeast/NOTF_Minus_Adh1_2015_17_combined.gnashy" ) #default="/scratch/rmlab/ref/calling_card_ref/yeast/yeast_S288C_dSir4_Background.txt"
	parser.add_argument('-rp','--refpath',help='path to reference files notORF and orf coding file',required=False,default="/scratch/rmlab/ref/calling_card_ref/yeast" )

	args = parser.parse_args()

	if not args.refpath[-1] == "/":
		args.refpath = args.refpath+"/"	
	os.chdir(args.outputpath)
	find_significant_IGRs(args.bfile,args.gnashyfile,args.refpath)


	


