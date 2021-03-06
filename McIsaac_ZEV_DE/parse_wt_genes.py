#!/usr/bin/python
import pandas as pd
import numpy as np
from datetime import date
import sys

## parse orf name conversion
data = pd.read_csv('orf_name_conversion.tab', dtype=str, delimiter='\t', header=None)
conv_dict = {}
for i, row in data.iterrows():
	conv_dict[row[1]] = row[0]

## parse gene expression at time 0
data = pd.read_csv("final_data%2Fyeast_data_table_20171115.tsv", delimiter='\t')
data_t0 = data[data["time"] == 0][['TF', 'strain', 'date', 'restriction', 'mechanism', 'time', 'seq', 'rseq', 'GeneName', 'r_g_median']]

## save wt expression file
for tf in pd.unique(data_t0['TF']):
	print "... working on", tf
	data_t0_tf = data_t0[data_t0['TF']==tf]
	## multiple strains could be used for each TF induction, or multiple date
	strains = pd.unique(data_t0_tf['strain'])
	dates = pd.unique(data_t0_tf[data_t0_tf['strain']==strains[0]]['date'])
	dates_formatted = []
	for i in range(len(dates)):
		month, day, year = [int(x) for x in dates[i].split('/')]
		dates_formatted.append(date(year, month, day))
	recent_date = dates[np.argmax(dates_formatted)]
	## get expression data of each TF induction
	wt_expr = data_t0_tf.loc[(data_t0_tf['strain']==strains[0]) & (data_t0_tf['date']==recent_date)]['r_g_median']
	wt_expr = wt_expr.reset_index().drop(['index'], axis=1)
	## gene name conversion
	genes_symbol = data_t0_tf.loc[(data_t0_tf['strain']==strains[0]) & (data_t0_tf['date']==recent_date)]['GeneName'].tolist()
	genes_systematic = pd.Series([conv_dict[genes_symbol[i]] if genes_symbol[i] in conv_dict else genes_symbol[i] for i in range(len(genes_symbol))], name='GeneSystematic')
	genes_symbol = pd.Series(genes_symbol, name='GeneName')
	## combine output and write to file
	out = pd.concat([genes_systematic, genes_symbol, wt_expr], axis=1)
	output_file = 'all_expression/'+ conv_dict[tf] +'.WT.txt' if tf in conv_dict else 'all_expression/'+ tf +'.WT.txt'
	out.to_csv(output_file, sep="\t", index=False)
