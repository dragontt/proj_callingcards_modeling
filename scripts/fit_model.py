#!/usr/bin/python
import os
import sys
import numpy as np
import argparse
import glob
from model_fitting_util import *
import pickle

from sklearn.model_selection import train_test_split

"""
Example usage:
module load scipy
module load scikit-learn
module load matplotlib

python fit_model.py -m holdout_feature_classification -t highest_peaks -c ../output/ -l ../resources/optimized_cc_subset.txt -o ../output/feature_holdout_analysis.6_mimic_cc
"""

def parse_args(argv):
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-m","--method", 
    					help="choose from ['holdout_feature_classification', 'holdout_feature_regression', 'tree_rank_highest_peaks', 'sequential_forward_selection', 'sequential_backward_selection', 'tree_rank_linked_peaks']")
    parser.add_argument("-t","--feature_type", 
    					help="choose from ['highest_peaks', 'binned_promoter']")
    parser.add_argument("-c","--cc_dir", help="Calling Cards feature directory")
    parser.add_argument("-d","--de_dir", help="Differential expression directory")
    parser.add_argument("-p","--bp_dir", help="Binding potential directory")
    parser.add_argument("-a","--file_ca", help="Chromatin accessibility file")
    parser.add_argument("-w","--wt_dir", help="WT expressions directory")
    parser.add_argument("-l","--optimized_labels")
    parser.add_argument("-o","--output_filename")
    parsed = parser.parse_args(argv[1:])
    return parsed


def main(argv):
	parsed = parse_args(argv)
	
	if parsed.optimized_labels: ## parse optimized set if available 
		optimized_labels = parse_optimized_labels(parsed.optimized_labels)
		valid_sample_names = optimized_labels.keys()
		label_type = "categorical"
	elif parsed.de_dir: ## parse valid DE samples if available
		files_de = glob.glob(parsed.de_dir +"/*.DE.tsv")
		valid_sample_names = [os.path.basename(f).split('.')[0] for f in files_de]
		label_type = "continuous"
	else:
		sys.exit("Require the label directory: optimized subset file or DE folder!") 


	if parsed.method == "holdout_feature_classification":
		## parse input
		files_cc = glob.glob(parsed.cc_dir +"/*.cc_feature_matrix."+ parsed.feature_type +".txt")
		feature_filtering_prefix = "logrph" if parsed.feature_type == "binned_promoter" else None
		# data_collection, cc_features = process_data_collection(files_cc, optimized_labels,
		# 										valid_sample_names, label_type)
		label_type = "conti2categ"
		data_collection, cc_features = process_data_collection(files_cc, files_de,
												valid_sample_names, label_type, True, 0.1)
		## query samples
		# for sample_name in ['combined-all']:
		for sample_name in sorted(data_collection.keys()):
			labels, cc_data, cc_features = query_data_collection(data_collection, sample_name, 
												cc_features, feature_filtering_prefix)
			## print label information
			chance = calculate_chance(labels)
			## model the holdout feature
			classifier = "RandomForestClassifier"
			scores_test, scores_holdout, features_var = model_holdout_feature(cc_data, labels, 
														cc_features, sample_name, classifier, 
														True, 10, 100, False)
			plot_holdout_features(scores_test, scores_holdout, features_var, cc_features, 
								'_'.join([parsed.output_filename, sample_name]), "accu")
			plot_holdout_features(scores_test, scores_holdout, features_var, cc_features, 
								'_'.join([parsed.output_filename, sample_name]), "sens_n_spec")
			plot_holdout_features(scores_test, scores_holdout, features_var, cc_features, 
								'_'.join([parsed.output_filename, sample_name]), "prob_DE")
			plot_holdout_features(scores_test, scores_holdout, features_var, cc_features, 
								'_'.join([parsed.output_filename, sample_name]), "rel_prob_DE")


	elif parsed.method == "holdout_feature_regression":
		## parse input 
		files_cc = glob.glob(parsed.cc_dir +"/*.cc_feature_matrix."+ parsed.feature_type +".txt")
		feature_filtering_prefix = "logrph" if parsed.feature_type == "binned_promoter" else None
		data_collection, cc_features = process_data_collection(files_cc, files_de,
													valid_sample_names, label_type)
		## query samples
		for sample_name in sorted(data_collection.keys()):
			labels, cc_data, cc_features= query_data_collection(data_collection, sample_name, 
													cc_features, feature_filtering_prefix)

			## print label information: dummy regressor? -> average lfc 
			print "Dummy regressor:", calculate_dummy_regression_score(cc_data, labels)

			## model the holdout feature
			# regressor = "RidgeRegressor"
			regressor = "RandomForestRegressor"
			# regressor = "GradientBoostingRegressor"
			# regressor = "GaussianProcessRegressor"
			# regressor = "MLPRegressor"
			scores_test, scores_holdout, features_var = model_holdout_feature(cc_data, 
															labels, cc_features, sample_name, 
															regressor, False, 20, 100, False)
			
			rsq_hist = np.histogram(scores_test['rsq'])
			var_exp_hist = np.histogram(scores_test['var_exp'])
			lfc_hist = np.histogram(scores_test['lfc'])
			print "-------------------------------------"
			print "R-squared: (max) %.3f, (min) %.3f, (median) %.3f" % (np.max(scores_test['rsq']), np.min(scores_test['rsq']), np.median(scores_test['rsq']))
			print "Var explained: (max) %.3f, (min) %.3f, (median) %.3f" % (np.max(scores_test['var_exp']), np.min(scores_test['var_exp']), np.median(scores_test['var_exp']))
			print "-------------------------------------"
	

	elif parsed.method == "tree_rank_highest_peaks":
		## parse input
		files_cc = glob.glob(parsed.cc_dir +"/*.cc_feature_matrix."+ parsed.feature_type +".txt")
		feature_filtering_prefix = "logrph" if parsed.feature_type == "binned_promoter" else None
		data_collection, cc_features = process_data_collection(files_cc, optimized_labels,
												valid_sample_names, label_type)
		## query samples
		sample_name = 'combined-all'
		labels, cc_data, cc_features = query_data_collection(data_collection, sample_name, 
												cc_features, feature_filtering_prefix)
		## print label information
		chance = calculate_chance(labels)
		## rank features
		rank_highest_peaks_features(cc_data, labels, cc_features, sample_name)


	elif parsed.method == "tree_rank_linked_peaks":
		cc = {}
		files_cc = glob.glob(parsed.cc_dir +"/*.cc_feature_matrix.linked_peaks.json")
		for file_cc in files_cc: ## remove wildtype samples
			sample_name = os.path.splitext(os.path.basename(file_cc))[0].split(".")[0]
			if (sample_name in valid_sample_names) and (not sample_name.startswith('BY4741')):
				print '... loading %s' % sample_name
				cc[sample_name] = parse_cc_json(file_cc)

		iteration = 0 ## track peak iteration
		focused_orfs = None
		while True:
			print '\n***** Iteration %d *****' % (iteration+1)
			data_collection = {}
			for sample_name in cc.keys():
				print '... working on %s' % sample_name
				cc_data, labels, cc_features, orfs = prepare_subset_w_optimized_labels(\
										cc[sample_name], optimized_labels[sample_name],
										sample_name, iteration, focused_orfs)
				# rank_linked_tree_features(cc_features, cc_data, labels)
				data_collection[sample_name] = {'cc_data': cc_data, 
												'labels': labels,
												'orfs': orfs}

			## combine samples
			data_collection = combine_samples(data_collection, len(cc_features))
			# sample_names = ['combined-plusLys', 'combined-minusLys', 'combined-all']:
			sample_name = 'combined-all'
			print '\n... working on %s\n' % sample_name
			labels = data_collection[sample_name]['labels']
			cc_data = data_collection[sample_name]['cc_data']
			orfs = data_collection[sample_name]['orfs']

			if len(labels) == 0:
				print "DONE: No more next highest peak to work on.\n"
				break

			## print label information
			neg_labels = len(labels[labels==-1])
			pos_labels = len(labels[labels==1])
			total_labels = float(len(labels))
			chance = (pos_labels/total_labels*pos_labels + neg_labels/total_labels*neg_labels)/ total_labels
			print 'Bound not DE %d | Bound and DE %d | chance ACC: %.3f' % (neg_labels, pos_labels,chance)
			focused_orfs = rank_linked_tree_features(cc_data, labels, cc_features, orfs,
													sample_name, iteration)
			print '# of misclassified targets / total targets = %d / %d\n' % (len(focused_orfs), len(orfs))

			if len(focused_orfs) == 0: ##no more misclassified orfs or no next peak in any orf
				print "DONE:No more misclassified targets.\n"
				break

			iteration += 1 ## build next tree


	elif parsed.method in ['sequential_forward_selection', 'sequential_backward_selection']:
		## run feature selection on each sample
		files_cc = glob.glob(parsed.cc_dir +"/*.cc_feature_matrix."+ parsed.feature_type +".txt")
		for file_cc in files_cc: ## remove wildtype samples
			sample_name = os.path.basename(file_cc).split(".")[0]
			if (sample_name not in valid_sample_names) or (sample_name.startswith('BY4741')):
				files_cc.remove(file_cc)

		data_collection = {}
		for file_cc in files_cc:
			sample_name = os.path.basename(file_cc).split(".")[0]
			print '... working on %s' % sample_name

			## parse calling cards and DE data
			cc_data, labels, cc_features, orfs = prepare_datasets_w_optimzed_labels(file_cc, optimized_labels[sample_name])

			## store data
			data_collection[sample_name] = {'cc_data': cc_data, 
											'labels': labels,
											'orfs': orfs}

		## combine samples
		data_collection = combine_samples(data_collection, len(cc_features))
		for sample in ['combined-all']:
			print '\n... working on %s\n' % sample
			labels = data_collection[sample]['labels']
			cc_data = data_collection[sample]['cc_data']
			orfs = data_collection[sample_name]['orfs']

			## print label information
			neg_labels = len(labels[labels==-1])
			pos_labels = len(labels[labels==1])
			total_labels = float(len(labels))
			chance = (pos_labels/total_labels*pos_labels + neg_labels/total_labels*neg_labels)/ total_labels
			print 'Bound not DE %d | Bound and DE %d | chance ACC: %.3f' % (neg_labels, pos_labels,chance)

			## rank features
			if parsed.feature_type == "binned_promoter":
				indx_focus = range(0,cc_data.shape[1],2)
				sequential_rank_features(cc_features[indx_focus], cc_data[:,indx_focus], labels, method=parsed.method)
				indx_focus = [i+1 for i in indx_focus]
				sequential_rank_features(cc_features[indx_focus], cc_data[:,indx_focus], labels, method=parsed.method)

				print "5-fold corss-validation"
				indx_focus = range(0,cc_data.shape[1],2)
				sequential_rank_features(cc_features[indx_focus], cc_data[:,indx_focus], labels, method=parsed.method, cv=5)
				indx_focus = [i+1 for i in indx_focus]
				sequential_rank_features(cc_features[indx_focus], cc_data[:,indx_focus], labels, method=parsed.method, cv=5)

			else:
				sequential_rank_features(cc_features, cc_data, labels, method=parsed.method, verbose=True)
				print "10-fold cross-validation"
				sequential_rank_features(cc_features, cc_data, labels, method=parsed.method, cv=10, verbose=True)


	elif parsed.method == "simple_precision_recall":
		## parse input
		files_cc = glob.glob(parsed.cc_dir +"/*.cc_feature_matrix."+ parsed.feature_type +".txt")
		data_collection, cc_features = process_data_collection(files_cc, files_de,
												valid_sample_names, label_type)
		## query samples
		compiled_results = np.empty((5,0))
		# for sample_name in ['combined-all']: #["combined-plusLys", "combined-minusLys"]: 
		for sample_name in sorted(data_collection.keys()):
			for feature_filtering_prefix in ["logrph_total"]:
			# for feature_filtering_prefix in ["tph_total", "rph_total", "logrph_total"]:
				labels, cc_data, _ = query_data_collection(data_collection,
															sample_name, cc_features, 
															feature_filtering_prefix)
				## plot precision-recall with randomly permuted signals
				# figname = '../output/PR_'+ feature_filtering_prefix.split('_')[0] +'.'+ sample_name +'.pdf'
				figname = None
				results = plot_precision_recall_w_random_signal(cc_data, labels, 0.1, figname)
				compiled_results = np.hstack((compiled_results, np.array(results).reshape(-1,1)))
		compiled_results = np.vstack((np.array(sorted(data_collection.keys()))[np.newaxis], compiled_results))
		np.savetxt('../output2/tmp.txt', compiled_results, fmt="%s", delimiter='\t')


	elif parsed.method == "single_feature_learning":
		## parse input
		files_cc = glob.glob(parsed.cc_dir +"/*.cc_feature_matrix."+ parsed.feature_type +".txt")
		label_type = "conti2categ"
		data_collection, cc_features = process_data_collection(files_cc, files_de,
												valid_sample_names, label_type, False, 0.1)
		## query samples
		classifier = "RandomForestClassifier"
		# classifier = "GradientBoostingClassifier"
		feature_filtering_prefix = "logrph_total"

		compiled_results = np.empty((10,0))
		for sample_name in sorted(data_collection):
			labels, cc_data, _ = query_data_collection(data_collection, sample_name, 
														cc_features, feature_filtering_prefix)
			## use single feature to train and predict
			results = model_interactive_feature(cc_data, labels, classifier)
			compiled_results = np.hstack((compiled_results, np.array(results).reshape(-1,1)))
		np.savetxt(parsed.output_filename, compiled_results, fmt="%s", delimiter='\t')


	elif parsed.method == "interactive_bp_feature_learning":
		## parse input
		label_type = "conti2categ"
		files_cc = glob.glob(parsed.cc_dir +"/*.cc_feature_matrix."+ parsed.feature_type +".txt")
		cc_data_collection, cc_features = process_data_collection(files_cc, files_de,
												valid_sample_names, label_type, False)
		if parsed.bp_dir:
			files_bp = glob.glob(parsed.bp_dir +"/*.cc_feature_matrix."+ 
									parsed.feature_type +".txt")
			bp_data_collection, bp_features = process_data_collection(files_bp, files_de,
												valid_sample_names, label_type, False)
		if parsed.file_ca:
			ca_data, _, ca_features, _ = prepare_datasets_w_de_labels(parsed.file_ca, files_de[0], "pval", 0.1)
		if parsed.wt_dir:
			files_wt = glob.glob(parsed.wt_dir +"/*.WT_median.expr")
			wt_data_collection, wt_features = process_data_collection(files_wt, files_de,
												valid_sample_names, label_type, False)
		## query samples
		classifier = "RandomForestClassifier"
		# classifier = "GradientBoostingClassifier"
		cc_feature_filtering_prefix = "logrph_total"
		# cc_feature_filtering_prefix = "logrph"
		bp_feature_filtering_prefix = ["sum_score", "count", "dist"]
		ca_feature_filtering_prefix = ['H3K27ac_prom_-1','H3K36me3_prom_-1','H3K4me3_prom_-1',
										'H3K79me_prom_-1','H4K16ac_prom_-1','H3K27ac_body',
										'H3K36me3_body','H3K4me3_body','H3K79me_body',
										'H4K16ac_body']
		
		compiled_results = np.empty((10,0))
		for sample_name in sorted(cc_data_collection):
			compiled_results_col = []
			# for i in range(len(bp_feature_filtering_prefix)):
			for i in [len(bp_feature_filtering_prefix)-1]:
				labels, cc_data, _ = query_data_collection(cc_data_collection, sample_name,
												cc_features, cc_feature_filtering_prefix)
				combined_data = cc_data
				if parsed.bp_dir:
					_, bp_data, _ = query_data_collection(bp_data_collection, sample_name,
											bp_features, bp_feature_filtering_prefix[:(i+1)])
					combined_data = np.concatenate((combined_data, bp_data),axis=1)
				if parsed.file_ca:
					ca_feat_indx = [k for k in range(len(ca_features)) if ca_features[k] in ca_feature_filtering_prefix]
					combined_data = np.concatenate((combined_data, ca_data[:,ca_feat_indx]), axis=1)
				if parsed.wt_dir:
					_, wt_data, _ = query_data_collection(wt_data_collection, 
												sample_name, wt_features)
					combined_data = np.concatenate((combined_data, wt_data), axis=1)
				print combined_data.shape, "+1:", len(labels[labels ==1]), "-1:", len(labels[labels ==-1])
				## use binding potential feature to train and predict
				# results = model_interactive_feature(combined_data, labels, classifier)
				results = model_interactive_feature(combined_data, labels, classifier, 10, True)
				compiled_results_col += results
			compiled_results = np.hstack((compiled_results, 
											np.array(compiled_results_col).reshape(-1,1)))
		np.savetxt(parsed.output_filename, compiled_results, fmt="%s", delimiter='\t')


	elif parsed.method == "interactive_bp_feature_holdout_analysis":
		## parse input
		label_type = "conti2categ"
		files_cc = glob.glob(parsed.cc_dir +"/*.cc_feature_matrix."+ parsed.feature_type +".txt")
		cc_data_collection, cc_features = process_data_collection(files_cc, files_de,
												valid_sample_names, label_type, False)
		if parsed.bp_dir:
			files_bp = glob.glob(parsed.bp_dir +"/*.cc_feature_matrix."+ 
									parsed.feature_type +".txt")
			bp_data_collection, bp_features = process_data_collection(files_bp, files_de,
												valid_sample_names, label_type, False)
		if parsed.file_ca:
			ca_data, _, ca_features, _ = prepare_datasets_w_de_labels(parsed.file_ca, files_de[0], "pval", 0.1)
		if parsed.wt_dir:
			files_wt = glob.glob(parsed.wt_dir +"/*.WT_median.expr")
			wt_data_collection, wt_features = process_data_collection(files_wt, files_de,
												valid_sample_names, label_type, False)

		## query samples
		classifier = "RandomForestClassifier"
		# classifier = "GradientBoostingClassifier"
		cc_feature_filtering_prefix = "logrph_total"
		# cc_feature_filtering_prefix = "logrph"
		bp_feature_filtering_prefix = ["sum_score", "count", "dist"]
		ca_feature_filtering_prefix = ['H3K27ac_prom_-1','H3K36me3_prom_-1','H3K4me3_prom_-1',
										'H3K79me_prom_-1','H4K16ac_prom_-1','H3K27ac_body',
										'H3K36me3_body','H3K4me3_body','H3K79me_body',
										'H4K16ac_body']
		
		for sample_name in sorted(cc_data_collection):
			# for i in range(len(bp_feature_filtering_prefix)):
			for i in [len(bp_feature_filtering_prefix)-1]:
				labels, cc_data, cc_f = query_data_collection(cc_data_collection, sample_name,
												cc_features, cc_feature_filtering_prefix)
				combined_data = cc_data
				combined_feat = cc_f
				if sample_name in ["YKL038W-minusLys", "YKL038W-plusLys"]:
					print cc_data
					print np.max(cc_data), np.min(cc_data), np.median(cc_data), np.percentile(cc_data, 75), np.percentile(cc_data, 25)
				if parsed.bp_dir:
					_, bp_data, bp_f = query_data_collection(bp_data_collection, sample_name,
											bp_features, bp_feature_filtering_prefix[:(i+1)])
					combined_data = np.concatenate((combined_data, bp_data),axis=1)
					combined_feat = np.append(combined_feat, bp_f)
				if parsed.file_ca:
					ca_feat_indx = [k for k in range(len(ca_features)) if ca_features[k] in ca_feature_filtering_prefix]
					ca_f = ca_features[ca_feat_indx]
					combined_data = np.concatenate((combined_data, ca_data[:,ca_feat_indx]), axis=1)
					combined_feat = np.append(combined_feat, ca_f)
				if parsed.wt_dir:
					_, wt_data, wt_f = query_data_collection(wt_data_collection, 
												sample_name, wt_features)
					combined_data = np.concatenate((combined_data, wt_data), axis=1)
					combined_feat = np.append(combined_feat, wt_f)
				print combined_data.shape, "+1:", len(labels[labels ==1]), "-1:", len(labels[labels ==-1])
				## use binding potential feature to train and predict
				from sklearn.model_selection import train_test_split
				combined_data_tr, _, labels_tr, _ = train_test_split(combined_data, labels, 
													test_size=1./10, random_state=1)

				# import matplotlib
				# matplotlib.use('Agg')
				# import matplotlib.pyplot as plt
				# import matplotlib.gridspec as gridspec
				# fig = plt.figure(num=None, figsize=(15, 10), dpi=300)
				# num_cols = np.ceil(np.sqrt(len(combined_feat)))
				# num_rows = np.ceil(len(combined_feat)/num_cols)
				# num_cols = int(num_cols)
				# num_rows = int(num_rows)
				# for i in range(num_rows):
				# 	for j in range(num_cols):
				# 		k = num_cols*i+j ## feature index
				# 		if k < len(combined_feat):
				# 			ax = fig.add_subplot(num_rows, num_cols, k+1)
				# 			ax.boxplot([combined_data[:,k][labels == -1], 
				# 						combined_data[:,k][labels == 1]], 0 , '')
				# 			ax.set_title('%s' % combined_feat[k])
				# 			ax.set_xticklabels(['not DE', 'DE'])
				# plt.savefig('../output/boxplot.'+sample_name+'.pdf', format='pdf')

				scores_test, scores_holdout, features_var = model_holdout_feature(
													combined_data_tr, labels_tr, 
													combined_feat, sample_name, classifier, 
													True, 10, 100, False)
				plot_holdout_features(scores_test, scores_holdout, features_var,combined_feat, 
							'_'.join([parsed.output_filename, sample_name]), "accu")
				plot_holdout_features(scores_test, scores_holdout, features_var,combined_feat, 
							'_'.join([parsed.output_filename, sample_name]), "sens_n_spec")
				plot_holdout_features(scores_test, scores_holdout, features_var,combined_feat, 
							'_'.join([parsed.output_filename, sample_name]), "prob_DE")
				plot_holdout_features(scores_test, scores_holdout, features_var,combined_feat, 
							'_'.join([parsed.output_filename, sample_name]), "rel_prob_DE")


	elif parsed.method == "interactive_bp_feature_ranking":
		## parse input
		label_type = "conti2categ" 
		## label_type = "conti2top500DE"
		files_cc = glob.glob(parsed.cc_dir +"/*.cc_feature_matrix."+ parsed.feature_type +".txt")
		cc_data_collection, cc_features = process_data_collection(files_cc, files_de,
												valid_sample_names, label_type, False)
		if parsed.bp_dir:
			files_bp = glob.glob(parsed.bp_dir +"/*.cc_feature_matrix."+ 
									parsed.feature_type +".txt")
			bp_data_collection, bp_features = process_data_collection(files_bp, files_de,
												valid_sample_names, label_type, False)
		if parsed.file_ca:
			ca_data, _, ca_features, _ = prepare_datasets_w_de_labels(parsed.file_ca, files_de[0], "pval", 0.1)
		if parsed.wt_dir:
			files_wt = glob.glob(parsed.wt_dir +"/*.WT_median.expr")
			wt_data_collection, wt_features = process_data_collection(files_wt, files_de,
												valid_sample_names, label_type, False)
		## query samples
		classifier = "RandomForestClassifier"
		# classifier = "GradientBoostingClassifier"
		cc_feature_filtering_prefix = "logrph_total"
		bp_feature_filtering_prefix = ["sum_score", "count", "dist"]
		
		ca_feature_filtering_prefix = ['H3K27ac_prom_-1','H3K36me3_prom_-1','H3K4me3_prom_-1',
										'H3K79me_prom_-1','H4K16ac_prom_-1','H3K27ac_body',
										'H3K36me3_body','H3K4me3_body','H3K79me_body',
										'H4K16ac_body']
		
		for sample_name in sorted(cc_data_collection):
			compiled_results_col = []
			labels, cc_data, cc_features0 = query_data_collection(cc_data_collection, 
																sample_name, cc_features, 
																cc_feature_filtering_prefix)
			combined_data = cc_data
			all_features = list(cc_features0)
			if parsed.bp_dir:	
				_, bp_data, bp_features0 = query_data_collection(bp_data_collection, sample_name,
										bp_features, bp_feature_filtering_prefix)
				combined_data = np.concatenate((combined_data, bp_data),axis=1)
				all_features += list(bp_features0)
			if parsed.file_ca:
				ca_feat_indx = [k for k in range(len(ca_features)) if ca_features[k] in ca_feature_filtering_prefix]
				combined_data = np.concatenate((combined_data, ca_data[:,ca_feat_indx]), axis=1)
				all_features += list(ca_features[ca_feat_indx])
			if parsed.wt_dir:
				_, wt_data, wt_features0 = query_data_collection(wt_data_collection, 
											sample_name, wt_features)
				combined_data = np.concatenate((combined_data, wt_data), axis=1)
				all_features += list(wt_features0)
			print combined_data.shape, len(all_features)
			print all_features
			
			## use binding potential feature to train and predict
			combined_data_tr, combined_data_te, labels_tr, labels_te = train_test_split(combined_data, labels, test_size=1./10, random_state=1)
			print combined_data_tr.shape, labels_tr.shape
			# sequential_rank_features(all_features, combined_data_tr, labels_tr, "sequential_forward_selection", 10, True)
			sequential_rank_features(all_features, combined_data_tr, labels_tr, "sequential_backward_selection", 10, True)


	else:
		sys.exit("Wrong ranking method!")


if __name__ == "__main__":
    main(sys.argv)
