#!/usr/bin/python
import os
import sys
import numpy as np
import argparse
import glob
from model_fitting_util import *
import pickle

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
    parser.add_argument("-l","--optimized_labels")
    parser.add_argument("-o","--output_fig_filename")
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
								'_'.join([parsed.output_fig_filename, sample_name]), "accu")
			plot_holdout_features(scores_test, scores_holdout, features_var, cc_features, 
								'_'.join([parsed.output_fig_filename, sample_name]), "sens_n_spec")
			plot_holdout_features(scores_test, scores_holdout, features_var, cc_features, 
								'_'.join([parsed.output_fig_filename, sample_name]), "prob_DE")
			plot_holdout_features(scores_test, scores_holdout, features_var, cc_features, 
								'_'.join([parsed.output_fig_filename, sample_name]), "rel_prob_DE")


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
			print("Dummy regressor:", calculate_dummy_regression_score(cc_data, labels))

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
			print("-------------------------------------")
			print("R-squared: (max) %.3f, (min) %.3f, (median) %.3f" % (np.max(scores_test['rsq']), np.min(scores_test['rsq']), np.median(scores_test['rsq'])))
			print("Var explained: (max) %.3f, (min) %.3f, (median) %.3f" % (np.max(scores_test['var_exp']), np.min(scores_test['var_exp']), np.median(scores_test['var_exp'])))
			print("-------------------------------------")
	

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
				print('... loading %s' % sample_name)
				cc[sample_name] = parse_cc_json(file_cc)

		iteration = 0 ## track peak iteration
		focused_orfs = None
		while True:
			print('\n***** Iteration %d *****' % (iteration+1))
			data_collection = {}
			for sample_name in cc.keys():
				print('... working on %s' % sample_name)
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
			print('\n... working on %s\n' % sample_name)
			labels = data_collection[sample_name]['labels']
			cc_data = data_collection[sample_name]['cc_data']
			orfs = data_collection[sample_name]['orfs']

			if len(labels) == 0:
				print("DONE: No more next highest peak to work on.\n")
				break

			## print label information
			neg_labels = len(labels[labels==-1])
			pos_labels = len(labels[labels==1])
			total_labels = float(len(labels))
			chance = (pos_labels/total_labels*pos_labels + neg_labels/total_labels*neg_labels)/ total_labels
			print('Bound not DE %d | Bound and DE %d | chance ACC: %.3f' % (neg_labels, pos_labels,chance))
			focused_orfs = rank_linked_tree_features(cc_data, labels, cc_features, orfs,
													sample_name, iteration)
			print('# of misclassified targets / total targets = %d / %d\n' % (len(focused_orfs), len(orfs)))

			if len(focused_orfs) == 0: ##no more misclassified orfs or no next peak in any orf
				print("DONE:No more misclassified targets.\n")
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
			print('... working on %s' % sample_name)

			## parse calling cards and DE data
			cc_data, labels, cc_features, orfs = prepare_datasets_w_optimzed_labels(file_cc, optimized_labels[sample_name])

			## store data
			data_collection[sample_name] = {'cc_data': cc_data, 
											'labels': labels,
											'orfs': orfs}

		## combine samples
		data_collection = combine_samples(data_collection, len(cc_features))
		for sample in ['combined-all']:
			print('\n... working on %s\n' % sample)
			labels = data_collection[sample]['labels']
			cc_data = data_collection[sample]['cc_data']
			orfs = data_collection[sample_name]['orfs']

			## print label information
			neg_labels = len(labels[labels==-1])
			pos_labels = len(labels[labels==1])
			total_labels = float(len(labels))
			chance = (pos_labels/total_labels*pos_labels + neg_labels/total_labels*neg_labels)/ total_labels
			print('Bound not DE %d | Bound and DE %d | chance ACC: %.3f' % (neg_labels, pos_labels,chance))

			## rank features
			if parsed.feature_type == "binned_promoter":
				indx_focus = range(0,cc_data.shape[1],2)
				sequential_rank_features(cc_features[indx_focus], cc_data[:,indx_focus], labels, method=parsed.method)
				indx_focus = [i+1 for i in indx_focus]
				sequential_rank_features(cc_features[indx_focus], cc_data[:,indx_focus], labels, method=parsed.method)

				print("5-fold corss-validation")
				indx_focus = range(0,cc_data.shape[1],2)
				sequential_rank_features(cc_features[indx_focus], cc_data[:,indx_focus], labels, method=parsed.method, cv=5)
				indx_focus = [i+1 for i in indx_focus]
				sequential_rank_features(cc_features[indx_focus], cc_data[:,indx_focus], labels, method=parsed.method, cv=5)

			else:
				sequential_rank_features(cc_features, cc_data, labels, method=parsed.method, verbose=True)
				print("10-fold cross-validation")
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
		np.savetxt('../output/tmp.'+classifier+'.txt', compiled_results, fmt="%s", delimiter='\t')


	elif parsed.method == "interactive_tf_feature_learning":
		## parse input
		files_cc = glob.glob(parsed.cc_dir +"/*.cc_feature_matrix."+ parsed.feature_type +".txt")
		label_type = "conti2categ"
		data_collection, cc_features = process_data_collection(files_cc, files_de,
												valid_sample_names, label_type, False)
		## query samples
		classifier = "RandomForestClassifier"
		# classifier = "GradientBoostingClassifier"
		feature_filtering_prefix = "logrph_total"

		compiled_results = np.empty((20,0))
		paired_samples = []
		for sample_name in sorted(data_collection):
			compiled_results_col = []
			for other_sample in data_collection.keys():
				if other_sample.endswith(sample_name.split('-')[1]) and other_sample != sample_name:
					print("$$$%s" % ",".join([sample_name, other_sample]))
					labels, cc_data0, _ = query_data_collection(data_collection, sample_name,
														cc_features, feature_filtering_prefix)
					_, cc_data1, _ = query_data_collection(data_collection, other_sample,
														cc_features, feature_filtering_prefix)
					cc_data = np.hstack((cc_data0, cc_data1))
					## use interactive feature to train and predict
					results = model_interactive_feature(cc_data, labels, classifier)
					compiled_results_col += results
			compiled_results = np.hstack((compiled_results, 
											np.array(compiled_results_col).reshape(-1,1)))
		np.savetxt('../output/tmp.'+classifier+'.txt', compiled_results, 
					fmt="%s", delimiter='\t')


	elif parsed.method == "interactive_bp_feature_learning":
		## parse input
		label_type = "conti2categ"
		files_cc = glob.glob(parsed.cc_dir +"/*.cc_feature_matrix."+ parsed.feature_type +".txt")
		cc_data_collection, cc_features = process_data_collection(files_cc, files_de,
												valid_sample_names, label_type, False)
		files_bp = glob.glob(parsed.bp_dir +"/*.cc_feature_matrix."+ parsed.feature_type +".txt")
		bp_data_collection, bp_features = process_data_collection(files_bp, files_de,
												valid_sample_names, label_type, False)
		## query samples
		classifier = "RandomForestClassifier"
		# classifier = "GradientBoostingClassifier"
		cc_feature_filtering_predix = "logrph_total"
		bp_feature_filtering_prefix = ["sum_score", "count", "dist"]
		
		compiled_results = np.empty((30,0))
		for sample_name in sorted(cc_data_collection):
			compiled_results_col = []
			for i in range(len(bp_feature_filtering_prefix)):
			# for i in [len(bp_feature_filtering_prefix)-1]:
				labels, cc_data, _ = query_data_collection(cc_data_collection, sample_name,
												cc_features, cc_feature_filtering_predix)
				_, bp_data, _ = query_data_collection(bp_data_collection, sample_name,
												bp_features, bp_feature_filtering_prefix[:(i+1)])
				ccbp_data = np.hstack((cc_data, bp_data))
				print ccbp_data.shape
				## use binding potential feature to train and predict
				results = model_interactive_feature(ccbp_data, labels, classifier)
				compiled_results_col += results
			compiled_results = np.hstack((compiled_results, 
											np.array(compiled_results_col).reshape(-1,1)))
		np.savetxt('../output/tmp.'+classifier+'.txt', compiled_results, 
					fmt="%s", delimiter='\t')


	elif parsed.method == "interactive_tf_bp_feature_learning":
		## parse input
		label_type = "conti2categ"
		files_cc = glob.glob(parsed.cc_dir +"/*.cc_feature_matrix."+ parsed.feature_type +".txt")
		cc_data_collection, cc_features = process_data_collection(files_cc, files_de,
												valid_sample_names, label_type, False)
		files_bp = glob.glob(parsed.bp_dir +"/*.cc_feature_matrix."+ parsed.feature_type +".txt")
		bp_data_collection, bp_features = process_data_collection(files_bp, files_de,
												valid_sample_names, label_type, False)
		## query samples
		classifier = "RandomForestClassifier"
		# classifier = "GradientBoostingClassifier"
		cc_feature_filtering_predix = "logrph_total"
		bp_feature_filtering_prefix = ["sum_score", "count", "dist"]
		
		compiled_results = np.empty((60,0))
		for sample_name in sorted(cc_data_collection):
			compiled_results_col = []
			for other_sample in cc_data_collection.keys():
				if other_sample.endswith(sample_name.split('-')[1]) and other_sample != sample_name:
					print("$$$%s" % ",".join([sample_name, other_sample]))
					for i in range(len(bp_feature_filtering_prefix)):
					# for i in [len(bp_feature_filtering_prefix)-1]:
						labels, cc_data0, _ = query_data_collection(cc_data_collection, 
															sample_name, cc_features,
															cc_feature_filtering_predix)
						_, cc_data1, _ = query_data_collection(cc_data_collection, 
															other_sample, cc_features, 
															feature_filtering_prefix)
						_, bp_data0, _ = query_data_collection(bp_data_collection, 
														sample_name, bp_features, 
														bp_feature_filtering_prefix[:(i+1)])
						_, bp_data1, _ = query_data_collection(bp_data_collection, 
														other_sample, bp_features, 
														bp_feature_filtering_prefix[:(i+1)])
						ccbp_data = np.concatenate((cc_data0, cc_data1, bp_data0, bp_data1),axis=1)
						print ccbp_data.shape
						## use binding potential feature to train and predict
						results = model_interactive_feature(ccbp_data, labels, classifier)
						compiled_results_col += results
					compiled_results = np.hstack((compiled_results, 
											np.array(compiled_results_col).reshape(-1,1)))
		np.savetxt('../output/tmp.'+classifier+'.txt', compiled_results, 
					fmt="%s", delimiter='\t')


	else:
		sys.exit("Wrong ranking method!")


if __name__ == "__main__":
    main(sys.argv)
