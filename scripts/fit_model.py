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

python fit_model.py -m holdout_feature_variation -t highest_peaks -c ../output/ -o ../resources/optimized_cc_subset.txt -f ../output/feature_holdout_analysis.6_mimic_cc
"""

def parse_args(argv):
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-m","--ranking_method", help="choose from ['holdout_feature_variation', 'tree_rank_highest_peaks', 'tree_rank_linked_peaks', 'sequential_forward_selection', 'sequential_backward_selection']")
    parser.add_argument("-t","--data_type", default="highest_peaks", help="choose from ['highest_peaks, binned_promoter")
    parser.add_argument("-c","--cc_dir")
    parser.add_argument("-d","--de_dir")
    parser.add_argument("-o","--optimized_labels", default=None)
    parser.add_argument("-l","--threshold_log2FC", type=float)
    parser.add_argument("-p","--threshold_adjustedP", type=float)
    parser.add_argument("-f","--fig_filename")
    parsed = parser.parse_args(argv[1:])
    return parsed


def main(argv):
	parsed = parse_args(argv)
	
	if parsed.optimized_labels: ## parse optimized set if available 
		optimized_labels = parse_optimized_labels(parsed.optimized_labels)
		valid_sample_names = optimized_labels.keys()


	if parsed.ranking_method == "holdout_feature_variation":
		## parse input
		files_cc = glob.glob(parsed.cc_dir +"/*.cc_feature_matrix."+ parsed.data_type +".txt")
		sample_name = 'combined-all'
		feature_filtering_prefix = "tph" if parsed.data_type == "binned_promoter" else None
		labels, cc_data, cc_features = process_data_collection(files_cc, optimized_labels,
												valid_sample_names, sample_name,
												feature_filtering_prefix)
		## print label information
		chance = calculate_chance(labels)
		## model the holdout feature
		classifier = "RandomForestClassifier"
		scores_test, scores_holdout, features_var = model_holdout_feature(cc_data, labels, 
													cc_features, sample_name, classifier,
													10, 100, False)
		plot_holdout_features(scores_test, scores_holdout, features_var, 
							parsed.fig_filename, "accu")
		plot_holdout_features(scores_test, scores_holdout, features_var, 
							parsed.fig_filename, "sens_n_spec")
		plot_holdout_features(scores_test, scores_holdout, features_var, 
							parsed.fig_filename, "prob_DE")
	

	elif parsed.ranking_method == "tree_rank_highest_peaks":
		## parse input
		files_cc = glob.glob(parsed.cc_dir +"/*.cc_feature_matrix.highest_peaks.txt")
		sample_name = 'combined-all'
		labels, cc_data, cc_features = process_data_collection(files_cc, optimized_labels,
												valid_sample_names, sample_name, "tph")
		## print label information
		chance = calculate_chance(labels)
		## rank features
		rank_highest_peaks_features(cc_data, labels, cc_features, sample_name)


	elif parsed.ranking_method == "tree_rank_linked_peaks":
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


	elif parsed.ranking_method in ['sequential_forward_selection', 'sequential_backward_selection']:
		## run feature selection on each sample
		files_cc = glob.glob(parsed.cc_dir +"/*.cc_feature_matrix.binned_promoter.txt")
		for file_cc in files_cc: ## remove wildtype samples
			sample_name = os.path.basename(file_cc).split(".")[0]
			if (sample_name in valid_sample_names) and (sample_name.startswith('BY4741')):
				files_cc.remove(file_cc)

		data_collection = {}
		for file_cc in files_cc:
			sample_name = os.path.basename(file_cc).split(".")[0]
			print '\n... working on %s\n' % sample_name

			## parse calling cards and DE data
			if not parsed.optimized_labels: ## parse DE when optimized set isn't available
				file_de = parsed.de_dir +'/'+ sample_name +'.cuffdiff'
				cc_data, labels, cc_features = prepare_datasets(file_cc, file_de, 
									parsed.threshold_log2FC, parsed.threshold_adjustedP)
			else:
				cc_data, labels, cc_features, orfs = prepare_datasets_w_optimzed_labels(file_cc, optimized_labels[sample_name])

			# ## print label information
			# neg_labels = len(labels[labels==-1])
			# pos_labels = len(labels[labels==1])
			# total_labels = float(len(labels))
			# chance = (pos_labels/total_labels*pos_labels + neg_labels/total_labels*neg_labels)/ total_labels
			# print 'Bound not DE %d | Bound and DE %d | chance ACC: %.3f' % (neg_labels, pos_labels,chance)

			# ## rank features
			# indx_focus = range(0,cc_data.shape[1],2)
			# rank_binned_features(cc_features[indx_focus], cc_data[:,indx_focus], labels, method=parsed.ranking_method)
			# indx_focus = [i+1 for i in indx_focus]
			# rank_binned_features(cc_features[indx_focus], cc_data[:,indx_focus], labels, method=parsed.ranking_method)

			# print "5-fold corss-validation"
			# indx_focus = range(0,cc_data.shape[1],2)
			# rank_binned_features(cc_features[indx_focus], cc_data[:,indx_focus], labels, method=parsed.ranking_method, cv=5)
			# indx_focus = [i+1 for i in indx_focus]
			# rank_binned_features(cc_features[indx_focus], cc_data[:,indx_focus], labels, method=parsed.ranking_method, cv=5)

			## store data
			data_collection[sample_name] = {'cc_data': cc_data, 
											'labels': labels,
											'orfs': orfs}

		## combine samples
		data_collection = combine_samples(data_collection, len(cc_features))
		for sample in ['combined-plusLys', 'combined-minusLys', 'combined-all']:
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
			indx_focus = range(0,cc_data.shape[1],2)
			rank_binned_features(cc_features[indx_focus], cc_data[:,indx_focus], labels, method=parsed.ranking_method)
			indx_focus = [i+1 for i in indx_focus]
			rank_binned_features(cc_features[indx_focus], cc_data[:,indx_focus], labels, method=parsed.ranking_method)

			print "5-fold corss-validation"
			indx_focus = range(0,cc_data.shape[1],2)
			rank_binned_features(cc_features[indx_focus], cc_data[:,indx_focus], labels, method=parsed.ranking_method, cv=5)
			indx_focus = [i+1 for i in indx_focus]
			rank_binned_features(cc_features[indx_focus], cc_data[:,indx_focus], labels, method=parsed.ranking_method, cv=5)


	else:
		sys.exit("Wrong ranking method!")


if __name__ == "__main__":
    main(sys.argv)
