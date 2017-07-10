#!/usr/bin/python
import os
import sys
import numpy as np
import yaml

from sklearn import tree
from sklearn.preprocessing import scale
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import ttest_ind, mannwhitneyu

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from mlxtend.feature_selection import SequentialFeatureSelector as SFS


def rank_binned_features(features, X, y, method, cv=0, verbose=False):
	## use feature selection algorithms to rank features
	estimator = LogisticRegression(fit_intercept=False,class_weight='balanced')
	flag_forward = (method == "sequential_forward_selection")
	model = SFS(estimator=estimator, k_features=(1,10), forward=flag_forward, floating=True, scoring="accuracy", cv=cv)
	pipe = make_pipeline(StandardScaler(), model)
	pipe.fit(X, y)

	## rank features from most to least important
	feature_idx = []
	for k in sorted(model.subsets_): 
		feature_idx += list(np.setdiff1d(model.subsets_[k]['feature_idx'], feature_idx))
		if model.k_score_ == model.subsets_[k]["avg_score"]:
			break
	best_features = [features[i] for i in feature_idx]

	## get regression coeffs using the best feature combo
	estimator.fit(X[:,feature_idx], y)
	best_features_coef = estimator.coef_[0]

	## print best feature combo (sorted)
	sys.stdout.write("best combination (ACC: %.3f): %s\n" % (model.k_score_, ", ".join(["".join([best_features[i]," (","%.3f" % best_features_coef[i],")"]) for i in range(len(best_features))])))
	
	## print output of each selection iteration
	if verbose:
		for k in sorted(model.subsets_):
			v = model.subsets_[k]
			sys.stdout.write("%d\tscore=%.3f\tfeatures=%s\n" % (k, v["avg_score"], ", ".join([features[i] for i in v["feature_idx"]]))) 


def rank_linked_tree_features(X, y, features, orfs, sample_name, iteration):
	## use decision tree to classify orfs
	model = DecisionTreeClassifier(class_weight=None, max_depth=len(features))
	model.fit(X, y)
	print "Feature importance:"
	for i in range(len(features)):
		print "\t%s %.3f" % (features[i], model.feature_importances_[i])
	print "Model accuracy: %.1f%%" % (model.score(X, y)*100)

	## save tree object for visualization
	filename = "".join(['../output/tree_', sample_name, '_iter', str(iteration+1),'.dot'])
	tree.export_graphviz(model, out_file=filename, feature_names=features, 
						rounded=True, proportion=False, filled=True, node_ids=True, 
						class_names=np.array(model.classes_, dtype=str))
	# os.system('dot -Tpng ../output/tree.dot -o ../output/tree.png')
	
	## find misclassified orfs
	focused_orfs = []
	y_pred = model.predict(X)
	for i in range(len(orfs)):
		if y[i] != y_pred[i]:
			focused_orfs.append(orfs[i])
	return focused_orfs


def rank_highest_peaks_features(X, y, features, sample_name):
	## use decision tree to classify orfs
	model = DecisionTreeClassifier(class_weight=None, min_samples_leaf=3)
	model.fit(X, y)
	print "Feature importance:"
	for i in range(len(features)):
		print "\t%s %.3f" % (features[i], model.feature_importances_[i])
	print "Model accuracy: %.1f%%" % (model.score(X, y)*100)

	## save tree object for visualization
	filename = "".join(['../output/tree_', sample_name, '.dot'])
	tree.export_graphviz(model, out_file=filename, feature_names=features, 
						rounded=True, proportion=False, filled=True, node_ids=True, 
						class_names=np.array(model.classes_, dtype=str))
	# os.system('dot -Tpng ../output/tree.dot -o ../output/tree.png')


def prepare_datasets(file_cc, file_de, thld_lfc, thld_p):
	## load DE and CC data
	de = parse_de_matrix(file_de, thld_lfc, thld_p)
	cc, features = parse_cc_matrix(file_cc)

	## find common orfs (samples)
	common_orfs = np.sort(np.intersect1d(cc[:,0], de[:,0]))
	indx_cc = [np.where(cc[:,0] == orf)[0][0] for orf in common_orfs]
	indx_de = [np.where(de[:,0] == orf)[0][0] for orf in common_orfs]
	cc = np.array(cc[indx_cc, 1:], dtype=float)
	labels = np.array(de[indx_de, 1], dtype=int)

	return cc, labels, features


def prepare_datasets_w_optimzed_labels(file_cc, opt_labels):
	## parse labels of orfs
	cc, features = parse_cc_matrix(file_cc)
	labels = []
	for orf in opt_labels['bound']:
		if orf in opt_labels['intersection']:
			labels.append([orf, '1'])
		else:
			labels.append([orf, '-1'])
	labels = np.array(labels)

	## find common orfs (samples) for feature ranking
	common_orfs = np.sort(np.intersect1d(cc[:,0], labels[:,0]))
	indx_cc = [np.where(cc[:,0] == orf)[0][0] for orf in common_orfs]
	indx_labels = [np.where(labels[:,0] == orf)[0][0] for orf in common_orfs]
	cc = np.array(cc[indx_cc, 1:], dtype=float)
	labels = np.array(labels[indx_labels, 1], dtype=int)

	return cc, labels, features, common_orfs


def prepare_subset_w_optimized_labels(cc_dict, opt_labels, sample_name, iteration=0, focused_orfs=None):
	## parse labels of orfs
	labels = []
	for orf in opt_labels['bound']:
		if orf in opt_labels['intersection']:
			labels.append([orf, '1'])
		else:
			labels.append([orf, '-1'])
	labels = np.array(labels)

	## choose which orfs to use
	if iteration == 0:
		cc_orfs = np.array(cc_dict.keys())
	else:
		cc_orfs = []
		for sample, orf in focused_orfs:
			if sample == sample_name:
				cc_orfs.append(orf)
		cc_orfs = np.intersect1d(cc_orfs, cc_dict.keys())

	## get feature names and start appending cc data	
	features = np.array(cc_dict[cc_dict.keys()[0]]["1"].keys())
	cc_data = np.empty((0,len(features)))
	if len(cc_orfs) > 0:
		cc_orfs_w_valid_peaks = []
		for orf in cc_orfs:
			if iteration < len(cc_dict[orf]["Sorted_peaks"]): ## if have next highest peak
				cc_orfs_w_valid_peaks.append(orf)
				peak_indx = cc_dict[orf]["Sorted_peaks"][iteration]
				tmp_data = cc_dict[orf][peak_indx]
				tmp_data = np.array([tmp_data[x] for x in features])
				cc_data = np.vstack(( cc_data, tmp_data[np.newaxis] ))
		## find common orfs (samples) for building decision tree
		common_orfs = np.sort(np.intersect1d(cc_orfs_w_valid_peaks, labels[:,0]))
		indx_cc = [np.where(np.array(cc_orfs_w_valid_peaks) == orf)[0][0] for orf in common_orfs]
		indx_labels = [np.where(labels[:,0] == orf)[0][0] for orf in common_orfs]
		cc_data = np.array(cc_data[indx_cc, :], dtype=float)
		labels = np.array(labels[indx_labels, 1], dtype=int)
	else:
		labels = np.empty(0)
		common_orfs = np.empty(0)

	return cc_data, labels, features, common_orfs


def parse_de_matrix(file, thld_lfc, thld_p):
	data = np.loadtxt(file, dtype=str, skiprows=1, delimiter='\t', usecols=[1,9,12])
	de = []
	for i in range(len(data)):
		label = 1 if abs(float(data[i,1])) > thld_lfc and float(data[i,2]) < thld_p else -1
		de.append([data[i,0], label])
	return np.array(de)


def parse_cc_matrix(file):
	cc = np.loadtxt(file, dtype=str, delimiter='\t')
	f = open(file, 'r')
	features = np.array(f.readline().strip().split('\t')[1:])
	f.close()
	return cc, features


def parse_cc_json(file):
	with open(file, "r") as fp:
		cc_dict = yaml.safe_load(fp)
	return cc_dict


def parse_optimized_labels(file):
	opt_dict = {}
	data = np.loadtxt(file, dtype=str, skiprows=1, delimiter='\t')
	for i in range(len(data)):
		sample = data[i,0]
		intersection = [x.strip('"') for x in data[i,1].strip('{').strip('}').split(', ')]
		bound = [x.strip('"') for x in data[i,2].strip('{').strip('}').split(', ')]
		opt_dict[sample] = {'intersection': intersection, 'bound': bound}
	return opt_dict


def combine_samples(D, n_feat):
	combo_dict = {'plusLys': 'combined-plusLys', 
				'minusLys': 'combined-minusLys', 
				'':'combined-all'}
	## intialize dictionary
	D2 = {}
	for k,v in combo_dict.iteritems():
		D2[v] = {'cc_data': np.empty((0, n_feat)), 
				'labels': np.empty(0), 
				'orfs': []}
	## add data of the matching samples 
	for sample in D.keys():
		for k,v in combo_dict.iteritems():
			if sample.endswith(k):
				D2[v]['cc_data'] = np.vstack((D2[v]['cc_data'], D[sample]['cc_data']))
				D2[v]['labels'] = np.append(D2[v]['labels'], D[sample]['labels'])
				for orf in D[sample]['orfs']:
					D2[v]['orfs'].append((sample, orf))  ## add sample name
	D.update(D2) ## merge two dictionaries
	return D


def rank_features_RFE(features, X, y, estimator_type, cv=False):
	"""DEPRECATED"""
	## choose estimator type
	if estimator_type == 'SVM':
		estimator = SVC(kernel='linear', C=1, class_weight='balanced')
	elif estimator_type == 'LogisticRegression':
		estimator = LogisticRegression(fit_intercept=True,class_weight='balanced')
	else:
		sys.exit('Wrong estimator type')
	## cross validation or not
	if cv:
		model = RFECV(estimator, cv=cv, n_jobs=cv, step=1, verbose=0)
	else:
		model = RFE(estimator, n_features_to_select=1, step=1, verbose=0)
	model = model.fit(scale(X),y)
	indx = np.argsort(model.ranking_)
	## print feature ranking results
	print 'Bound not DE', len(y[y==-1]), '| Bound and DE', len(y[y==1])
	for i in indx:
		X_neg = X[y==-1, i]
		X_pos = X[y==1, i]
		# t,p = mannwhitneyu(X_neg, X_pos)
		# sys.stdout.write( "%s %.2f +/- %.2f | %.2f +/- %.2f | p=%.2e\n" % (features[i], np.median(X_neg), np.std(X_neg), np.median(X_pos), np.std(X_pos), p) )
		sys.stdout.write( "%s %.2f +/- %.2f | %.2f +/- %.2f\n" % (features[i], np.median(X_neg), np.std(X_neg), np.median(X_pos), np.std(X_pos)))


def rank_features_SFS(features, X, y):
	"""DEPRECATED"""
	estimator = LogisticRegression(fit_intercept=True,class_weight='balanced')
	model = SFS(estimator=estimator, k_features=(1,10), forward=True, floating=False, scoring="accuracy", cv=10)
	pipe = make_pipeline(StandardScaler(), model)
	pipe.fit(X, y)
	print('best combination (ACC: %.3f): %s\n' % (model.k_score_, ", ".join([features[i] for i in model.k_feature_idx_])))
	print('all subsets:\n', model.subsets_)
	for k in sorted(model.subsets_):
		v = model.subsets_[k]
		sys.stdout.write("%d\tscore=%.3f\tfeatures=%s\n" % (k, v["avg_score"], ", ".join([features[i] for i in v["feature_idx"]]))) 

