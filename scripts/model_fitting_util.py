#!/usr/bin/python
import os
import sys
import numpy as np
import yaml
import copy

from sklearn import tree
from sklearn.preprocessing import scale, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import explained_variance_score, r2_score, precision_recall_curve, average_precision_score
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold, train_test_split, RandomizedSearchCV
from sklearn.dummy import DummyRegressor
from bayes_opt import BayesianOptimization

from sklearn.pipeline import make_pipeline
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from scipy.stats import randint, uniform
from scipy.stats import ttest_ind, mannwhitneyu

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def sequential_rank_features(features, X, y, method, cv=0, verbose=False):
	## use feature selection algorithms to rank features
	# estimator = LogisticRegression(fit_intercept=False,
	# 								class_weight='balanced')
	hyparam = {"n_estimators": 20, 
				"max_depth": 3, 
				"min_samples_leaf": 5}
	estimator = RandomForestClassifier(n_estimators=hyparam["n_estimators"], 
										max_depth=hyparam["max_depth"],
										min_samples_leaf=hyparam["min_samples_leaf"],
										class_weight="balanced")
	flag_forward = (method == "sequential_forward_selection")
	model = SFS(estimator=estimator, k_features=(1,len(features)), forward=flag_forward, floating=True, scoring="average_precision", cv=cv, n_jobs=cv)
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
	# estimator.fit(X[:,feature_idx], y)
	# best_features_coef = estimator.coef_[0]
	## print best feature combo (sorted)
	# sys.stdout.write("best combination (ACC: %.3f): %s\n" % (model.k_score_, ", ".join(["".join([best_features[i]," (","%.3f" % best_features_coef[i],")"]) for i in range(len(best_features))])))

	##
	estimator.fit(X, y)
	print(features, ":", estimator.feature_importances_)
	sys.stdout.write("best combination (ACC: %.3f): %s\n" % (model.k_score_, ", ".join(["".join([best_features[i]]) for i in range(len(best_features))])))
	
	## print output of each selection iteration
	if verbose:
		for k in sorted(model.subsets_):
			v = model.subsets_[k]
			sys.stdout.write("%d\tscore=%.3f\tfeatures=%s\n" % (k, v["avg_score"], ", ".join([features[i] for i in v["feature_idx"]]))) 


def rank_linked_tree_features(X, y, features, orfs, sample_name, iteration):
	## use decision tree to classify orfs
	model = DecisionTreeClassifier(class_weight=None, max_depth=len(features))
	model.fit(X, y)
	print("Feature importance:")
	for i in range(len(features)):
		print("\t%s %.3f" % (features[i], model.feature_importances_[i]))
	print("Model accuracy: %.1f%%" % (model.score(X, y)*100))

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
	scaler = StandardScaler().fit(X)
	X = scaler.transform(X)
	model.fit(X, y)
	print("Feature importance:")
	for i in range(len(features)):
		print("\t%s %.3f" % (features[i], model.feature_importances_[i]))
	print("Model accuracy: %.1f%%" % (model.score(X, y)*100))

	## save tree object for visualization
	filename = "".join(['../output/tree_', sample_name, '.dot'])
	tree.export_graphviz(model, out_file=filename, feature_names=features, 
						rounded=True, proportion=False, filled=True, node_ids=True, 
						class_names=np.array(model.classes_, dtype=str))
	# os.system('dot -Tpng ../output/tree.dot -o ../output/tree.png')


def model_holdout_feature(X, y, features, sample_name, algorithm, is_classif, num_fold=10, num_step=20, opt_param=False, verbose=True):
	"""
	Use K-1 folds of samples to train a model, then use the holdout samples 
	to test the model. In testing, one feature will be varied within a range
	of values, while other features will be hold as they are. Thus the testing 
	accuracy is modeled as a function of the holdout feature.
	"""
	if is_classif:
		## define K folds for CV
		k_fold = StratifiedKFold(num_fold, shuffle=True, random_state=1)
		scores_test = {"accu":[], "sens":[], "spec":[], "prDE":[]}
		scores_holdout = {"accu":{}, "sens":{}, "spec":{}, "prDE":{}}
		features_var = {}

		## perform CV
		for k, (train, test) in enumerate(k_fold.split(X, y)):
			## preprocessing data 
			cv_scaler = StandardScaler().fit(X[train])
			X_tr = cv_scaler.transform(X[train])
			X_te = cv_scaler.transform(X[test])
			## construct model
			model = construct_classification_model(X_tr, y[train], algorithm, opt_param)
			## train the model
			model.fit(X_tr, y[train]) 
			## test without varying feature
			accu_te = model.score(X_te, y[test])
			sens_te, spec_te = cal_sens_n_spec(y[test], model.predict(X_te)) 
			scores_test["accu"].append(accu_te)
			scores_test["sens"].append(sens_te)
			scores_test["spec"].append(spec_te)
			scores_test["prDE"] += list(model.predict_proba(X_te)[:,1])
			if verbose:
				print("... cv fold %d" % k)
				print("   accu: %.3f\tsens: %.3f\tspec %.3f" % (accu_te, sens_te, spec_te))
				# pred_probs = model.predict_proba(X_te)
				# pred_class = model.predict(X_te)
				# for i in range(X_te.shape[0]):
				# 	print("   ", y[test][i], pred_probs[i,:], pred_class[i])
				# print("  ", np.unique(y[test], return_counts=True))

			for i in range(len(features)): 
				X_teho = X_te
				## vary the value of holdout feature
				# f_max = max(X[:,i])
				f_max = max(X[:,i]) if features[i].endswith("Dist") else np.percentile(X[:,i], 95)
				f_min = min(X[:,i])
				step = (f_max-f_min)/float(num_step) 
				feature_values = np.arange(f_min, f_max+step, step) 
				feature_values = (feature_values - cv_scaler.mean_[i]) / cv_scaler.scale_[i]

				accu_ho = []
				sens_ho = []
				spec_ho = []
				prDE_ho = np.empty((X_teho.shape[0],0))
				for v in feature_values:
					X_teho[:,i] = np.ones(X_teho.shape[0])*v
					accu_ho.append(model.score(X_teho, y[test]))
					sens_tmp, spec_tmp = cal_sens_n_spec(y[test], model.predict(X_teho))
					prDE_tmp = model.predict_proba(X_teho)[:,1]
					sens_ho.append(sens_tmp)
					spec_ho.append(spec_tmp)
					prDE_ho = np.hstack((prDE_ho, prDE_tmp[np.newaxis].T))
				## store accuracy metrics 
				try:	
					scores_holdout["accu"][features[i]].append(accu_ho)
					scores_holdout["sens"][features[i]].append(sens_ho)
					scores_holdout["spec"][features[i]].append(spec_ho)
					scores_holdout["prDE"][features[i]] = np.vstack((scores_holdout["prDE"][features[i]], prDE_ho))
				except KeyError:
					features_var[features[i]] = (feature_values * cv_scaler.scale_[i]) + cv_scaler.mean_[i] ## inversely transform back to original scale 
					scores_holdout["accu"][features[i]] = [accu_ho]
					scores_holdout["sens"][features[i]] = [sens_ho]
					scores_holdout["spec"][features[i]] = [spec_ho]
					scores_holdout["prDE"][features[i]] = prDE_ho
				# if verbose:
				# 	print("   %s\t%.3f\t%.3f\t%.3f" % (features[i], np.min(accu_ho), np.median(accu_ho), np.max(accu_ho)))
	
	else:
		## define K folds for CV
		k_fold = KFold(k, shuffle=True, random_state=1)
		scores_test = {"rsq":[], "lfc":[], "var_exp":[]}
		scores_holdout = {"rsq":{}, "lfc":{}, "var_exp":{}}
		features_var = {}

		## perform CV
		scores_train = []
		for k, (train, test) in enumerate(k_fold.split(X, y)):
			## preprocessing data
			cv_scaler = StandardScaler().fit(X[train])
			X_tr = cv_scaler.transform(X[train])
			X_te = cv_scaler.transform(X[test])
			## construct model 
			model = construct_regression_model(X_tr, y[train], algorithm)
			## train the model
			model.fit(X_tr, y[train])
			scores_train.append(model.score(X_tr, y[train]))
			## test without varying feature
			# rsq = model.score(X_te, y[test])
			y_pred = model.predict(X_te)
			rsq = r2_score(y[test], y_pred)
			var_exp = explained_variance_score(y[test], y_pred)
			scores_test["rsq"].append(rsq)
			scores_test["var_exp"].append(var_exp)
			scores_test["lfc"] += list(y_pred)

		print("R-squared: (max) %.3f, (min) %.3f, (median) %.3f" % (np.max(scores_train), np.min(scores_train), np.median(scores_train)))

		## TODO: add feature variation

	return (scores_test, scores_holdout, features_var)


def construct_classification_model(X, y, classifier, opt_param=False):
	"""
	classifier - chose from ["RandomForestClassifier", 
							"GradientBoostingClassifier"]
	opt_param - use Bayesian Optimization or Random Search to tune the hyperparameters
	"""
	if classifier == "RandomForestClassifier":
		if opt_param: 
			## hyperparameter opitimization
			use_BO = True
			hyparam = optimize_model_hyparam(X, y, classifier, use_BO)
			print(hyparam)
		else:
			hyparam = {"n_estimators": 20, 
						"max_depth": 3, 
						"min_samples_leaf": 5}
		## build model with desired hyperparameters
		model = RandomForestClassifier(n_estimators=hyparam["n_estimators"], 
										max_depth=hyparam["max_depth"],
										min_samples_leaf=hyparam["min_samples_leaf"],
										class_weight="balanced")

	elif classifier == "GradientBoostingClassifier":
		if opt_param: 
			## hyperparameter opitimization
			use_BO = True
			hyparam = optimize_model_hyparam(X, y, classifier, use_BO)
			print(hyparam)
		else:
			hyparam = {"n_estimators": 20, 
						"learning_rate": 0.05, 
						"subsample": 1.0,
						"min_samples_leaf": 1}
		## build model with desired hyperparameters
		model = GradientBoostingClassifier(n_estimators=hyparam["n_estimators"],
											learning_rate=hyparam["learning_rate"],
											subsample=hyparam["subsample"],
											min_samples_leaf=hyparam["min_samples_leaf"])

	else: 
		sys.exit(classifier +" not implemented yet!")
	
	## other potential classifier
	# model = KNeighborsClassifier() 
	# model = NuSVC(nu=.5, kernel="rbf")
	# model = LinearSVC()
	# model = AdaBoostClassifier(n_estimators=100, 
	# 							learning_rate=0.1)

	return model


def construct_regression_model(X, y, regressor):
	if regressor == "RidgeRegressor":
		model = RidgeCV()
	elif regressor == "KernelRidgeRegressor":
		model = KernelRidge(kernel="rbf")
	elif regressor == "RandomForestRegressor":
		model = RandomForestRegressor()
	elif regressor == "GradientBoostingRegressor":
		model = GradientBoostingRegressor()
	elif regressor == "GaussianProcessRegressor":
		model = GaussianProcessRegressor(n_restarts_optimizer=0)
	elif regressor == "MLPRegressor":
		model = MLPRegressor()
	else:
		sys.exit(regressor +" not implemented yet!")

	return model


def optimize_model_hyparam(X, y, classifier, use_BO):
	## define inner functions
	def rf_cv_score(n_estimators, max_depth, min_samples_leaf):
		cv_model = RandomForestClassifier(n_estimators=int(n_estimators), 
											max_depth=int(max_depth), 
											min_samples_leaf=int(min_samples_leaf),
											class_weight="balanced")
		cv_score = cross_val_score(cv_model, X, y, scoring="accuracy", 
									cv=5, n_jobs=5).mean()
		return cv_score

	def gb_cv_score(n_estimators, learning_rate, subsample, min_samples_leaf):
		cv_model = GradientBoostingClassifier(n_estimators=int(n_estimators), 
											learning_rate=learning_rate,
											subsample=subsample, 
											min_samples_leaf=int(min_samples_leaf))
		cv_score = cross_val_score(cv_model, X, y, scoring="accuracy", 
									cv=5, n_jobs=5).mean()
		return cv_score

	## tune model hyperparameters
	if classifier == "RandomForestClassifier":
		if use_BO:
			gp_params = {"alpha": 1e-5}
			hyparam_distr = {"n_estimators": (20,200),
							"max_depth": (1,20),
							"min_samples_leaf": (1,20)}
			model = BayesianOptimization(rf_cv_score, hyparam_distr, verbose=1)
			model.maximize(init_points=10, n_iter=25, **gp_params)
			hyparam = model.res["max"]["max_params"]
			for k in hyparam.keys():
				hyparam[k] = int(hyparam[k])
		else:
			hyparam_distr = {"n_estimators": range(20,201,20),
								"max_depth": randint(1,21),
								"min_samples_leaf": randint(1,11)}
			model = RandomizedSearchCV(RandomForestClassifier(), 
										param_distributions=hyparam_distr,
										n_iter=50, n_jobs=10)
			model.fit(X, y)
			hyparam = model.best_params_

	elif classifier == "GradientBoostingClassifier":
		if use_BO:
			gp_params = {"alpha": 1e-5}
			hyparam_distr = {"n_estimators": (20,200),
							"learning_rate": (0.005, 0.5),
							"subsample": (0.5, 1.0),
							"min_samples_leaf": (1,20)}
			model = BayesianOptimization(gb_cv_score, hyparam_distr, verbose=1)
			model.explore({"n_estimators": np.linspace(20,200,5),
							"learning_rate": 5*np.logspace(-4, -1, num=5, base=10),
							"subsample": np.linspace(0.5,1.0,5),
							"min_samples_leaf": np.linspace(1,20,5)})
			model.maximize(init_points=10, n_iter=25, **gp_params)
			hyparam = model.res["max"]["max_params"]
			for k in ["n_estimators", "min_samples_leaf"]:
				hyparam[k] = int(hyparam[k])
		else:
			hyparam_distr = {"n_estimators": range(20,201,20),
							"learning_rate": uniform(loc=0.001, scale=0.499),
							"subsample": uniform(loc=0.5, scale=0.5),
							"min_samples_leaf": randint(1,20)}
			model = RandomizedSearchCV(GradientBoostingClassifier(), 
										param_distributions=hyparam_distr,
										n_iter=50, n_jobs=10)
			model.fit(X, y)
			hyparam = model.best_params_

	else:
		sys.exit("Wrong classifier")
	return hyparam


def plot_holdout_features(scores_test, scores_holdout, features_var, feature_names, filename, metric="accu"):
	features = sorted(features_var.keys())
	# feature_names = sorted(features_var.keys())
	# if len(features_var)==7:
	# 	feature_names.insert(1,feature_names[4])
	# 	feature_names.pop(5)
	# 	feature_names.insert(2,feature_names[4])
	# 	feature_names.pop(5)
	# 	feature_names.insert(3,feature_names[4])
	# 	feature_names.pop(5)
	
	## define subplots
	fig = plt.figure(num=None, figsize=(10, 7), dpi=300)
	num_cols = np.ceil(np.sqrt(len(feature_names)))
	num_rows = np.ceil(len(feature_names)/num_cols)
	num_cols = int(num_cols)
	num_rows = int(num_rows)
	for i in range(num_rows):
		for j in range(num_cols):
			k = num_cols*i+j ## feature index
			if k < len(feature_names):
				if metric == "accu":
					accu_ho = np.array(scores_holdout["accu"][feature_names[k]])
					num_var = accu_ho.shape[1]
					## relative to testing acc
					accu_ho = accu_ho - np.repeat(np.array(scores_test["accu"])[np.newaxis].T, 
													num_var, axis=1) 
					accu_ho_med = np.median(accu_ho, axis=0)
					accu_ho_max = np.max(accu_ho, axis=0)
					accu_ho_min = np.min(accu_ho, axis=0)

					## make plots
					ax = fig.add_subplot(num_rows, num_cols, k+1)
					ax.fill_between(features_var[feature_names[k]], accu_ho_min, accu_ho_max, facecolor="blue", alpha=.25)
					ax.plot(features_var[feature_names[k]], accu_ho_med, 'k', linewidth=2)
					ax.set_title('%s' % feature_names[k])
					ax.set_ylim(-.5, .5)

				elif metric == "sens_n_spec":
					sens_ho = np.array(scores_holdout["sens"][feature_names[k]])
					spec_ho = np.array(scores_holdout["spec"][feature_names[k]])
					num_var = sens_ho.shape[1]
					## relative to testing acc
					sens_ho = sens_ho - np.repeat(np.array(scores_test["sens"])[np.newaxis].T, 
													num_var, axis=1)
					sens_ho_med = np.median(sens_ho, axis=0)
					sens_ho_max = np.max(sens_ho, axis=0)
					sens_ho_min = np.min(sens_ho, axis=0)
					spec_ho = spec_ho - np.repeat(np.array(scores_test["spec"])[np.newaxis].T, 
													num_var, axis=1) 
					spec_ho_med = np.median(spec_ho, axis=0)
					spec_ho_max = np.max(spec_ho, axis=0)
					spec_ho_min = np.min(spec_ho, axis=0)

					## make plots
					ax = fig.add_subplot(num_rows, num_cols, k+1)
					ax.fill_between(features_var[feature_names[k]], sens_ho_min, sens_ho_max, 
									facecolor="#336666", alpha=.25)
					ax.plot(features_var[feature_names[k]], sens_ho_med, "#336666", linewidth=2)
					ax.fill_between(features_var[feature_names[k]], spec_ho_min, spec_ho_max, 
									facecolor="#6d212d", alpha=.25)
					ax.plot(features_var[feature_names[k]], spec_ho_med, "#6d212d", linewidth=2)
					ax.set_title('%s' % feature_names[k])
					ax.set_ylim(-.5, .5)

				elif metric == "prob_DE":
					prDE_ho = np.array(scores_holdout["prDE"][feature_names[k]])
					num_var = prDE_ho.shape[1]
					## relative to testing acc
					prDE_ho_med = np.median(prDE_ho, axis=0)
					prDE_ho_max = np.max(prDE_ho, axis=0)
					prDE_ho_min = np.min(prDE_ho, axis=0)
					prDE_ho_pctl = [np.percentile(prDE_ho, 2.5, axis=0), 
									np.percentile(prDE_ho, 97.5, axis=0)]

					## make plots
					ax = fig.add_subplot(num_rows, num_cols, k+1)
					# ax.fill_between(features_var[feature_names[k]], prDE_ho_min, prDE_ho_max, facecolor="blue", alpha=.25)
					ax.fill_between(features_var[feature_names[k]], prDE_ho_pctl[0], prDE_ho_pctl[1], facecolor="#0066cc", alpha=.25)
					ax.plot(features_var[feature_names[k]], prDE_ho_med, "#0066cc", linewidth=2)
					ax.set_title('%s' % feature_names[k])
					ax.set_ylim(0, 1)

				elif metric == "rel_prob_DE":
					## calculate relative prob of predicting DE
					prDE_ho = np.array(scores_holdout["prDE"][feature_names[k]])
					num_var = prDE_ho.shape[1]
					prDE_ho_sub = prDE_ho - np.repeat(np.array(scores_test["prDE"])[np.newaxis].T, num_var, axis=1)
					## relative to testing acc 
					prDE_ho_med_pos, prDE_ho_med_neg = [], []
					prDE_ho_max_pos, prDE_ho_max_neg = [], []
					prDE_ho_min_pos, prDE_ho_min_neg = [], []
					prDE_ho_pctl_pos, prDE_ho_pctl_neg = [[],[]], [[],[]]
					for l in range(prDE_ho.shape[1]):
						prDE_ho_pos = prDE_ho_sub[:,l][prDE_ho[:,l] > .5]
						prDE_ho_neg = prDE_ho_sub[:,l][prDE_ho[:,l] < .5]
						if len(prDE_ho_pos) > 0:
							prDE_ho_med_pos.append(np.median(prDE_ho_pos))
							prDE_ho_max_pos.append(np.max(prDE_ho_pos))
							prDE_ho_min_pos.append(np.min(prDE_ho_pos))
							prDE_ho_pctl_pos[0].append(np.percentile(prDE_ho_pos, 2.5))
							prDE_ho_pctl_pos[1].append(np.percentile(prDE_ho_pos, 97.5))
						else:
							prDE_ho_med_pos.append(np.nan)
							prDE_ho_max_pos.append(np.nan)
							prDE_ho_min_pos.append(np.nan)
							prDE_ho_pctl_pos[0].append(np.nan)
							prDE_ho_pctl_pos[1].append(np.nan)
						if len(prDE_ho_neg) > 0:
							prDE_ho_med_neg.append(np.median(prDE_ho_neg))
							prDE_ho_max_neg.append(np.max(prDE_ho_neg))
							prDE_ho_min_neg.append(np.min(prDE_ho_neg))
							prDE_ho_pctl_neg[0].append(np.percentile(prDE_ho_neg, 2.5))
							prDE_ho_pctl_neg[1].append(np.percentile(prDE_ho_neg, 97.5))
						else:
							prDE_ho_med_neg.append(np.nan)
							prDE_ho_max_neg.append(np.nan)
							prDE_ho_min_neg.append(np.nan)
							prDE_ho_pctl_neg[0].append(np.nan)
							prDE_ho_pctl_neg[1].append(np.nan)
					## make plots
					ax = fig.add_subplot(num_rows, num_cols, k+1)
					ax.fill_between(features_var[feature_names[k]], prDE_ho_pctl_pos[0], prDE_ho_pctl_pos[1], facecolor="#0066cc", alpha=.25)
					ax.fill_between(features_var[feature_names[k]], prDE_ho_pctl_neg[0], prDE_ho_pctl_neg[1], facecolor="#FFA500", alpha=.25)
					ax.plot(features_var[feature_names[k]], prDE_ho_med_pos, "#0066cc", linewidth=2)
					ax.plot(features_var[feature_names[k]], prDE_ho_med_neg, "#FFA500", linewidth=2)
					ax.set_title('%s' % feature_names[k])
					ax.set_ylim(-.5, .5)

				else:
					sys.exit("No such metric to plot")
	plt.tight_layout()
	plt.savefig(''.join([filename,'.',metric,'.pdf']), format='pdf')


def model_interactive_feature(X, y, algorithm, num_fold=10, opt_param=False):
	results = []
	num_rnd_permu = 80
	## define training, testing split
	X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=1./num_fold, random_state=1)
	y_all_tr = np.empty(0)
	y_pred_prob = np.empty(0)
	y_rnd_prob = {}
	for i in range(num_rnd_permu):
		y_rnd_prob[i] = np.empty(0)
	## perform CV
	k_fold = StratifiedKFold(num_fold, shuffle=True, random_state=1)
	sys.stderr.write("... cv: ") 
	for k, (cv_tr, cv_te) in enumerate(k_fold.split(X_tr, y_tr)):
		## preprocessing data
		cv_scaler = StandardScaler().fit(X_tr[cv_tr])
		X_cv_tr = cv_scaler.transform(X_tr[cv_tr])
		X_cv_te = cv_scaler.transform(X_tr[cv_te])
		## construct model
		sys.stderr.write("%d " % k) 
		model = construct_classification_model(X_cv_tr, y_tr[cv_tr], algorithm, opt_param)
		model.fit(X_cv_tr, y_tr[cv_tr]) 
		## internal validation 
		de_class_indx = np.where(model.classes_ == 1)[0][0]
		y_all_tr = np.append(y_all_tr, y_tr[cv_te])
		y_pred_prob = np.append(y_pred_prob, model.predict_proba(X_cv_te)[:,de_class_indx])
		for i in range(num_rnd_permu):
			## only permute the last column
			rnd_X_tr_cv_te = copy.deepcopy(X_cv_te) 
			col_permu = rnd_X_tr_cv_te.shape[1]-1
			rnd_X_tr_cv_te[:,col_permu] = np.random.permutation(rnd_X_tr_cv_te[:,col_permu])
			y_rnd_prob[i] = np.append(y_rnd_prob[i], model.predict_proba(rnd_X_tr_cv_te)[:,de_class_indx]) 
			## randomly permute each column
			# y_rnd_prob[i] = np.append(y_rnd_prob[i], model.predict_proba(X_cv_te)[:,de_class_indx]) 
	## calculate AUPRs
	aupr_pred = average_precision_score(y_all_tr, y_pred_prob)
	aupr_rnd = sorted([average_precision_score(y_all_tr, y_rnd_prob[i]) for i in y_rnd_prob.keys()])
	print("\nCV AUPR = %.3f" % aupr_pred)
	print("CV Randomized, median AUPR = %.3f, 95%% CI = [%.3f, %.3f]" % (np.median(aupr_rnd), aupr_rnd[2], aupr_rnd[-3]))
	results += [np.median(aupr_rnd), aupr_rnd[2], aupr_rnd[-3], aupr_rnd[-3]-np.median(aupr_rnd), aupr_pred]

	## preprocessing data
	scaler = StandardScaler().fit(X_tr)
	X_tr = cv_scaler.transform(X_tr)
	X_te = cv_scaler.transform(X_te)
	## model trained with full training set
	model = construct_classification_model(X_tr, y_tr, algorithm, opt_param)
	model.fit(X_tr, y_tr) 
	de_class_indx = np.where(model.classes_ == 1)[0][0]
	## predict with the leftout testing set
	y_pred_prob = model.predict_proba(X_te)[:,de_class_indx] ## overwriting
	for i in range(num_rnd_permu):
		y_rnd_prob[i] = model.predict_proba(np.random.permutation(X_te))[:,de_class_indx] ## overwriting
	## calculate AUPRs
	aupr_pred = average_precision_score(y_te, y_pred_prob)
	aupr_rnd = sorted([average_precision_score(y_te, y_rnd_prob[i]) for i in y_rnd_prob.keys()])
	print("\nTesting AUPR = %.3f" % aupr_pred)
	print("Testing Randomized, median AUPR = %.3f, 95%% CI = [%.3f, %.3f]" % (np.median(aupr_rnd), aupr_rnd[2], aupr_rnd[-3]))
	results += [np.median(aupr_rnd), aupr_rnd[2], aupr_rnd[-3], aupr_rnd[-3]-np.median(aupr_rnd), aupr_pred]	

	## TODO: return model trained with full training set and X_te, y_te
	return results


def calculate_precision_recall(X0, y0, DE_p_thld=0.1):
	## treat CC signal as probability of predicting DE
	X0 = np.ndarray.flatten(X0)
	X = X0/float(np.max(X0))
	## convert to categorical labels
	y = -1*np.ones(len(y0))
	y[y0 < DE_p_thld] = 1
	precision, recall, _ = precision_recall_curve(y, X)
	aupr = average_precision_score(y, X)
	## special headling on the first actual prediction
	# precision = precision[:-1]
	# recall = recall[:-1]
	# if y[np.argmax(X)] == 1:
	# 	precision[-1] = 1
	# 	recall[-1] = 2./(len(y)+1)
	# else:
	# 	precision[-1] = 1./2
	# 	recall[-1] = 1./(len(y)+1)
	return precision, recall, aupr


def plot_precision_recall_w_random_signal(cc_data, labels, DE_p_thld, figname):
	## check zero-length set
	if len(labels) == 0:
		return [np.nan]*5
	num_rnd_permu = 80
	## true signals
	precision, recall, aupr = calculate_precision_recall(cc_data, labels, DE_p_thld)
	print("AUPR = %.3f" % aupr)
	## randomly permute signals
	precision_rnd = []
	recall_rnd = []
	aupr_rnd = []
	for i in range(num_rnd_permu):
		tmp_pr, tmp_re, tmp_aupr = calculate_precision_recall(np.random.permutation(cc_data), labels, DE_p_thld)
		precision_rnd.append(tmp_pr)
		recall_rnd.append(tmp_re)
		aupr_rnd.append(tmp_aupr)
	aupr_rnd = sorted(aupr_rnd)
	## fix different PR length issue from randomized data
	max_length = max([len(tmp_pr) for tmp_pr in precision_rnd])
	for i in range(len(precision_rnd)):
		tmp_pr = precision_rnd[i]
		tmp_re = recall_rnd[i]
		precision_rnd[i] = np.append(np.array([tmp_pr[0]]*(max_length-len(tmp_pr))), tmp_pr)
		recall_rnd[i] = np.append(np.array([tmp_re[0]]*(max_length-len(tmp_re))), tmp_re)
	print("Randomized, median AUPR = %.3f, 95%% CI = [%.3f, %.3f]" % (np.median(aupr_rnd), aupr_rnd[2], aupr_rnd[-3]))
	results = [np.median(aupr_rnd), aupr_rnd[2], aupr_rnd[-3], aupr_rnd[-3]-np.median(aupr_rnd), aupr]

	## make plot
	if figname:
		plt.figure(num=None, figsize=(6, 5), dpi=300)
		plt.plot(recall, precision, color="blue", lw=2,label="experiment (AUPR=%.3f)" % aupr)
		plt.fill_between(np.median(np.array(recall_rnd),axis=0), np.min(np.array(precision_rnd),axis=0), np.max(np.array(precision_rnd),axis=0), color="lightgrey", lw=1,label="random (100 runs\nmedian AUPR=%.3f)" % np.median(aupr_rnd))
		## set format
		plt.xscale('log')
		plt.xlabel('recall'); plt.ylabel('precision')
		plt.xlim(0,1); plt.ylim(0,1)
		leg = plt.legend()
		leg.get_frame().set_edgecolor('k')
		plt.savefig(figname, fmt='pdf')
	return results


def cal_sens_n_spec(y, y_pred):
	tp, fp, fn, tn = 0,0,0,0
	for i in range(len(y)):
		if y[i] == 1 and y_pred[i] == 1:
			tp += 1
		elif y[i] == -1 and y_pred[i] == 1:
			fp += 1
		elif y[i] == 1 and y_pred[i] == -1:
			fn += 1
		elif y[i] == -1 and y_pred[i] == -1:
			tn += 1
	return (float(tp)/(tp+fn), float(tn)/(tn+fp))


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
	cc[np.isnan(cc)] = 0
	labels = np.array(labels[indx_labels, 1], dtype=int)

	return cc, labels, features, common_orfs


def prepare_datasets_w_de_labels(file_cc, file_label, label_type, p_cutoff=.1):
	## parse labels of orfs
	cc, features = parse_cc_matrix(file_cc)
	if label_type == "logfc":
		orfs, labels = parse_de_matrix(file_label)
		labels = np.absolute(labels)
	else:
		orfs, labels = parse_de_matrix(file_label, False)
		labels[labels > p_cutoff] = -1
		labels[labels >= 0] = 1

	## find common orfs (samples) for feature ranking
	cc_out = []
	for i in range(len(orfs)):
		orf = orfs[i]
		if orf in cc[:,0]:
			indx = np.where(cc[:,0] == orf)[0][0]
			cc_out.append(list(np.array(cc[indx,1:], dtype=float)))
		else:
			cc_out.append([0.]*len(features))
	cc_out = np.array(cc_out, dtype=float)
	cc_out[np.isnan(cc_out)] = 0
	return cc_out, labels, features, orfs


def parse_de_matrix(file, output_lfc=False, p_cutoff=1., label_bound=15):
	## load data
	data = np.loadtxt(file, dtype=str, delimiter='\t')
	orfs = data[:,0]
	pvals = np.array(data[:,1], dtype=float)
	lfcs = np.array(data[:,2], dtype=float)
	## get proper systematic name
	valid = [i for i in range(len(orfs)) if orfs[i].startswith("Y") and orfs[i].split('-')[0][-1] in ['W','C']] 
	orfs = orfs[valid]
	pvals = pvals[valid]
	lfcs = lfcs[valid]
	## set logFC to zero if having insignificant p value
	lfcs[pvals > p_cutoff] = 0
	## set upper and lower bound on logFC
	lfcs[lfcs > label_bound] = label_bound
	lfcs[lfcs < -1*label_bound] = -1*label_bound
	## sort by systematic orf name
	indx = np.argsort(orfs)
	if output_lfc:
		return orfs[indx], lfcs[indx]
	else:
		return orfs[indx], pvals[indx]


def parse_cc_matrix(file):
	## load data
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


def process_data_collection(files_cc, file_labels, valid_sample_names, label_type, flag_comb_samples=True, p_cutoff=.1):
	data_collection = {}
	for file_cc in files_cc: 
		sn = os.path.splitext(os.path.basename(file_cc))[0].split(".")[0]
		if (sn in valid_sample_names) and (not sn.startswith('BY4741')):## exclude WT samples
			print('... loading %s' % sn)
			if label_type == "categorical":
				cc_data, labels, cc_features, orfs = prepare_datasets_w_optimzed_labels(file_cc, file_labels[sn])
			elif label_type == "continuous":
				file_label = file_labels[[k for k in range(len(file_labels)) if os.path.basename(file_labels[k]).split('.')[0] == sn][0]] ## find the continuous label file that matches CC file
				cc_data, labels, cc_features, orfs = prepare_datasets_w_de_labels(file_cc, file_label, "logfc")
			elif label_type == "conti2categ":
				file_label = file_labels[[k for k in range(len(file_labels)) if os.path.basename(file_labels[k]).split('.')[0] == sn][0]] ## find the continuous label file that matches CC file
				cc_data, labels, cc_features, orfs = prepare_datasets_w_de_labels(file_cc, file_label, "pval", p_cutoff)
			else:
				sys.exit("No label type given!")
			data_collection[sn] = {'cc_data': cc_data, 
									'labels': labels,
									'orfs': orfs}
	## combine samples
	if flag_comb_samples:
		data_collection = combine_samples(data_collection, len(cc_features))
	return data_collection, cc_features


def query_data_collection(data_collection, query_sample_name, cc_features, feature_prefix=None):
	## make query
	print('-----------------------------\n... working on %s\n' % query_sample_name)
	labels = data_collection[query_sample_name]['labels']
	cc_data = data_collection[query_sample_name]['cc_data']
	orfs = data_collection[query_sample_name]['orfs']
	## filter features by prefix
	if feature_prefix:
		if type(feature_prefix) is str:
			fp = feature_prefix
			indx = [i for i in range(len(cc_features)) if cc_features[i].startswith(fp)]
		elif type(feature_prefix) is list:
			indx = []
			for fp in feature_prefix:
				indx += [i for i in range(len(cc_features)) if cc_features[i].startswith(fp)]
		cc_features = cc_features[indx]
		cc_data = cc_data[:,indx]
	return (labels, cc_data, cc_features)


def calculate_chance(labels, verbose=True):
	neg_labels = len(labels[labels==-1])
	pos_labels = len(labels[labels==1])
	total_labels = float(len(labels))
	chance = (pos_labels/total_labels*pos_labels + neg_labels/total_labels*neg_labels)/ total_labels
	if verbose:
		print('Bound not DE %d | Bound and DE %d | chance ACC: %.3f' % (neg_labels, pos_labels,chance))
	return chance


def calculate_dummy_regression_score(X, y):
	model = DummyRegressor()
	model.fit(X, y)
	return model.score(X, y)


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
	print('Bound not DE', len(y[y==-1]), '| Bound and DE', len(y[y==1]))
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


def prepare_subset_w_optimized_labels(cc_dict, opt_labels, sample_name, iteration=0, focused_orfs=None):
	"""DEPRECATED"""
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


