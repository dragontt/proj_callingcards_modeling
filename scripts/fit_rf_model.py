#!/usr/bin/python
from __future__ import print_function
import os
import sys
import numpy as np
import argparse
import glob
import copy
from model_fitting_util import *
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score


def parse_args(argv):
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-t","--feature_type", 
                        help="choose from ['highest_peaks', 'binned_promoter']")
    parser.add_argument("-c","--cc_dir", help="Calling Cards feature directory")
    parser.add_argument("-d","--de_dir", help="Differential expression directory")
    parser.add_argument("-p","--bp_dir", help="Binding potential directory")
    parser.add_argument("-a","--file_ca", help="Chromatin accessibility file")
    parser.add_argument("-v","--valid_sample_name", help="TF sample name to focus")
    parser.add_argument("--weighted_class", action="store_true", default=False)
    parser.add_argument("--balanced_trainset", action="store_true", default=False)
    parser.add_argument("-o","--output_filename")
    parsed = parser.parse_args(argv[1:])
    return parsed


def prepare_data(parsed, cc_feature_filtering_prefix="logrph", shuffle_sample=True):
    if parsed.de_dir: ## parse valid DE samples if available
        files_de = glob.glob(parsed.de_dir +"/*15min.DE.txt")
        if parsed.valid_sample_name:
            valid_sample_names = [parsed.valid_sample_name]
        else:
            #valid_sample_names = [os.path.basename(f).split('-')[0] for f in files_de]
            valid_sample_names = ['YLR451W', 'YKL038W', 'YDR034C']
        label_type = "continuous"
    else:
        sys.exit("Require the label directory: optimized subset file or DE folder!") 

    ## parse input
    label_type = "conti2top5pct"
    files_cc = glob.glob(parsed.cc_dir +"/*.cc_feature_matrix."+ parsed.feature_type +".txt")
    cc_data_collection, cc_features = process_data_collection(files_cc, files_de,
                                            valid_sample_names, label_type, False)

    if parsed.file_ca is not None:
        ca_data, _, ca_features, _ = prepare_datasets_w_de_labels(parsed.file_ca, files_de[0], "pval", 0.1)
    
    ## query samples
    # cc_feature_filtering_prefix = "logrph"
    # cc_feature_filtering_prefix = "logrph_total"
    bp_feature_filtering_prefix = ["sum_score", "count", "dist"]
    ca_feature_filtering_prefix = ['H3K27ac_prom_-1','H3K36me3_prom_-1',
                                    'H3K4me3_prom_-1','H3K79me_prom_-1',
                                    'H4K16ac_prom_-1','H3K27ac_body',
                                    'H3K36me3_body','H3K4me3_body',
                                    'H3K79me_body','H4K16ac_body']
    print("... querying data")
    if parsed.file_ca is None:
        if cc_feature_filtering_prefix.endswith("total"):
            combined_data = np.empty((0,1))
        else:
            combined_data = np.empty((0,160))
    else:
        if cc_feature_filtering_prefix.endswith("total"):
            combined_data = np.empty((0,11))
        else:
            combined_data = np.empty((0,170))
    combined_labels = np.empty(0)

    for sample_name in sorted(cc_data_collection):
        labels, cc_data, _ = query_data_collection(cc_data_collection, sample_name,
                                        cc_features, cc_feature_filtering_prefix)
        if cc_feature_filtering_prefix.endswith("total"):
            cc_data = cc_data[:,-1].reshape(-1,1)
        else:
            cc_data = cc_data[:,:-1]
        ## only use data with non-zero signals
        mask = [np.any(cc_data[k,] != 0) for k in range(cc_data.shape[0])]
        ## TODO: unmask -- use all samples
        #mask = np.arange(cc_data.shape[0])

        ## add histone marks
        if parsed.file_ca is not None:
            ca_feat_indx = [k for k in range(len(ca_features)) if ca_features[k] in ca_feature_filtering_prefix]
            cc_data = np.concatenate((cc_data, ca_data[:,ca_feat_indx]), axis=1)

        combined_data = np.vstack((combined_data, cc_data[mask,]))
        combined_labels = np.append(combined_labels, labels[mask])
    print(combined_data.shape, "+1:", sum(combined_labels == 1), "-1:", sum(combined_labels == -1))
    combined_labels = np.array(combined_labels, dtype=int)
    ## convert to 0/1 labeling 
    combined_labels[combined_labels == -1] = 0

    if shuffle_sample:
        indx_rand = np.random.permutation(len(combined_labels))
        combined_data, combined_labels = combined_data[indx_rand,], combined_labels[indx_rand]

    return (combined_data, combined_labels)


def cross_validate_model(X, y, algorithm, split_te_set=True, num_fold=10, opt_param=False, json_filename=None):

    ## TODO: dict to store simulated results
    sim_dict = {"cc_high":{}, "cc_low":{}, "hm_high":{}, "hm_low":{}}


    results = []
    num_rnd_permu = 80
    ## define training, testing split
    if split_te_set:
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=1./num_fold, random_state=1)
        print(len(y_tr[y_tr == 1]) / float(len(y_tr)), len(y_te[y_te == 1]) / float(len(y_te)))
    else:
        X_tr, y_tr = X, y
    ## perform CV
    y_all_tr = np.empty(0)
    y_pred_prob = np.empty(0)
    combined_auprcs = np.empty(0)
    y_rnd_prob = {}
    for i in range(num_rnd_permu):
        y_rnd_prob[i] = np.empty(0)

    k_fold = StratifiedKFold(num_fold, shuffle=True, random_state=1)
    sys.stderr.write("... cv: ") 
    for k, (cv_tr, cv_te) in enumerate(k_fold.split(X_tr, y_tr)):
        X_cv_tr = X_tr[cv_tr]
        y_cv_tr = y_tr[cv_tr]
        
        ## preprocessing data
        cv_scaler = StandardScaler().fit(X_cv_tr)
        X_cv_tr = cv_scaler.transform(X_cv_tr)
        X_cv_te = cv_scaler.transform(X_tr[cv_te])
        ## construct model
        sys.stderr.write("%d " % k)
        model = construct_classification_model(X_cv_tr, y_cv_tr, algorithm, opt_param)
        model.fit(X_cv_tr, y_cv_tr) 
        ## TODO: to be removed
        de_class_indx = np.where(model.classes_ == 1)[0][0]
        tr_y_pred_prob = model.predict_proba(X_cv_tr)[:,de_class_indx]
        ## internal validation 
        de_class_indx = np.where(model.classes_ == 1)[0][0]
        y_all_tr = np.append(y_all_tr, y_tr[cv_te])
        cv_pred_prob = model.predict_proba(X_cv_te)[:,de_class_indx]
        y_pred_prob = np.append(y_pred_prob, cv_pred_prob)

        auprc_te = 100*average_precision_score(y_tr[cv_te], cv_pred_prob)
        combined_auprcs = np.append(combined_auprcs, auprc_te)
        print('validation AuPRC: %.2f%%' % auprc_te)
        ## TODO: changed to rankings of predicted probs
        # y_pred_prob = np.append(y_pred_prob, rankdata(cv_pred_prob))

        for i in range(num_rnd_permu):
            ## randomly permute each column
            rnd_X_tr_cv_te = copy.deepcopy(X_cv_te) 
            for j in range(rnd_X_tr_cv_te.shape[1]):
                rnd_X_tr_cv_te[:,j] = np.random.permutation(rnd_X_tr_cv_te[:,j])
            y_rnd_prob[i] = np.append(y_rnd_prob[i], model.predict_proba(rnd_X_tr_cv_te)[:,de_class_indx]) 



        ## TODO: test on perturbed feature values
        auprc_loss_pert = []
        conf_change_pert = []
        for findx in range(0, 170):
            X_pert = np.copy(X_cv_te)
            X_pert[:,findx] = np.max(X_cv_te[:,findx])
            pert_prob = model.predict_proba(X_pert)[:,de_class_indx]
            auprc_pert = 100*average_precision_score(y_tr[cv_te], pert_prob)
            auprc_loss_pert.append(auprc_pert - auprc_te)
            conf_change_pert.append(np.mean(pert_prob - cv_pred_prob))
            # print('%d\thigh\t%.2f%%' % (findx, auprc_pert))
        sim_dict["cc_high"][k] = {
                    "indx": np.argsort(auprc_loss_pert[:160])[:20].tolist(),
                    "loss": np.sort(auprc_loss_pert[:160])[:20].tolist(),
                    "indx_conf": np.argsort(conf_change_pert[:160])[::-1][:20].tolist(),
                    "conf_change": np.sort(conf_change_pert[:160])[::-1][:20].tolist()
                    }
        sim_dict["hm_high"][k] = {
                    "indx": np.argsort(auprc_loss_pert[160:]).tolist(),
                    "loss": np.sort(auprc_loss_pert[160:]).tolist(),
                    "indx_conf": np.argsort(conf_change_pert[160:])[::-1].tolist(),
                    "conf_change": np.sort(conf_change_pert[160:])[::-1].tolist()
                    
                    }

        auprc_loss_pert = []
        conf_change_pert = []
        for findx in range(0, 170): 
            X_pert = np.copy(X_cv_te)
            X_pert[:,findx] = np.min(X_cv_te[:,findx])
            pert_prob = model.predict_proba(X_pert)[:,de_class_indx]
            auprc_pert = 100*average_precision_score(y_tr[cv_te], pert_prob)
            auprc_loss_pert.append(auprc_pert - auprc_te)
            conf_change_pert.append(np.mean(pert_prob - cv_pred_prob))
            # print('%d\tlow\t%.2f%%' % (findx, auprc_pert))
        sim_dict["cc_low"][k] = {
                    "indx": np.argsort(auprc_loss_pert[:160])[:20].tolist(),
                    "loss": np.sort(auprc_loss_pert[:160])[:20].tolist(),
                    "indx_conf": np.argsort(conf_change_pert[:160])[:20].tolist(),
                    "conf_change": np.sort(conf_change_pert[:160])[:20].tolist()
                    }
        sim_dict["hm_low"][k] = {
                    "indx": np.argsort(auprc_loss_pert[160:]).tolist(),
                    "loss": np.sort(auprc_loss_pert[160:]).tolist(),
                    "indx_conf": np.argsort(conf_change_pert[160:])[::-1].tolist(),
                    "conf_change": np.sort(conf_change_pert[160:])[::-1].tolist()
                    }


        
    ## TODO: save dict
    import json
    with open(json_filename, "w") as fp:
        json.dump(sim_dict, fp, sort_keys=True, indent=4)



    ## calculate AUPRs
    results = np.hstack((y_all_tr.reshape(-1,1), y_pred_prob.reshape(-1,1)))
    print('$$ Average AuPRCs: %.2f%%, Overall AuPRC: %.2f%%' % (np.mean(combined_auprcs), 100* average_precision_score(results[:,0], results[:,1])))

    if split_te_set:
        ## preprocessing data
        scaler = StandardScaler().fit(X_tr)
        X_tr = cv_scaler.transform(X_tr)
        X_te = cv_scaler.transform(X_te)
        ## model trained with full training set
        model = construct_classification_model(X_tr, y_tr, algorithm, opt_param)
        model.fit(X_tr, y_tr) 
        de_class_indx = np.where(model.classes_ == 1)[0][0]
        ## predict the leftout testing set
        y_pred_prob = model.predict_proba(X_te)[:,de_class_indx] ## overwriting
        for i in range(num_rnd_permu):
            y_rnd_prob[i] = model.predict_proba(np.random.permutation(X_te))[:,de_class_indx] ## overwriting
        ## calculate AUPRs
        aupr_pred = average_precision_score(y_te, y_pred_prob)
        aupr_rnd = sorted([average_precision_score(y_te, y_rnd_prob[i]) for i in y_rnd_prob.keys()])
        print("\nTesting AUPR = %.3f" % aupr_pred)
        print("Testing Randomized, median AUPR = %.3f, 95%% CI = [%.3f, %.3f]" % (np.median(aupr_rnd), aupr_rnd[2], aupr_rnd[-3]))
        results += [np.median(aupr_rnd), aupr_rnd[2], aupr_rnd[-3], aupr_rnd[-3]-np.median(aupr_rnd), aupr_pred]    

    return results


def prepare_coarse_data(parsed, cc_feature_filtering_prefix="logrph", shuffle_sample=True):
    if parsed.de_dir: ## parse valid DE samples if available
        files_de = glob.glob(parsed.de_dir +"/*15min.DE.txt")
        if parsed.valid_sample_name:
            valid_sample_names = [parsed.valid_sample_name]
        else:
            #valid_sample_names = [os.path.basename(f).split('-')[0] for f in files_de]
            valid_sample_names = ['YLR451W', 'YKL038W', 'YDR034C']
        label_type = "continuous"
    else:
        sys.exit("Require the label directory: optimized subset file or DE folder!") 

    ## parse input
    label_type = "conti2top5pct"
    files_cc = glob.glob(parsed.cc_dir +"/*.cc_feature_matrix."+ parsed.feature_type +".txt")
    cc_data_collection, cc_features = process_data_collection(files_cc, files_de,
                                            valid_sample_names, label_type, False)

    if parsed.file_ca is not None:
        ca_data, _, ca_features, _ = prepare_datasets_w_de_labels(parsed.file_ca, files_de[0], "pval", 0.1)
    
    ## query samples
    # cc_feature_filtering_prefix = "logrph"
    # cc_feature_filtering_prefix = "logrph_total"
    bp_feature_filtering_prefix = ["sum_score", "count", "dist"]
    ca_feature_filtering_prefix = ['H3K27ac_prom_-1','H3K36me3_prom_-1',
                                    'H3K4me3_prom_-1','H3K79me_prom_-1',
                                    'H4K16ac_prom_-1','H3K27ac_body',
                                    'H3K36me3_body','H3K4me3_body',
                                    'H3K79me_body','H4K16ac_body']
    print("... querying data")
    if parsed.file_ca is None:
        if cc_feature_filtering_prefix.endswith("total"):
            combined_data = np.empty((0,1))
        else:
            combined_data = np.empty((0,14))
    else:
        if cc_feature_filtering_prefix.endswith("total"):
            combined_data = np.empty((0,11))
        else:
            combined_data = np.empty((0,14))
    combined_labels = np.empty(0)

    for sample_name in sorted(cc_data_collection):
        labels, cc_data, _ = query_data_collection(cc_data_collection, sample_name,
                                        cc_features, cc_feature_filtering_prefix)
        if cc_feature_filtering_prefix.endswith("total"):
            cc_data = cc_data[:,-1].reshape(-1,1)
        else:
            cc_data = cc_data[:,:-1]
            tmp_cc_data = cc_data.copy()
            cc_data = np.zeros((tmp_cc_data.shape[0], 4)) 
            cc_data[:,0] = np.sum(tmp_cc_data[:,:30], axis=1) ## 800-650bp
            cc_data[:,1] = np.sum(tmp_cc_data[:,30:70], axis=1) ## 650-450bp
            cc_data[:,2] = np.sum(tmp_cc_data[:,70:140], axis=1) ## 450-100bp
            cc_data[:,3] = np.sum(tmp_cc_data[:,140:], axis=1) ## 0-100bp
        ## only use data with non-zero signals
        mask = [np.any(cc_data[k,] != 0) for k in range(cc_data.shape[0])]
        ## TODO: unmask -- use all samples
        #mask = np.arange(cc_data.shape[0])

        ## add histone marks
        if parsed.file_ca is not None:
            ca_feat_indx = [k for k in range(len(ca_features)) if ca_features[k] in ca_feature_filtering_prefix]
            cc_data = np.concatenate((cc_data, ca_data[:,ca_feat_indx]), axis=1)

        combined_data = np.vstack((combined_data, cc_data[mask,]))
        combined_labels = np.append(combined_labels, labels[mask])
    print(combined_data.shape, "+1:", sum(combined_labels == 1), "-1:", sum(combined_labels == -1))
    combined_labels = np.array(combined_labels, dtype=int)
    ## convert to 0/1 labeling 
    combined_labels[combined_labels == -1] = 0

    if shuffle_sample:
        indx_rand = np.random.permutation(len(combined_labels))
        combined_data, combined_labels = combined_data[indx_rand,], combined_labels[indx_rand]

    return (combined_data, combined_labels)


def cross_validate_coarse_model(X, y, algorithm, split_te_set=True, num_fold=10, opt_param=False, json_filename=None):

    ## TODO: dict to store simulated results
    sim_dict = {"cc_high":{}, "cc_low":{}, "hm_high":{}, "hm_low":{}}


    results = []
    num_rnd_permu = 80
    ## define training, testing split
    if split_te_set:
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=1./num_fold, random_state=1)
        print(len(y_tr[y_tr == 1]) / float(len(y_tr)), len(y_te[y_te == 1]) / float(len(y_te)))
    else:
        X_tr, y_tr = X, y
    ## perform CV
    y_all_tr = np.empty(0)
    y_pred_prob = np.empty(0)
    combined_auprcs = np.empty(0)
    y_rnd_prob = {}
    for i in range(num_rnd_permu):
        y_rnd_prob[i] = np.empty(0)

    k_fold = StratifiedKFold(num_fold, shuffle=True, random_state=1)
    sys.stderr.write("... cv: ") 
    for k, (cv_tr, cv_te) in enumerate(k_fold.split(X_tr, y_tr)):
        X_cv_tr = X_tr[cv_tr]
        y_cv_tr = y_tr[cv_tr]
        
        ## preprocessing data
        cv_scaler = StandardScaler().fit(X_cv_tr)
        X_cv_tr = cv_scaler.transform(X_cv_tr)
        X_cv_te = cv_scaler.transform(X_tr[cv_te])
        ## construct model
        sys.stderr.write("%d " % k)
        model = construct_classification_model(X_cv_tr, y_cv_tr, algorithm, opt_param)
        model.fit(X_cv_tr, y_cv_tr) 
        ## TODO: to be removed
        de_class_indx = np.where(model.classes_ == 1)[0][0]
        tr_y_pred_prob = model.predict_proba(X_cv_tr)[:,de_class_indx]
        ## internal validation 
        de_class_indx = np.where(model.classes_ == 1)[0][0]
        y_all_tr = np.append(y_all_tr, y_tr[cv_te])
        cv_pred_prob = model.predict_proba(X_cv_te)[:,de_class_indx]
        y_pred_prob = np.append(y_pred_prob, cv_pred_prob)

        auprc_te = 100*average_precision_score(y_tr[cv_te], cv_pred_prob)
        combined_auprcs = np.append(combined_auprcs, auprc_te)
        print('validation AuPRC: %.2f%%' % auprc_te)
        ## TODO: changed to rankings of predicted probs
        # y_pred_prob = np.append(y_pred_prob, rankdata(cv_pred_prob))

        for i in range(num_rnd_permu):
            ## randomly permute each column
            rnd_X_tr_cv_te = copy.deepcopy(X_cv_te) 
            for j in range(rnd_X_tr_cv_te.shape[1]):
                rnd_X_tr_cv_te[:,j] = np.random.permutation(rnd_X_tr_cv_te[:,j])
            y_rnd_prob[i] = np.append(y_rnd_prob[i], model.predict_proba(rnd_X_tr_cv_te)[:,de_class_indx]) 



        ## TODO: test on perturbed feature values
        auprc_loss_pert = []
        conf_change_pert = []
        for findx in range(0, 14):
            X_pert = np.copy(X_cv_te)
            X_pert[:,findx] = np.max(X_cv_te[:,findx])
            pert_prob = model.predict_proba(X_pert)[:,de_class_indx]
            auprc_pert = 100*average_precision_score(y_tr[cv_te], pert_prob)
            auprc_loss_pert.append(auprc_pert - auprc_te)
            conf_change_pert.append(np.mean(pert_prob - cv_pred_prob))
            # print('%d\thigh\t%.2f%%' % (findx, auprc_pert))
        print(len(conf_change_pert))
        sim_dict["cc_high"][k] = {
                    "indx": np.argsort(auprc_loss_pert[:4]).tolist(),
                    "loss": np.sort(auprc_loss_pert[:4]).tolist(),
                    "indx_conf": np.argsort(conf_change_pert[:4])[::-1].tolist(),
                    "conf_change": np.sort(conf_change_pert[:4])[::-1].tolist()
                    }
        sim_dict["hm_high"][k] = {
                    "indx": np.argsort(auprc_loss_pert[4:]).tolist(),
                    "loss": np.sort(auprc_loss_pert[4:]).tolist(),
                    "indx_conf": np.argsort(conf_change_pert[4:])[::-1].tolist(),
                    "conf_change": np.sort(conf_change_pert[4:])[::-1].tolist()
                    
                    }

        auprc_loss_pert = []
        conf_change_pert = []
        for findx in range(0, 14): 
            X_pert = np.copy(X_cv_te)
            X_pert[:,findx] = np.min(X_cv_te[:,findx])
            pert_prob = model.predict_proba(X_pert)[:,de_class_indx]
            auprc_pert = 100*average_precision_score(y_tr[cv_te], pert_prob)
            auprc_loss_pert.append(auprc_pert - auprc_te)
            conf_change_pert.append(np.mean(pert_prob - cv_pred_prob))
            # print('%d\tlow\t%.2f%%' % (findx, auprc_pert))
        sim_dict["cc_low"][k] = {
                    "indx": np.argsort(auprc_loss_pert[:4]).tolist(),
                    "loss": np.sort(auprc_loss_pert[:4]).tolist(),
                    "indx_conf": np.argsort(conf_change_pert[:4]).tolist(),
                    "conf_change": np.sort(conf_change_pert[:4]).tolist()
                    }
        sim_dict["hm_low"][k] = {
                    "indx": np.argsort(auprc_loss_pert[4:]).tolist(),
                    "loss": np.sort(auprc_loss_pert[4:]).tolist(),
                    "indx_conf": np.argsort(conf_change_pert[4:])[::-1].tolist(),
                    "conf_change": np.sort(conf_change_pert[4:])[::-1].tolist()
                    }


        
    ## TODO: save dict
    import json
    with open(json_filename, "w") as fp:
        json.dump(sim_dict, fp, sort_keys=True, indent=4)



    ## calculate AUPRs
    results = np.hstack((y_all_tr.reshape(-1,1), y_pred_prob.reshape(-1,1)))
    print('$$ Average AuPRCs: %.2f%%, Overall AuPRC: %.2f%%' % (np.mean(combined_auprcs), 100* average_precision_score(results[:,0], results[:,1])))

    if split_te_set:
        ## preprocessing data
        scaler = StandardScaler().fit(X_tr)
        X_tr = cv_scaler.transform(X_tr)
        X_te = cv_scaler.transform(X_te)
        ## model trained with full training set
        model = construct_classification_model(X_tr, y_tr, algorithm, opt_param)
        model.fit(X_tr, y_tr) 
        de_class_indx = np.where(model.classes_ == 1)[0][0]
        ## predict the leftout testing set
        y_pred_prob = model.predict_proba(X_te)[:,de_class_indx] ## overwriting
        for i in range(num_rnd_permu):
            y_rnd_prob[i] = model.predict_proba(np.random.permutation(X_te))[:,de_class_indx] ## overwriting
        ## calculate AUPRs
        aupr_pred = average_precision_score(y_te, y_pred_prob)
        aupr_rnd = sorted([average_precision_score(y_te, y_rnd_prob[i]) for i in y_rnd_prob.keys()])
        print("\nTesting AUPR = %.3f" % aupr_pred)
        print("Testing Randomized, median AUPR = %.3f, 95%% CI = [%.3f, %.3f]" % (np.median(aupr_rnd), aupr_rnd[2], aupr_rnd[-3]))
        results += [np.median(aupr_rnd), aupr_rnd[2], aupr_rnd[-3], aupr_rnd[-3]-np.median(aupr_rnd), aupr_pred]    

    return results


def main(argv):
    parsed = parse_args(argv)
    np.random.seed(1)
    if parsed.file_ca is None:
        combined_data, combined_labels = prepare_data(parsed)
        ## validate RF
        print("-----\nRF\n-----")
        _ = cross_validate_model(combined_data, combined_labels, "RandomForestClassifier", False, 10, False)
        # print("-----\nRF sum logRPH\n-----")
        # combined_data, combined_labels = prepare_data(parsed, "logrph_total")
        # _ = model_interactive_feature(combined_data, combined_labels, "RandomForestClassifier", False)

    else:
        # combined_data, combined_labels = prepare_data(parsed)
        # ## validate RF
        # print("-----\nRF\n-----")
        # if parsed.valid_sample_name is None:
        #     json_filename = "output4/multiTF_feature_ranking.rf_coarse.json"
        # else:
        #     json_filename = "output4/"+ parsed.valid_sample_name +"_feature_ranking.rf_corase.json"
        # _ = cross_validate_model(combined_data, combined_labels, "RandomForestClassifier", False, 10, False, json_filename)
        # # print("-----\nRF sum logRPH\n-----")
        # # combined_data, combined_labels = prepare_data(parsed, "logrph_total")
        # # _ = model_interactive_feature(combined_data, combined_labels, "RandomForestClassifier", False)


        combined_data, combined_labels = prepare_coarse_data(parsed)
        ## validate RF
        print("-----\nRF\n-----")
        if parsed.valid_sample_name is None:
            json_filename = "output4/multiTF_feature_ranking.rf_coarse.json"
        else:
            json_filename = "output4/"+ parsed.valid_sample_name +"_feature_ranking.rf_corase.json"
        _ = cross_validate_coarse_model(combined_data, combined_labels, "RandomForestClassifier", False, 10, False, json_filename)
        # print("-----\nRF sum logRPH\n-----")
        # combined_data, combined_labels = prepare_data(parsed, "logrph_total")
        # _ = model_interactive_feature(combined_data, combined_labels, "RandomForestClassifier", False)


if __name__ == "__main__":
    main(sys.argv)
