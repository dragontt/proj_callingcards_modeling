#!/usr/bin/python
import os
import sys
import numpy as np
import argparse
import glob
from model_fitting_util import *
import json, yaml


def parse_args(argv):
    parser = argparse.ArgumentParser(description="")
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


    ## parse input
    label_type = "conti2categ"
    ## label_type = "conti2top500DE"c
    if parsed.cc_dir:
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
    compiled_dict = {}

    for sample_name in sorted(wt_data_collection):
        compiled_results_col = []
        # for i in range(len(bp_feature_filtering_prefix)):
        for i in [len(bp_feature_filtering_prefix)-1]:
            if parsed.cc_dir:
                labels, cc_data, cc_f = query_data_collection(cc_data_collection, sample_name, cc_features, cc_feature_filtering_prefix)
                combined_data = cc_data
                features = cc_f

            if parsed.bp_dir:
                tmp_labels, bp_data, bp_f = query_data_collection(bp_data_collection, sample_name, bp_features, bp_feature_filtering_prefix[:(i+1)])
                if not 'combined_data' in vars():
                    combined_data = np.empty((bp_data.shape[0], 0))
                combined_data = np.concatenate((combined_data, bp_data),axis=1)
                if not 'labels' in vars():
                    labels = tmp_labels
                if not 'features' in vars():
                    features = np.empty((0))
                features = np.append(features, bp_f)

            if parsed.file_ca:
                ca_feat_indx = [k for k in range(len(ca_features)) if ca_features[k] in ca_feature_filtering_prefix]
                ca_f = ca_features[ca_feat_indx]
                if not 'combined_data' in vars():
                    combined_data = np.empty((ca_data[:,ca_feat_indx].shape[0], 0))
                combined_data = np.concatenate((combined_data, ca_data[:,ca_feat_indx]),axis=1)
                if not 'features' in vars():
                    features = np.empty((0))
                features = np.append(features, ca_f)

            if parsed.wt_dir:
                tmp_labels, wt_data, wt_f = query_data_collection(wt_data_collection, 
                                            sample_name, wt_features)
                if not 'combined_data' in vars():
                    combined_data = np.empty((wt_data.shape[0], 0))
                    labels = tmp_labels
                combined_data = np.concatenate((combined_data, wt_data), axis=1)
                if not 'labels' in vars():
                    labels = tmp_labels
                if not 'features' in vars():
                    features = np.empty((0))
                features = np.append(features, wt_f)

            print combined_data.shape, "+1:", len(labels[labels ==1]), "-1:", len(labels[labels ==-1])

            compiled_dict[sample_name] = {'data': combined_data.tolist(),
                                            'labels': labels.tolist(),
                                            'features': features.tolist()}

            ## use binding potential feature to train and predict
            # results = model_interactive_feature(combined_data, labels, classifier)

            # results = model_interactive_feature(combined_data, labels, classifier, 10, True)
            # compiled_results_col += results
            del combined_data
            del labels
            del features
        # compiled_results = np.hstack((compiled_results, 
        #                                 np.array(compiled_results_col).reshape(-1,1)))
    # np.savetxt(parsed.output_filename, compiled_results, fmt="%s", delimiter='\t')

    with open('../output/compiled_data.json', 'w') as fp:
                json.dump(compiled_dict, fp)


    print "\n\n"

    # with open('../output/compiled_data.json') as fp:
    #     compiled_dict = yaml.load(fp)
    #     # compiled_dict = json.load(fp)
    if 'YDR034C-plusLys' in compiled_dict.keys():
        compiled_dict.pop('YDR034C-plusLys', None)

    features = np.array(compiled_dict[compiled_dict.keys()[0]]['features'])
    findx_lvl1 = np.where(features !='logrph_total')[0]
    findx_lvl2 = np.where(features =='logrph_total')[0]

    for sample_te in sorted(compiled_dict.keys()):
        samples_tr = np.setdiff1d(compiled_dict.keys(), sample_te)
        ## combine training set
        data_tr = np.empty((0, np.array(compiled_dict[samples_tr[0]]['data']).shape[1]))
        label_tr = np.empty(0)
        for sample_tr in samples_tr:
            data_tr = np.concatenate((data_tr, np.array(compiled_dict[sample_tr]['data'])), axis=0)
            label_tr = np.append(label_tr, np.array(compiled_dict[sample_tr]['labels']))
        data_tr_lvl1 = data_tr[:, findx_lvl1]
        data_tr_lvl2 = data_tr[:, findx_lvl2]
        ## get test set
        data_te_lvl1 = np.array(compiled_dict[sample_te]['data'])[:, findx_lvl1]
        data_te_lvl2 = np.array(compiled_dict[sample_te]['data'])[:, findx_lvl2]
        label_te = np.array(compiled_dict[sample_te]['labels'])

        ## build classifier for level 1
        print "... training level 1 classifer"
        model_lvl1 = construct_classification_model(data_tr_lvl1, label_tr, "RandomForestClassifier")
        model_lvl1.fit(data_tr_lvl1, label_tr)
        de_class_indx = np.where(model_lvl1.classes_ == 1)[0][0]
        prob_lvl1 = model_lvl1.predict_proba(data_tr_lvl1)[:,de_class_indx]
        data_tr_lvl2 = np.hstack((data_tr_lvl2, prob_lvl1.reshape(-1,1)))
        ## build classifier for level 2
        print "... training level 2 classifer"
        model_lvl2 = construct_classification_model(data_tr_lvl2, label_tr, "RandomForestClassifier")
        model_lvl2.fit(data_tr_lvl2, label_tr)

        ## now make predicts
        print "... testing model"
        de_class_indx = np.where(model_lvl1.classes_ == 1)[0][0]
        prob_lvl1 = model_lvl1.predict_proba(data_te_lvl1)[:,de_class_indx]
        data_te_lvl2 = np.hstack((data_te_lvl2, prob_lvl1.reshape(-1,1)))
        de_class_indx = np.where(model_lvl2.classes_ == 1)[0][0]
        label_te_prob = model_lvl2.predict_proba(data_te_lvl2)[:,de_class_indx]
        aupr_pred = average_precision_score(label_te, label_te_prob)
        print sample_te, "AUPR:", aupr_pred
        print "lvl1:", len(prob_lvl1[prob_lvl1 >= .5])
        print "lvl2:", len(label_te_prob[label_te_prob >= .5])
        print ""

        ## predict randomized features
        rnd_auprs = []
        for i in range(200):
            rnd_data_te_lvl1 = np.empty(data_te_lvl1.shape)
            for j in range(data_te_lvl1.shape[1]):
                rnd_data_te_lvl1[j] = np.random.permutation(data_te_lvl1[j])
            de_class_indx = np.where(model_lvl1.classes_ == 1)[0][0]
            prob_lvl1 = model_lvl1.predict_proba(rnd_data_te_lvl1)[:,de_class_indx]
            # rnd_data_te_lvl2 = np.hstack((np.random.permutation(data_te_lvl2), prob_lvl1.reshape(-1,1))) ## no shuffle of predicted values from level 1
            rnd_data_te_lvl2 = np.hstack((np.random.permutation(data_te_lvl2), np.random.permutation(prob_lvl1.reshape(-1,1))))
            de_class_indx = np.where(model_lvl2.classes_ == 1)[0][0]
            label_te_prob = model_lvl2.predict_proba(data_te_lvl2)[:,de_class_indx]
            rnd_auprs.append(average_precision_score(label_te, label_te_prob))
            print "randomized", i
            print "lvl1:", len(prob_lvl1[prob_lvl1 >= .5])
            print "lvl2:", len(label_te_prob[label_te_prob >= .5])
            print ""
            if i > 4: 
                break
        print sample_te, "AUPR - randomized labels:", np.median(rnd_auprs), "+", np.percentile(rnd_auprs, 95) - np.median(rnd_auprs)
        sys.exit()



if __name__ == "__main__":
    main(sys.argv)