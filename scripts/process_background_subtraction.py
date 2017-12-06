#/usr/bin/python
from __future__ import print_function
import os
import sys
import numpy as np
import argparse
import glob
from model_fitting_util import *


def parse_args(argv):
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-t","--feature_type", default="binned_promoter")
    parser.add_argument("-c","--cc_dir", help="Calling Cards feature directory")
    parser.add_argument("-d","--de_dir", help="Differential expression directory")
    parser.add_argument("-a","--file_ca", help="Chromatin accessibility file")
    parser.add_argument("-v","--valid_sample_name", help="TF sample name to focus")
    parser.add_argument("-o","--output_directory")
    parsed = parser.parse_args(argv[1:])
    return parsed


def prepare_data(parsed, cc_feature_filtering_prefix="logrph", shuffle_sample=True):
    if parsed.de_dir: ## parse valid DE samples if available
        files_de = glob.glob(parsed.de_dir +"/*15min.DE.txt")
        if parsed.valid_sample_name is not None:
            valid_sample_names = [parsed.valid_sample_name]
        else:
            #valid_sample_names = [os.path.basename(f).split('-')[0] for f in files_de]
            valid_sample_names = ['YLR451W', 'YKL038W', 'YDR034C']
        background_name = "NOTF_Minus_Adh1_2015_17_combined"
        label_type = "continuous"
    else:
        sys.exit("Require the label directory: optimized subset file or DE folder!") 

    ## parse input
    label_type = "conti2top5pct"
    files_cc = glob.glob(parsed.cc_dir +"/*.cc_feature_matrix."+ parsed.feature_type +".txt")
    cc_data_collection, cc_features = process_data_collection(files_cc, files_de, valid_sample_names, label_type, False)
    cc_background_collection, _ = process_data_collection(files_cc, [parsed.de_dir +'NOTF_Minus_Adh1_2015_17_combined-15min.DE.txt'], [background_name], label_type, False)

    if parsed.file_ca is not None:
        ca_data, _, ca_features, _ = prepare_datasets_w_de_labels(parsed.file_ca, files_de[0], "pval", 0.1)
    
    ## query samples
    # cc_feature_filtering_prefix = "logrph"
    # cc_feature_filtering_prefix = "logrph_total"
    ca_feature_filtering_prefix = ['H3K27ac_prom_-1','H3K36me3_prom_-1',
                                    'H3K4me3_prom_-1','H3K79me_prom_-1',
                                    'H4K16ac_prom_-1','H3K27ac_body',
                                    'H3K36me3_body','H3K4me3_body',
                                    'H3K79me_body','H4K16ac_body']
    print("... querying data")
    ## query background 
    _, cc_background, _ = query_data_collection(cc_background_collection, background_name, cc_features, cc_feature_filtering_prefix)

    if parsed.file_ca is None:
        if cc_feature_filtering_prefix.endswith("total"):
            combined_data = np.empty((0,1))
            combined_data_bg = np.empty((0,1))
        else:
            combined_data = np.empty((0,160))
            combined_data_bg = np.empty((0,160))
    else:
        if cc_feature_filtering_prefix.endswith("total"):
            combined_data = np.empty((0,11))
            combined_data_bg = np.empty((0,1))
        else:
            combined_data = np.empty((0,170))
            combined_data_bg = np.empty((0,160))
    combined_labels = np.empty(0)

    ## query foreground CC data
    for sample_name in sorted(cc_data_collection):
        labels, cc_data, _ = query_data_collection(cc_data_collection, sample_name,
                                        cc_features, cc_feature_filtering_prefix)

        if cc_feature_filtering_prefix.endswith("total"):
            cc_data = cc_data[:,-1].reshape(-1,1)
            cc_background_filtered = cc_background[:,-1].reshape(-1,1)
        else:
            cc_data = cc_data[:,:-1]
            cc_background_filtered = cc_background[:,:-1]
        ## only use data with non-zero signals
        mask = [np.any(cc_data[k,] != 0) for k in range(cc_data.shape[0])]
        ## TODO: unmask -- use all samples
        #mask = np.arange(cc_data.shape[0])

        ## add histone marks
        if parsed.file_ca is not None:
            ca_feat_indx = [k for k in range(len(ca_features)) if ca_features[k] in ca_feature_filtering_prefix]
            cc_data = np.concatenate((cc_data, ca_data[:,ca_feat_indx]), axis=1)

        combined_data = np.vstack((combined_data, cc_data[mask,]))
        combined_data_bg = np.vstack((combined_data_bg, cc_background_filtered[mask,]))
        combined_labels = np.append(combined_labels, labels[mask])


    print(combined_data.shape, "+1:", sum(combined_labels == 1), "-1:", sum(combined_labels == -1))


    combined_labels = np.array(combined_labels, dtype=int)
    ## convert to 0/1 labeling 
    combined_labels[combined_labels == -1] = 0

    if shuffle_sample:
        indx_rand = np.random.permutation(len(combined_labels))
        combined_data, combined_labels = combined_data[indx_rand,], combined_labels[indx_rand]

    return (combined_data, combined_data_bg, combined_labels)


def main(argv):
    parsed = parse_args(argv)
    combined_data, combined_data_bg, combined_labels = prepare_data(parsed)
    np.savetxt(parsed.output_directory+"/multiTF_feature_matrix.txt", combined_data, fmt='%.10f', delimiter='\t')
    np.savetxt(parsed.output_directory+"/multiTF_bg_feature_matrix.txt", combined_data_bg, fmt='%.10f', delimiter='\t')
    np.savetxt(parsed.output_directory+"/multiTF_label_matrix.txt", combined_labels, fmt='%d', delimiter='\n')
    

if __name__ == "__main__":
    main(sys.argv)