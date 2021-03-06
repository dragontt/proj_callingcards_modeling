#!/usr/bin/python
import os
import sys
import numpy as np
import argparse
import json
from scipy.stats import rankdata
from model_fitting_util import *
from sklearn.model_selection import StratifiedKFold
# from sklearn.metrics import average_precision_score

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


global color_theme
color_theme = {"black":(0,0,0), 
                "blue":(31, 119, 180), "blue_L":(174, 199, 232), 
                "orange":(255, 127, 14), "orange_L":(255, 187, 120),
                "green":(44, 160, 44), "green_L":(152, 223, 138), 
                "red":(214, 39, 40), "red_L":(255, 152, 150),
                "magenta":(148, 103, 189), "magenta_L":(197, 176, 213),
                "brown":(140, 86, 75), "brown_L":(196, 156, 148),
                "pink":(227, 119, 194), "pink_L":(247, 182, 210), 
                "grey":(127, 127, 127), "grey_L":(199, 199, 199),
                "yellow":(255, 215, 0), "yellow_L":(219, 219, 141), 
                "cyan":(23, 190, 207), "cyan_L":(158, 218, 229)}
for c in color_theme.keys():
    (r, g, b) = color_theme[c]
    color_theme[c] = (r/255., g/255., b/255.)

global ml_algo_dict
ml_algo_dict = {"rf": ("RandomForestClassifier", "rf"),
                "gb": ("GradientBoostingClassifier", "gb"),
                "lr": ("LogisticRegression", "lr")}


def parse_args(argv):
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-a","--ml_algo", help="Machine learning algorithm to use")
    parser.add_argument("-c","--cc_dir", help="Calling Cards feature directory")
    parser.add_argument("-d","--de_dir", help="Differential expression directory")
    parser.add_argument("-hm","--file_hm", help="Histone mark file")
    parser.add_argument("-wt","--wt_dir", help="WT expression directory")
    parser.add_argument("-t","--tfs", help="Json file of TF name")
    parser.add_argument("-o","--output_dir")
    parser.add_argument("--tf_names", help="<sys_nam>,<common_name>")
    parsed = parser.parse_args(argv[1:])
    return parsed


def combine_data(file_cc, file_de, file_hm=None, file_wt=None):
    cc = np.loadtxt(file_cc, dtype=str)
    de = np.loadtxt(file_de, dtype=str, usecols=[0,2])
    if file_hm is not None:
        hm = np.loadtxt(file_hm, dtype=str, skiprows=1)
    if file_wt is not None:
        wt = np.loadtxt(file_wt, dtype=str, skiprows=1, usecols=[0,2])

    de_abs = np.abs(np.array(de[:,1], dtype=float))
    five_percentile = np.sort(de_abs)[::-1][int(np.floor(len(de)*.05))]
    if five_percentile == 0:
        indx_pos = np.where(de_abs != 0)[0] 
    else:
        indx_pos = np.where(de_abs >= five_percentile)[0]
    de[:,1] = -1
    de[indx_pos,1] = 1

    orfs = sorted(list(set(cc[:,0]) & set(de[:,0])))
    if file_hm is not None:
        orfs = sorted(list(set(orfs) & set(hm[:,0])))
    if file_wt is not None:
        orfs = sorted(list(set(orfs) & set(wt[:,0])))    
        
    data = []
    for orf in orfs:
        indx_cc = np.where(cc[:,0] == orf)[0][0]
        indx_de = np.where(de[:,0] == orf)[0][0]
        row = [de[indx_de,1]] + list(cc[indx_cc,1:]) 
        if file_hm is not None:
            indx_hm = np.where(de[:,0] == orf)[0][0]
            row += list(hm[indx_hm,1:])
        if file_wt is not None:
            indx_wt = np.where(de[:,0] == orf)[0][0]
            row += list(hm[indx_wt,1:])
        data.append(row)
    return np.array(data, dtype=float)


def cv_model(X, y, algorithm, nfolds=10, opt_param=False):
    y_labels = np.empty(0)
    y_preds = np.empty(0)
    combined_auprcs = np.empty(0)

    ## define k-fold cv
    k_fold = StratifiedKFold(nfolds, shuffle=True, random_state=1)
    sys.stderr.write("cv: ") 
    for k, (cv_tr, cv_te) in enumerate(k_fold.split(X, y)):
        sys.stderr.write("%d " % k)
        X_cv_tr, X_cv_te = X[cv_tr], X[cv_te]
        y_cv_tr, y_cv_te = y[cv_tr], y[cv_te]
        
        ## preprocessing data
        # cv_scaler = StandardScaler().fit(X_cv_tr)
        # X_cv_tr = cv_scaler.transform(X_cv_tr)
        # X_cv_te = cv_scaler.transform(X[cv_te])
        ## construct and fit model
        model = construct_classification_model(X_cv_tr, y_cv_tr, 
                                            algorithm, opt_param)
        model.fit(X_cv_tr, y_cv_tr)
        ## internal validation 
        de_pos_indx = np.where(model.classes_ == 1)[0][0]
        y_labels = np.append(y_labels, y_cv_te)
        y_cv_pred = model.predict_proba(X_cv_te)[:,de_pos_indx]
        y_preds = np.append(y_preds, y_cv_pred)
    sys.stderr.write("\n")
    return np.hstack((y_labels.reshape(-1,1), y_preds.reshape(-1,1)))


def cal_support_rates(data, step, bin, set_tie_rank=False):
    if set_tie_rank:
        data = data[data[:,1] >= data[step*bin,1] ,]
        xpts, supp_rates = [], []
        rankings = rankdata(-data[:,1], method='max')
        for r in np.unique(rankings):
            i = np.where(rankings == r)[0][-1]
            x = data[:(i+1),0]
            xpts.append(i)
            supp_rates.append(float(len(x[x==1]))/len(x))
    else:
        xpts = np.arange(1,bin+1)*step
        supp_rates = []
        for i in range(1,bin+1):
            x = data[data[:,1] >= data[i*step,1] ,0]
            r = float(len(x[x==1])) / len(x)
            supp_rates.append(r)
    supp_rates = [r*100 for r in supp_rates]
    return (np.array(xpts), np.array(supp_rates))


def plot_support_rate(data, data_types, line_colors, fig_filename, min_rank=300, set_tie_rank=False, step=5):
    bin = min_rank/step
    fig = plt.figure(num=None, figsize=(4,4), dpi=300)
    for i in range(len(data_types)):
        data_sorted = np.array(sorted(data[data_types[i]], 
                                key=lambda x: (x[1],x[0])))[::-1]
        xpts, rates = cal_support_rates(data_sorted, step, bin, set_tie_rank)
        plt.plot(xpts, rates, color=line_colors[i], label=data_types[i])
    random = 2.7 if 'Leu3' in fig_filename else 5
    plt.plot(np.arange(min_rank), random*np.ones(min_rank), color="#777777", linestyle=':', label="Random")
    plt.xlabel("Ranking by binding signal", fontsize=14)
    plt.ylabel("% responsive", fontsize=14)
    space = 25
    plt.xticks(np.arange(space-1,min_rank,space), 
                np.arange(space,step*(bin+1),space), rotation=60)
    plt.xlim([0,min_rank])
    # plt.ylim([0,100])
    plt.legend(loc="best", frameon=True)
    plt.tight_layout()
    plt.savefig(fig_filename, fmt="pdf")
    plt.close()


def main(argv):
    parsed = parse_args(argv)
    data_header = ['label', 'tph_total', 'rph_total', 'logrph_total', 
                'tph_bs_total', 'rph_bs_total', 'logrph_bs_total', '-log_p']
    cc_feature = 'tph_bs_total'

    cc_filename_suffix = ".cc_single_feature_matrix.txt"
    de_filename_suffix = "-15min.DE.txt"
    wt_filename_suffix = ".WT.txt"

    sys_name, common_name = parsed.tf_names.split(",")
    sys.stderr.write("... working on %s\t" % common_name)
    
    ## cv learning model
    result_dict = {}
    file_cc = parsed.cc_dir + sys_name + cc_filename_suffix
    file_de = parsed.de_dir + sys_name + de_filename_suffix
    file_wt = parsed.wt_dir + sys_name + wt_filename_suffix if parsed.wt_dir is not None else None
    data = combine_data(file_cc, file_de, file_hm=parsed.file_hm, file_wt=file_wt)
    
    feature_indx = [0,data_header.index(cc_feature)]
    feature_indx += range(len(data_header), data.shape[1])
    data = data[:, feature_indx]
    result_dict['simple'] = data[:,[0,1]]
    ## set algorithm and hyperparameter optimization
    ml_algo = ml_algo_dict[parsed.ml_algo]
    opt_hyparam = False
    opt_suffix = '_bo' if opt_hyparam else '' 
    result_dict[ml_algo[1]+'_cv10'] = cv_model(data[:,1:], data[:,0], 
                            ml_algo[0], nfolds=10, opt_param=opt_hyparam)
    result_dict[ml_algo[1]+'_cv100'] = cv_model(data[:,1:], data[:,0], 
                            ml_algo[0], nfolds=100, opt_param=opt_hyparam)
    ## save cv predicted probs
    for k in result_dict.keys():
        if k == 'simple':
            cv_filename = parsed.output_dir + k +"."+ common_name +".txt"
        else:
            cv_filename = parsed.output_dir + k + opt_suffix +"."+ common_name +".txt"
        np.savetxt(cv_filename, result_dict[k], delimiter="\t")
    ## make comparison plot
    labels = [ml_algo[1] +'_cv10', ml_algo[1] +'_cv100', 'simple']
    line_colors = [color_theme['blue'], 
                    color_theme['orange'], 
                    color_theme['black']]
    fig_filename = parsed.output_dir + ml_algo[1] +"_ranking"+ opt_suffix +"."+ common_name +".pdf"
    plot_support_rate(result_dict, labels, line_colors, 
                        fig_filename, set_tie_rank=False)


if __name__ == "__main__":
    main(sys.argv)
