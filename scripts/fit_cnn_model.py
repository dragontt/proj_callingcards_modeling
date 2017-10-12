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

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def parse_args(argv):
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-t","--feature_type", 
                        help="choose from ['highest_peaks', 'binned_promoter']")
    parser.add_argument("-c","--cc_dir", help="Calling Cards feature directory")
    parser.add_argument("-d","--de_dir", help="Differential expression directory")
    parser.add_argument("-p","--bp_dir", help="Binding potential directory")
    parser.add_argument("-a","--file_ca", help="Chromatin accessibility file")
    parser.add_argument("-v","--valid_sample_name", help="TF sample name to focus")
    parser.add_argument("-l","--optimized_labels")
    parser.add_argument("-o","--output_filename")
    parsed = parser.parse_args(argv[1:])
    return parsed


def prepare_data(parsed, shuffle_sample=True):
    if parsed.optimized_labels: ## parse optimized set if available 
        optimized_labels = parse_optimized_labels(parsed.optimized_labels)
        valid_sample_names = optimized_labels.keys()
        label_type = "categorical"
    elif parsed.de_dir: ## parse valid DE samples if available
        files_de = glob.glob(parsed.de_dir +"/*15min.DE.txt")
        if parsed.valid_sample_name:
            valid_sample_names = [parsed.valid_sample_name]
        else:
            valid_sample_names = [os.path.basename(f).split('-')[0] for f in files_de]
        label_type = "continuous"
    else:
        sys.exit("Require the label directory: optimized subset file or DE folder!") 

    ## parse input
    label_type = "conti2top5pct"
    files_cc = glob.glob(parsed.cc_dir +"/*.cc_feature_matrix."+ parsed.feature_type +".txt")
    cc_data_collection, cc_features = process_data_collection(files_cc, files_de,
                                            valid_sample_names, label_type, False)
    
    ## query samples
    cc_feature_filtering_prefix = "logrph"
    bp_feature_filtering_prefix = ["sum_score", "count", "dist"]
    ca_feature_filtering_prefix = ['H3K27ac_prom_-1','H3K36me3_prom_-1','H3K4me3_prom_-1',
                                    'H3K79me_prom_-1','H4K16ac_prom_-1','H3K27ac_body',
                                    'H3K36me3_body','H3K4me3_body','H3K79me_body',
                                    'H4K16ac_body']
    print("... querying data")
    combined_data = np.empty((0,160))
    combined_labels = np.empty(0)
    for sample_name in sorted(cc_data_collection):
        labels, cc_data, _ = query_data_collection(cc_data_collection, sample_name,
                                        cc_features, cc_feature_filtering_prefix)
        mask = [np.any(cc_data[k,] != 0) for k in range(cc_data[:,:-1].shape[0])]
        combined_data = np.vstack((combined_data, cc_data[mask,:-1]))
        combined_labels = np.append(combined_labels, labels[mask])
    print(combined_data.shape, "+1:", len(combined_labels[combined_labels == 1]), "-1:", len(combined_labels[combined_labels == -1]))
    combined_labels = np.array(combined_labels, dtype=int)
    ## convert to 0/1 labeling 
    combined_labels[combined_labels == -1] = 0

    if shuffle_sample:
        indx_rand = np.random.permutation(len(combined_labels))
        combined_data, combined_labels = combined_data[indx_rand,], combined_labels[indx_rand]

    return (combined_data, combined_labels)


def train_ConvNet(X, y):
    ## instantiate conv net
    conv_net = ConvNet()

    ## define loss function and optimizer
    # criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([.05,.95]))
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(conv_net.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(conv_net.parameters(), lr=0.001)

    ## subsample negative class, and shuffle combined data
    # indx_pos = np.where(y == 1)[0]
    # indx_neg = np.random.choice(np.where(y == 0)[0], size=sum(y == 1))
    # X, y = X[np.append(indx_pos, indx_neg),], y[np.append(indx_pos, indx_neg)]
    indx_rand = np.random.permutation(len(y))
    X, y = X[indx_rand,], y[indx_rand]

    ## run multiple epochs 
    for epoch in range(2):
        running_loss = 0.0
        mini_batch_size = 32
        for i in range(int(np.ceil(len(y)/float(mini_batch_size)))):
            ## prepare mini batch
            indx = range(i*mini_batch_size, min((i+1)*mini_batch_size,len(y)))
            Xi = np.expand_dims(X[indx,], axis=1)
            yi = y[indx]
            Xi, yi = Variable(torch.FloatTensor(Xi)), Variable(torch.LongTensor(yi))
            ## run forward/backward prop, and optimize
            optimizer.zero_grad()
            Xi_out = conv_net(Xi)
            loss = criterion(Xi_out, yi)
            loss.backward()
            optimizer.step()
            
            ## print status
            running_loss += loss.data[0]
            if i % 100 == 99:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss))
                running_loss = 0.0

    print('Finished Training')
    return conv_net


def test_ConvNet(conv_net, X, y):
    ## prepare data
    X = np.expand_dims(X, axis=1)
    X, y = Variable(torch.FloatTensor(X)), torch.LongTensor(y)
    ## make prediction
    y_out = conv_net(X)
    _, y_pred = torch.max(y_out.data, 1)
    
    ## calculate accuracy
    accu = 100* (y_pred == y).sum() / y.size(0)
    # print('Accuracy of the network on CV test: %.2f %%' % accu)
    ## calculate AuPRC
    auprc = average_precision_score(y.numpy(), y_out.data[:,1].numpy())
    return accu, auprc


def cross_validate_model(X, y, num_fold=10):
    ## perform CV
    k_fold = StratifiedKFold(num_fold, shuffle=True, random_state=1)
    for k, (train, test) in enumerate(k_fold.split(X, y)):
        print('... working on cv fold %d' % k)
        ## train and test
        conv_net = train_ConvNet(X[train], y[train])
        accu_test, auprc_test = test_ConvNet(conv_net, X[test], y[test])
        print('Test set\tAccu: %.2f%%\tAuPRC: %.2f%%' % (accu_test, auprc_test))
        ## test randomized data
        accu_rand, auprc_rand = np.zeros(20), np.zeros(20)
        for i in range(20):
            X_rand = copy.copy(X[test])
            for j in range(X_rand.shape[1]):
                X_rand[:,j] = np.random.permutation(X_rand[:,j])
            accu_rand[i], auprc_rand[i] = test_ConvNet(conv_net, X_rand, y[test])
        print('Randomized\tAccu: %.2f%%\tAuPRC: %.2f%%' % (np.median(accu_rand), np.median(auprc_rand)))
        # sys.exit()


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv1d(1, 10, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(10, 20, kernel_size=5, padding=2)
        self.pool = nn.MaxPool1d(4)
        # self.dropout = nn.Dropout(p=.1)
        self.fc1 = nn.Linear(200, 64)
        self.fc2 = nn.Linear(64, 8)
        self.fc3 = nn.Linear(8, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 200)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def main(argv):
    parsed = parse_args(argv)
    combined_data, combined_labels = prepare_data(parsed)
    cross_validate_model(combined_data, combined_labels)


if __name__ == "__main__":
    main(sys.argv)
