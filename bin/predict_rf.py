#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 08:41:50 2020

@author: matt
"""


import os
import random
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import time
 
###### For reproducable ML
def set_seed(seed_value):
    os.environ['PYTHONHASHSEED']=str(seed_value) # 1. Set `PYTHONHASHSEED` environment variable at a fixed value
    random.seed(seed_value) # 2. Set `python` built-in pseudo-random generator at a fixed value
    np.random.seed(seed_value) # 3. Set `numpy` pseudo-random generator at a fixed value

def parseargs():
    parser = argparse.ArgumentParser(description='Run RF with kraken2 or metaphlan')
    parser.add_argument('--input', '-i', help='file containing report summary table')
    parser.add_argument('--taxo', '-t', choices=['metaphlan', 'kraken2'],
                        help='which taxonomic profiler has been used?')
    parser.add_argument('--seed', '-s', default=42, help='the seed for random values')
    parser.add_argument('--threads', default=-1, help='number of threads to be used for multithreading')
    parser.add_argument('--metadata', '-m', help='metadata file')
    parser.add_argument('--loops', '-l', default=10, help='how many splits and loops for validating')

    return parser.parse_args()

# handling metadata
def filter_metadata(md_total, df_summary):
    metadata = pd.read_csv(md_total, index_col=0)
    return metadata[metadata.index.isin(list(df_summary.index))]

# spliting the datasets into test/train sets
# returns arrays X_train, X_test, y_train, y_test of length #splits  
def split_data(label_df, df_data, splits, seed_value):
    sss = StratifiedShuffleSplit(n_splits=splits, test_size=0.2, random_state=seed_value)
    X_train, X_test, y_train, y_test = [], [], [], []
    X = np.zeros(label_df.shape[0])
    for train_index, test_index in sss.split(X, label_df.iloc[:,0]):
        y_train.append(label_df.iloc[train_index])
        y_test.append(label_df.iloc[test_index])
        X_train.append(df_data.iloc[train_index])
        X_test.append(df_data.iloc[test_index])
    return X_train, X_test, y_train, y_test

# Evaluation function: auc score, precision, accuracy, recall and f1
def evaluate_performance(y_true, y_pred):
    auc = roc_auc_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    accuracy = (tp + tn) / (tp + fp + tn + fn)
    if tp > 0.0:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = (2 * precision * recall) / (precision + recall)
    else:
        precision, recall, f1 = 0.0, 0.0, 0.0
    return [auc, accuracy, precision, recall, f1]

# perform grid search of model with given param grid (dict of params and values to consider)
def grid_search(X_train_data, X_test_data, y_train_data, model, param_grid, threads=-1, cv=5):
    gs = GridSearchCV(
        estimator=model,
        param_grid=param_grid, 
        cv=cv, 
        n_jobs=threads, 
        verbose=2)
    fitted_model = gs.fit(X_train_data, y_train_data)
    return fitted_model

# runs #loops^2, splits given set #loops times, then performs grid_search #loop times
# returns the best hyperparameters (set of parameters with highest sum of mean_test_score)
def param_fitting(X_train, X_test, y_train, y_test, loops, seed, model, param_grid):
    cv_params = []
    for i in range(loops):
        X_train_s, X_test_s, y_train_s, y_test_s = split_data(y_train[i], X_train[i], loops, seed)
        for j in range(loops):
            clf = grid_search(X_train_s[j], X_test_s[j], y_train_s[j], model, param_grid)
            cv_params.append(clf.cv_results_)
            
    df_total=pd.concat(map(lambda x: pd.DataFrame(x).set_index('params'), cv_params), axis=1)
    best_params = df_total['mean_test_score'].sum(axis=1).idxmax()
    return best_params
    

def main():
    start_time = time.time()
    args=parseargs()
    seed = int(args.seed)
    loops = int(args.loops)
    set_seed(seed)

    if args.taxo == 'metaphlan':
        df=pd.read_csv(args.input, index_col=0, header=0)
    if args.taxo == 'kraken2':
        df=pd.read_csv(args.input, index_col=0, header=[0,1])

    md_filtered=filter_metadata(args.metadata, df)
    
    X_train, X_test, y_train, y_test = split_data(md_filtered, df, loops, seed)
    
    param_grid = {
        'n_estimators': [10, 100],
        'max_depth': [6, 10, None],
        'max_features': [50, 100, 'auto'],
        'criterion': ['gini', 'entropy']
    }

    rf = RandomForestClassifier(random_state=seed)
    best_params = param_fitting(X_train, X_test, y_train, y_test, loops, seed, rf, param_grid)

    best_prediction = []
    best_rf = RandomForestClassifier(**best_params, random_state=seed)
    for i in range(loops):
        best_rf.fit(X_train[i], y_train[i])
        best_prediction.append(evaluate_performance(y_test[i], best_rf.predict(X_test[i])))
    #print(best_prediction)
    sum_roc = 0
    for j in best_prediction:
        print(j[0])
        sum_roc += j[0]
    print("Mean of ROC: %.3f" % sum_roc/len(best_prediction))
    print("--- %.2f seconds ---" % (time.time()-start_time))
 
if __name__ == '__main__':
    main()
