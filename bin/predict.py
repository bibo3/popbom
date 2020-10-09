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
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
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
    parser.add_argument('--loops', '-l', default=10, help='how many splits and loops for validating')
    parser.add_argument('--output', '-o', help='name of output file')
    parser.add_argument('--classifier', '-c', choices=['RF', 'SVM'], help='which classifier to use')

    return parser.parse_args()


# handling metadata
def filter_metadata(md_total, df_summary):
    metadata = pd.read_csv(md_total, index_col=0)
    return metadata[metadata.index.isin(list(df_summary.index))]


# spliting the datasets into test/train sets
# returns arrays X_train, X_test, y_train, y_test of length #splits  
def split_data(df_data, splits, seed_value):
    sss = StratifiedShuffleSplit(n_splits=splits, test_size=0.2, random_state=seed_value)
    X_train, X_test, y_train, y_test = [], [], [], []
    labels = df_data.index.get_level_values('disease').values
    X = np.zeros(len(labels))
    for train_index, test_index in sss.split(X, labels):
        y_train.append(df_data.iloc[train_index].index.get_level_values('disease').values)
        y_test.append(df_data.iloc[test_index].index.get_level_values('disease').values)
        X_train.append(df_data.iloc[train_index])
        X_test.append(df_data.iloc[test_index])
    return X_train, X_test, y_train, y_test


# Evaluation function: auc score, precision, accuracy, recall and f1
def evaluate_performance(y_true, y_pred, y_pred_proba):
    auc = roc_auc_score(y_true, y_pred_proba)
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
 #       print(type(y_train[i]))
  #      print(X_train[i])
        X_train_s, X_test_s, y_train_s, y_test_s = split_data(X_train[i], loops, seed)
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
        df=pd.read_csv(args.input, index_col=[0,1], header=0)
    if args.taxo == 'kraken2':
        df=pd.read_csv(args.input, index_col=[0,1], header=[0,1])
    
    X_train, X_test, y_train, y_test = split_data(df, loops, seed)
    best_prediction = []
    # Random Forrest Classifier 
    if args.classifier == 'RF':
        param_grid_rf = {
            'n_estimators': [100, 500, 700, 1000],
            'max_depth': [2, 6, 10, None],
            'max_features': [0.33, 0.5, 1, 100, 'auto'],
            'min_samples_split': [2, 3, None],
            'criterion': ['gini', 'entropy']
        }
    
        rf = RandomForestClassifier(random_state=seed)
        best_params = param_fitting(X_train, X_test, y_train, y_test, loops, seed, rf, param_grid_rf)
    
        proba, pred = [], []
        best_rf = RandomForestClassifier(**best_params, random_state=seed)
        #best_rf = RandomForestClassifier(n_estimators=500, max_depth=None, min_samples_split=2, n_jobs=-1, random_state=seed)
        for i in range(loops):
            best_rf.fit(X_train[i], y_train[i])
            proba = best_rf.predict_proba(X_test[i])[:,1]
            pred = best_rf.predict(X_test[i])
            best_prediction.append(evaluate_performance(y_test[i], pred, proba))

    # SVM
    if args.classifier == 'SVM':
        param_grid_svm = {
            'kernel': ['linear', 'poly'],
            'C': [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
            }
        svm = SVC(random_state=seed, probability=True)
        best_params = param_fitting(X_train, X_test, y_train, y_test, loops, seed, svm, param_grid_svm)
        best_svm = SVC(**best_params, random_state=seed, probability=True)
        for i in range(loops):
            best_svm.fit(X_train[i], y_train[i])
            best_prediction.append(evaluate_performance(y_test[i], best_svm.predict(X_test[i])))

    # Round results to r digits
    r = 5
    bp = np.array(best_prediction)
    print(f'Mean of ROC: {round(np.mean(bp[:,0]), r)}')
    print(f'--- {round((time.time()-start_time), 3)} seconds ---')
    
    # Writing output to file
    with open(args.output, 'w') as fh:
        fh.write('Script parameters:\n')
        fh.write(f'Input file: {args.input} \nTaxonomic profiler used: {args.taxo} \nValidation loops: {args.loops}\nClassifier: {args.classifier}\n')
        fh.write('\nBest hyperparameters:\n')
        fh.write(f'{best_params}\n')       
        fh.write('\nMean values of evaluation runs with standard deviation:\n')     
        fh.write(f'ROC: {round(np.mean(bp[:,0]), r)}\t{round(np.std(bp[:,0]), r)} \n')
        fh.write(f'Accuracy: {round(np.mean(bp[:,1]), r)}\t{round(np.std(bp[:,1]), r)} \n')
        fh.write(f'Precision: {round(np.mean(bp[:,2]), r)}\t{round(np.std(bp[:,2]), r)} \n')
        fh.write(f'Recall: {round(np.mean(bp[:,3]), r)}\t{round(np.std(bp[:,3]), r)} \n')
        fh.write(f'F1-Score: {round(np.mean(bp[:,4]), r)}\t{round(np.std(bp[:,4]), r)} \n')

        fh.write('\nAll evaluation scores:\nROC\tAccuracy\tPrecision\tRecall\tF1-Score\n')
        [fh.write(f'{round(i[0], r)}\t{round(i[1], r)}\t{round(i[2], r)}\t{round(i[3], r)}\t{round(i[4], r)} \n') for i in best_prediction]

        
if __name__ == '__main__':
    main()
