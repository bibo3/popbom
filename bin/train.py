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
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve, matthews_corrcoef, balanced_accuracy_score, make_scorer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.feature_selection import VarianceThreshold
from joblib import dump
import time
 
###### For reproducable ML
def set_seed(seed_value):
    os.environ['PYTHONHASHSEED']=str(seed_value) # 1. Set `PYTHONHASHSEED` environment variable at a fixed value
    random.seed(seed_value) # 2. Set `python` built-in pseudo-random generator at a fixed value
    np.random.seed(seed_value) # 3. Set `numpy` pseudo-random generator at a fixed value

def parseargs():
    parser = argparse.ArgumentParser(description='Predict phenotype with RF or SVM')
    parser.add_argument('--input', '-i', help='file containing report summary table')
    parser.add_argument('--seed', '-s', default=42, help='the seed for random values')
    parser.add_argument('--threads', default=-1, help='number of threads to be used for multithreading')
    parser.add_argument('--loops_validation', '-lv', default=10, help='how many splits and loops for validating')
    parser.add_argument('--loops_tuning', '-lt', default=10, help='how many splits and loops for hyperparameter tuning')
    parser.add_argument('--output', '-o', help='name of output file')
    parser.add_argument('--classifier', '-c', choices=['RF', 'SVM', 'L2linear', 'XGB'], help='which classifier to use')
    parser.add_argument('--scorer', '-st', default='balanced_accuracy', help='The scoring function to use for hyperparameter tuning')
    parser.add_argument('--varience_threshold', '-v', default=0, help='threshold for varience based feature selection')
    return parser.parse_args()


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


# Evaluation function: auc score, precision, accuracy, recall, f1, mcc, balanced accuracy, tpr, fpr
def evaluate_performance(y_true, y_pred, y_pred_proba):
    auc = roc_auc_score(y_true, y_pred_proba)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    confusion = [tp, fp, fn, tn]
    accuracy = (tp + tn) / (tp + fp + tn + fn)
    if tp > 0.0:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = (2 * precision * recall) / (precision + recall)
    else:
        precision, recall, f1 = 0.0, 0.0, 0.0
    mcc = matthews_corrcoef(y_true, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    return [auc, accuracy, precision, recall, f1, mcc, balanced_accuracy, confusion]


# perform grid search of model with given param grid (dict of params and values to consider)
def grid_search(X_train_data, X_test_data, y_train_data, model, param_grid, scorer, threads=-1, cv=5):
    gs = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=scorer,
        cv=cv, 
        n_jobs=threads, 
        verbose=2)
    fitted_model = gs.fit(X_train_data, y_train_data)
    return fitted_model


# runs #loops_validation * #loops_tuning, splits given set #loops_tuning times, then performs grid_search
# returns the best hyperparameters (set of parameters with highest sum of mean_test_score)
def param_fitting(X_train, X_test, y_train, y_test, loops_v, loops_t, seed, model, param_grid, scorer):
    cv_params = []
    
    for i in range(loops_v):
        X_train_s, X_test_s, y_train_s, y_test_s = split_data(X_train[i], loops_t, seed)
        for j in range(loops_t):
            clf = grid_search(X_train_s[j], X_test_s[j], y_train_s[j], model, param_grid, scorer)
            cv_params.append(clf.cv_results_)
            
    df_total=pd.concat(map(lambda x: pd.DataFrame(x).set_index('params'), cv_params), axis=1)
    best_params = df_total['mean_test_score'].sum(axis=1).idxmax()
    return best_params
    

# report the evalation metrics to file
def write_evaluation_file(args, best_prediction, hyper_params, time, all_features, sel):
    r = 5
    bp = np.array(best_prediction,dtype=object)
    with open(args.output+'.txt', 'w') as fh:
        fh.write('Script parameters:\n')
        fh.write(f'Input file: {args.input}\nClassifier: {args.classifier}\n')
        fh.write(f'Validation loops: {args.loops_validation}\nTuning loops: {args.loops_tuning}\nTuning scorer: {args.scorer}\n')
        fh.write(f'Varience threshold: {args.varience_threshold}\n')
        fh.write(f'Selected features: {sel} ({all_features - sel} features removed)\n')
        fh.write(f'Execution time: {round(time/60, 3)} minutes')
        fh.write('\nBest hyperparameters:\n')
        fh.write(f'{hyper_params}\n')       
        fh.write('\nMean values of evaluation runs with standard deviation:\n')     
        fh.write(f'ROC: {round(np.mean(bp[:,0]), r)}\t{round(np.std(bp[:,0]), r)} \n')
        fh.write(f'Accuracy: {round(np.mean(bp[:,1]), r)}\t{round(np.std(bp[:,1]), r)} \n')
        fh.write(f'Precision: {round(np.mean(bp[:,2]), r)}\t{round(np.std(bp[:,2]), r)} \n')
        fh.write(f'Recall: {round(np.mean(bp[:,3]), r)}\t{round(np.std(bp[:,3]), r)} \n')
        fh.write(f'F1-Score: {round(np.mean(bp[:,4]), r)}\t{round(np.std(bp[:,4]), r)} \n')
        fh.write(f'MCC: {round(np.mean(bp[:,5]), r)}\t{round(np.std(bp[:,5]), r)} \n')
        fh.write(f'Balanced accuracy: {round(np.mean(bp[:,6]), r)}\t{round(np.std(bp[:,6]), r)} \n')

        fh.write('\nAll evaluation scores:\nROC\tAccuracy\tPrecision\tRecall\tF1-Score\tMCC\tBalanced accuracy\n')
        [fh.write(f'{round(i[0], r)}\t{round(i[1], r)}\t{round(i[2], r)}\t{round(i[3], r)}\t{round(i[4], r)}\t{round(i[5], r)}\t{round(i[6], r)} \n') for i in best_prediction]
        fh.write('\nConfusion Matrices:\n')
        [fh.write(f'TP: {i[7][0]}\tFP: {i[7][1]}\tFN: {i[7][2]}\tTN: {i[7][3]}\n') for i in best_prediction]


# report sorted feature importances of rf to csv file
def write_fi_file(feature_importance, output, df_index):
    fi_mean = np.mean(feature_importance, axis=0)
    df_fi = pd.DataFrame(fi_mean, index=df_index, columns=['Feature Importance'])
    df_fi = df_fi.sort_values('Feature Importance', ascending=False)
    df_fi.to_csv(output+'_feature_importance.csv')


def main():
    start_time = time.time()
    args=parseargs()
    seed = int(args.seed)
    loops_validation = int(args.loops_validation)
    loops_tuning = int(args.loops_tuning)
    set_seed(seed)
    if args.scorer == 'MCC':
        scorer=make_scorer(matthews_corrcoef)
    else:
        scorer=args.scorer

    df=pd.read_csv(args.input, index_col=[0,1], header=0)
    all_features = df.shape[1]
    sel = VarianceThreshold(float(args.varience_threshold))
    sel_cols = sel.fit(df).get_support(indices=True)
    df = df.iloc[:,sel_cols]

    X_train, X_test, y_train, y_test = split_data(df, loops_validation, seed)

    # Set grid for Random Forrest Classifier 
    if args.classifier == 'RF':
        param_grid = {
            'n_estimators': [100, 500, 700, 1000],
            'max_depth': [2, 6, 10, None],
            'max_features': [0.33, 0.5, 1, 100, 'auto'],
            'min_samples_split': [2, 3],
            'criterion': ['gini', 'entropy'],
            'class_weight': ['balanced', 'balanced_subsample', None]
            }
        clf = RandomForestClassifier(random_state=seed)
        
    # set grid for SVM
    if args.classifier == 'SVM':
        param_grid = {
            'kernel': ['linear', 'poly'],# 'rbf', 'sigmoid'],
            'C': [0.25, 0.5, 0.75, 1.0]#, 1.25, 1.5, 1.75, 2.0, 3.0, 4.0, 5.0]
            }
        clf = SVC(random_state=seed, probability=True)
        
    if args.classifier == 'L2linear':
        param_grid = {
            'dual': [True, False],
            'tol': [0.001, 0.0001, 0.00001, 0.000001],
            'C': [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 3.0, 4.0, 5.0],
            'solver': ['liblinear', 'newton-cg', 'lbfgs', 'sag', 'saga'],
            'max_iter': [50, 100, 200, 500, 1000]
            }
        clf = LogisticRegression(random_state=seed)
    
    if args.classifier == 'XGB':
        param_grid = {
            'n_estimators': [5, 10, 50],
            'max_depth': [2, 10, None],
            'learning_rate': [0.3, 0.2, 0.1],
            'gamma': [0, 1, 5],
            'min_child_weight': [1, 3, 6],
            'subsample': [0.5, 0.6, 0.7, 1],
            'colsample_bytree': [0.5, 0.6, 0.7, 1]
            }
        clf = xgb.XGBClassifier(objective='binary:logistic', random_state=seed)

    # Hyperparameter tuning
    best_params = param_fitting(X_train, X_test, y_train, y_test, loops_validation, loops_tuning, seed, clf, param_grid, scorer)
    proba, pred, feature_importance = [], [], []
    if args.classifier == 'RF':
        best_clf = RandomForestClassifier(**best_params, random_state=seed)
    if args.classifier == 'SVM':
        best_clf = SVC(**best_params, random_state=seed, probability=True)
    if args.classifier == 'L2linear':
        best_clf = LogisticRegression(**best_params, random_state=seed)
    if args.classifier == 'XGB':
        best_clf = xgb.XGBClassifier(**best_params, objective='binary:logistic', random_state=seed)
        
    best_prediction, roc = [], []
    # Validation, evaluate performance and plot ROC
    for i in range(loops_validation):
        best_clf.fit(X_train[i], y_train[i])
        proba = best_clf.predict_proba(X_test[i])[:,1]
        pred = best_clf.predict(X_test[i])
        best_prediction.append(evaluate_performance(y_test[i], pred, proba))
        fpr, tpr, _ = roc_curve(y_test[i], proba)
        roc.append([i, fpr, tpr])

        if args.classifier == 'RF' or args.classifier == 'XGB':
            feature_importance.append(best_clf.feature_importances_)        
            
    # console output
    finish_time = time.time()-start_time
    print(f'Mean of ROC: {round(np.mean(np.array(best_prediction)[:,0]), 5)}')
    print(f'--- {round(finish_time, 3)} seconds ---')

    # Writing metrics to output file
    write_evaluation_file(args, best_prediction, best_params, finish_time, all_features, df.shape[1])
    if args.classifier == 'RF' or args.classifier == 'XGB':
        write_fi_file(feature_importance, args.output, df.columns)
    with open(args.output+'_roc.txt', 'w') as fh:
        [fh.write(f'Loop {i[0]}\nFPR\n{i[1]}\nTPR\n{i[2]}\n') for i in roc]
    
    # fit clf on all data, pickle with joblib to export clf 
    best_clf_fit = best_clf.fit(df, df.index.get_level_values('disease').values)
    dump(best_clf_fit, args.output+'.joblib', compress=9)
    
    
if __name__ == '__main__':
    main()