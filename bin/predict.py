#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 09:37:20 2020

@author: matt
"""

import argparse
import pandas as pd
#from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve, matthews_corrcoef, balanced_accuracy_score, make_scorer
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.svm import SVC
#from sklearn.linear_model import LogisticRegression
#import xgboost as xgb

from joblib import load

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--classifier')
    parser.add_argument('--input')
    parser.add_argument('--output')
    args = parser.parse_args()
    
    clf = load(args.classifier)
    df = pd.read_csv(args.input, index_col=0, header=0)

    pred = clf.predict(df)
    phenotype = {0: 'Healthy', 1: 'Disease'}

    [print(phenotype[i]) for i in pred]
    with open(args.output+'_prediction.tsv', 'w') as fh:
        fh.write('Sample\tPrediction\n')
        [fh.write(f'{list(df.index)[num]}\t{phenotype[i]}\n') for num, i in enumerate(pred)]


if __name__ == '__main__':
    main()
