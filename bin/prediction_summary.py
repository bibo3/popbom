#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 12:38:02 2021

@author: matt
"""

import sys
import pandas as pd


def single_df(file):
    with open(file, 'r') as f:

        meta_name, meta_val = [], []
        metric_name, metric_val = [], []
        prev = 'not empty'
        meta_flag = False
        for line in f:
            if prev != '\n' and not meta_flag:
                # meta info, which file, classifier, taxo profiler, loops etc
                meta = line.strip().split(': ')
                if len(meta) == 2:
                    if meta[0] == 'Input file':
                        meta_name.append('Experiment')
                        meta_val.append(file.replace('.txt', '').replace('_', ' '))

                    else:
                        meta_name.append(meta[0])
                        meta_val.append(meta[1])
                elif len(meta) > 2:
                    meta_name.append('Best Hyperparameters')
                    meta_val.append(line.strip())
            elif prev != '\n':
                # performance metricies
                metric = line.strip().split(': ')
                if len(metric) == 2:
                    metric_name.append(metric[0])
                    metric_name.append(metric[0]+'_std')
                    metric_val.append(metric[1].split()[0])
                    metric_val.append(metric[1].split()[1])
            elif prev == '\n' and meta_flag:
                break
            else:
                meta_flag = True
            prev = line
            
    index = pd.MultiIndex.from_tuples([tuple(meta_val)], names=meta_name)
    df = pd.DataFrame([metric_val], columns=metric_name, index=index)
    return df


def main():
    path = sys.argv[1].split()
    files = [x for x in path if '_roc.txt' not in x]
    files = [x for x in files if '.txt' in x]
    df = pd.concat([single_df(x) for x in files])
    df.to_csv('prediction_summary.csv')
    

if __name__ == '__main__':
    main()
    