#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 12:38:02 2021

@author: matt
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def single_df(file):
    table_name = {'mpa': 'metaphlan_table.csv', 'strain': 'metaphlan_marker_table.csv', 'kraken': 'kraken2_table.csv', 'combined': 'strain_species.csv'}
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
                        meta_val.append(file.split('/')[-1].replace('.txt', '').replace('_', ' '))
                        meta_name.append('taxo profiler')
                        meta_val.append(''.join([key for key, value in table_name.items() if value in meta[1]]))
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
            
    index = [file.split('/')[-1].replace('.txt', '').replace('_', ' ')]
    df = pd.DataFrame([metric_val], columns=metric_name, index=index)
    return df


def main():
    path = sys.argv[1][1:-1].split(', ')
    files = [x for x in path if '_roc.txt' not in x]
    df = pd.concat([single_df(x) for x in files])
    df = df.apply(pd.to_numeric)
    fig, ax = plt.subplots(figsize=(12,8))
    sns.heatmap(df[['ROC', 'Balanced accuracy', 'Precision', 'Recall', 'F1-Score', 'MCC']], annot=True, cmap="coolwarm", vmin=0, vmax=1, ax=ax)
    fig.savefig('heatmap_metrics.png')



if __name__ == '__main__':
    main()
    