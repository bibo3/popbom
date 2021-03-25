#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 08:59:13 2020

@author: matt
"""

import pandas as pd
import glob
import matplotlib.pyplot as plt
import numpy as np

directory = '/home/matt/master/results_ml'
test = '/ibd/rf/mpa/5v_2t_combined_feature_importance.csv'
#test = '/ibd/rf/mpa/5v_2t_genus_feature_importance.csv'
test = '/ibd/rf/mpa/mpa_10x10_feat_sel_feature_importance.csv'
test = '/cirrhosis/rf/mpa/balanced_varience_sel_10x_feature_importance.csv'


subdir1 = glob.glob(directory+'/*/*/*/*_feature_importance.csv')
subdir2 = glob.glob(directory+'/*/*/*_feature_importance.csv')


def plot_fi(list_of_paths, n):
    for i in list_of_paths:
        df = pd.read_csv(i)
        fig, ax = plt.subplots(figsize=(12,8))
        ax.set_title('Feature Importance')
        ax.bar(df.iloc[:,0].head(n), df.iloc[:,1].head(n))
        plt.xticks(rotation=90)
        plt.tight_layout()
        fig.savefig(i[:-3]+'png')

# uncomment to plot the feature importance
#plot_fi(subdir1+subdir2, 20)


def plot_roc(path):
    # reading in fpr and tpr from file
    fpr, tpr = [],[]
    with open(path, 'r') as file:
        f = file.read().replace('\n','')
        fpr_tpr_list = [l.split(']')[0] for l in f.split('[')[1:]]
        for n, i in enumerate(fpr_tpr_list):
            el = i.split()
            el = [float(j) for j in el]
            if n%2 == 0:
                fpr.append(el)
            else:
                tpr.append(el)
    
    # the actual plotting
    fig, ax = plt.subplots(figsize=(12,8))
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
            label='Chance', alpha=.8)
    for n, i in enumerate(tpr):
        ax.plot(fpr[n], tpr[n], color='grey',
            label=f'ROC fold {n}', lw=2, alpha=.3) # keep label?
    
    # ups
    max_len = max([len(x) for x in fpr])
    for i in fpr:
        while len(i) < max_len: i.append(i[-1])
    for i in tpr:
        while len(i) < max_len: i.append(i[-1])
    
    mean_fpr = np.mean(fpr, axis=0)
    mean_tpr = np.mean(tpr, axis=0)
    ax.plot(mean_fpr,mean_tpr, color='b', lw=2, alpha=.8, label='mean')
    # insert name of plot below ?
    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
           title="Receiver operating curve")
    std_tpr = np.std(tpr, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                    label=r'$\pm$ 1 std. dev.')
    ax.legend(loc="lower right")
    # save the figure
    fig.savefig(path[:-3]+'png')


# loop over paths for execution
dir1 = glob.glob(directory+'/*/*/*/*_roc.txt') 
dir2 = glob.glob(directory+'/*/*/*_roc.txt')

#test = '/obesity/rf/mpa/5v_2t_species_roc.txt'
#plot_roc(directory+test)
for i in dir1+dir2:
    plot_roc(i)
