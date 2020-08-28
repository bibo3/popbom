#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 17:57:37 2020

@author: matt
"""
import pandas as pd
import glob
import argparse

# reading in kraken2 reports
def reading_kraken2(basepath):
    # setting glob path
    k_path = glob.glob(basepath + "/*/kraken2_report.txt")
    # filenames become the index
    kraken_total = pd.concat(
        map(lambda file: 
            pd.read_csv(file, 
                        sep='\t', 
                        names=('rel_ab', file.split('/')[-2], 'assigned', 'rank', 'tax_id', 'sci_name'), 
                        usecols=(1,3,4,5), 
                        index_col=('tax_id','sci_name','rank')).T,
            k_path))
    
    # total values of abundances (unassigned+root) 
    total_ab_kraken = kraken_total.loc[:,[0, 1]].sum(axis=1)
    # filter so that only species remain and drop rank column afterwards
    kraken_total = kraken_total.loc[:,kraken_total.columns.get_level_values(2).isin(['S'])].droplevel('rank', axis=1)
    # relative abundances
    kraken_total = kraken_total.div(total_ab_kraken, axis=0)
    return kraken_total.fillna(0)


# reading in metaphlan reports
def reading_metaphlan(basepath):
    # setting glob path    
    m_path =  glob.glob(basepath + "/*/metaphlan_report.txt")
    # clade names become column names, filenames the index 
    metaphlan_total = pd.concat(
        map(lambda file: 
            pd.read_csv(file, 
                        sep='\t', 
                        skiprows=4, 
                        names=('clade_name', 'path', file.split('/')[-2], 'add_clades'), 
                        usecols=(0,2), 
                        index_col='clade_name').T, 
            m_path))
    # filter that only species remain
    metaphlan_total = metaphlan_total.filter(like='|s__')
    # rename columns for better readability
    metaphlan_total = metaphlan_total.rename(columns=lambda x: x.split('|s__')[1])
    return metaphlan_total.fillna(0)


def main():
    # read in reports and write to a single file
    parser = argparse.ArgumentParser(description='Run RF with kraken2 or metaphlan')
    parser.add_argument('--taxo', '-t', choices=['metaphlan', 'kraken2', 'centrifuge'], required=True, help='which taxonomic profiler is used?')
    parser.add_argument('--directory', '-d', required=True, help='directory containing the reports in individual directories with their name corresponding to sample name')
    parser.add_argument('--outdir', '-o', help='directory for output')
    args = parser.parse_args()
    
    if args.taxo == 'metaphlan':
        mpa = reading_metaphlan(args.directory)
        mpa.to_csv(args.outdir+'/metaphlan_table.csv')
        
    if args.taxo == 'kraken2':
        kraken = reading_kraken2(args.directory)
        kraken.to_csv(args.outdir+'/kraken2_table.csv')        

if __name__ == '__main__':
    main()