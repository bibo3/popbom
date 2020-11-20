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
def reading_kraken2(basepath, metadata, species):
    # filenames become the index
    kraken_total = pd.concat(
        map(lambda file: 
            pd.read_csv(file, 
                        sep='\t', 
                        names=('rel_ab', file.split('.')[0], 'assigned', 'rank', 'tax_id', 'sci_name'), 
                        usecols=(1,3,4,5), 
                        index_col=('tax_id','sci_name','rank')).T,
            basepath.split()))
    
    if 'HV1' in kraken_total.index:
        kraken_total.index=kraken_total.index.str.replace('V','V-')
    if 'MetaHIT-MH0001' in kraken_total.index:
        kraken_total.index=kraken_total.index.str.replace('MetaHIT-M','M')

    # total values of abundances (unassigned+root) 
    total_ab_kraken = kraken_total.loc[:,[0, 1]].sum(axis=1)
    # relative abundances
    kraken_total = kraken_total.div(total_ab_kraken, axis=0)

    if species:
        # filter so that only species remain and drop rank column afterwards
        kraken_total = kraken_total.loc[:,kraken_total.columns.get_level_values(2).isin(['S'])].droplevel('rank', axis=1)

    df_metadata = pd.read_csv(metadata, index_col=0)
    kraken_total = pd.concat([kraken_total, df_metadata], axis=1)

    kraken_total = kraken_total.set_index([kraken_total.index, 'disease'])
    return kraken_total.dropna()


# reading in metaphlan reports
def reading_metaphlan(basepath, metadata, species):
    # clade names become column names, filenames the index 
    metaphlan_total = pd.concat(
        map(lambda file: 
            pd.read_csv(file, 
                        sep='\t', 
                        skiprows=4, 
                        names=('clade_name', 'path', file.split('.')[0], 'add_clades'), 
                        usecols=(0,2), 
                        index_col='clade_name').T, 
            basepath.split()))

    if 'HV1' in metaphlan_total.index:
        metaphlan_total.index=metaphlan_total.index.str.replace('V','V-')
    if 'MetaHIT-MH0001' in metaphlan_total.index:
        metaphlan_total.index=metaphlan_total.index.str.replace('MetaHIT-M','M')
    
    df_metadata = pd.read_csv(metadata, index_col=0)
    metaphlan_total = pd.concat([metaphlan_total, df_metadata], axis=1)
    metaphlan_total = metaphlan_total.set_index([metaphlan_total.index, 'disease'])
    metaphlan_total=metaphlan_total[metaphlan_total.k__Bacteria.notnull()]

    if species:
        # filter that only species remain
        metaphlan_total = metaphlan_total.filter(like='|s__')
        # rename columns for better readability
        metaphlan_total = metaphlan_total.rename(columns=lambda x: x.split('|s__')[1])
        
    # rename columns for XGBoost
    metaphlan_total = metaphlan_total.rename(columns=lambda x: x.replace('[','(').replace(']',')').replace('<','_'))
    return metaphlan_total.fillna(0)


# reading in marker based metaphlan reports
def reading_mpa_marker(basepath, metadata):
    # clade names become column names, filenames the index 
    metaphlan_total = pd.concat(
        map(lambda file: 
            pd.read_csv(file, 
                        sep='\t', 
                        skiprows=4, 
                        names=('marker_name', file.split('.')[0]), 
                        index_col='marker_name').T, 
            basepath.split()))
    if 'HV1' in metaphlan_total.index:
        metaphlan_total.index=metaphlan_total.index.str.replace('V','V-')
    if 'MetaHIT-MH0001' in metaphlan_total.index:
        metaphlan_total.index=metaphlan_total.index.str.replace('MetaHIT-M','M')
    
    df_metadata = pd.read_csv(metadata, index_col=0)
    metaphlan_total = pd.concat([metaphlan_total, df_metadata], axis=1)
    metaphlan_total = metaphlan_total.set_index([metaphlan_total.index, 'disease'])
    metaphlan_total = metaphlan_total.rename(columns=lambda x: x.replace('[','(').replace(']',')').replace('<','_'))
    return metaphlan_total.fillna(0)


def main():
    # read in reports and write to a single file
    parser = argparse.ArgumentParser(description='Create summary tables for kraken2 and metaphlan reports')
    parser.add_argument('--metaphlan', help='metaphlan report files to be summarized')
    parser.add_argument('--kraken2', help='kraken2 report files to be summarized')
    parser.add_argument('--mpa_marker', help='metaphlan strain report files to be summarized')
    parser.add_argument('--metadata', '-m', help='metadata file')
    parser.add_argument('--species_filter', '-s', help='filter to species level?', action='store_true')
    args = parser.parse_args()
    
    if args.metaphlan:
        df = reading_metaphlan(args.metaphlan, args.metadata, args.species_filter)
        df.to_csv('metaphlan_table.csv')
        
    if args.kraken2:
        df = reading_kraken2(args.kraken2, args.metadata, args.species_filter)
        df.to_csv('kraken2_table.csv')
        
    if args.mpa_marker:
        df = reading_mpa_marker(args.mpa_marker, args.metadata)
        df.to_csv('metaphlan_marker_table.csv')


if __name__ == '__main__':
    main()
