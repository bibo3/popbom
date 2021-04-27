#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 17:57:37 2020

@author: matt
"""
import pandas as pd
import argparse


# reading in kraken2 reports
def reading_kraken2(basepath, metadata, level):
    # filenames become the index
    kraken_total = pd.concat(
        map(lambda file: 
            pd.read_csv(file, 
                        sep='\t', 
                        names=('rel_ab', file[:-4], 'assigned', 'rank', 'tax_id', 'sci_name'), 
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

    if level == 'species':
        # filter so that only species remain and drop rank column afterwards
        kraken_total = kraken_total.loc[:,kraken_total.columns.get_level_values(2).isin(['S'])].droplevel('rank', axis=1)

    if level == 'genus':
        # filter so that only species remain and drop rank column afterwards
        kraken_total = kraken_total.loc[:,kraken_total.columns.get_level_values(2).isin(['G'])].droplevel('rank', axis=1)
        
    if metadata:
        df_metadata = pd.read_csv(metadata, index_col=0)
        kraken_total = pd.concat([kraken_total, df_metadata], axis=1)
        kraken_total = kraken_total.set_index([kraken_total.index, 'disease'])
    else:        
        kraken_total.columns = kraken_total.columns.to_series().apply(lambda x: "".join(str(x)))
        
    # rename columns for XGBoost
    kraken_total.columns = kraken_total.columns.to_series().apply(lambda x: "".join(str(x)).replace('[','(').replace(']',')').replace('<','_'))
    return kraken_total.dropna()


# reading in metaphlan reports
def reading_metaphlan(basepath, metadata, level):
    # clade names become column names, filenames the index 
    metaphlan_total = pd.concat(
        map(lambda file: 
            pd.read_csv(file, 
                        sep='\t', 
                        skiprows=4, 
                        names=('clade_name', 'path', file[:-8], 'add_clades'), 
                        usecols=(0,2), 
                        index_col='clade_name').T, 
            basepath.split()))

    if 'HV1' in metaphlan_total.index:
        metaphlan_total.index=metaphlan_total.index.str.replace('V','V-')
    if 'MetaHIT-MH0001' in metaphlan_total.index:
        metaphlan_total.index=metaphlan_total.index.str.replace('MetaHIT-M','M')
    if metadata:
        df_metadata = pd.read_csv(metadata, index_col=0)
        metaphlan_total = pd.concat([metaphlan_total, df_metadata], axis=1)
        metaphlan_total = metaphlan_total.set_index([metaphlan_total.index, 'disease'])
    metaphlan_total = metaphlan_total[metaphlan_total.k__Bacteria.notnull()]

    if level == 'species':
        # filter that only species remain
        metaphlan_total = metaphlan_total.filter(like='|s__')
        # rename columns for better readability
        metaphlan_total = metaphlan_total.rename(columns=lambda x: x.split('|s__')[1])
    if level == 'genus':
        # filter that only genus remain
        metaphlan_total = metaphlan_total.filter(like='|g__')
        metaphlan_total = metaphlan_total.drop(columns=metaphlan_total.filter(like='|s__'))
        # rename columns for better readability
        metaphlan_total = metaphlan_total.rename(columns=lambda x: x.split('|g__')[1])
    
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
                        names=('marker_name', file[:-11]), 
                        index_col='marker_name').T, 
            basepath.split()))
    if 'HV1' in metaphlan_total.index:
        metaphlan_total.index=metaphlan_total.index.str.replace('V','V-')
    if 'MetaHIT-MH0001' in metaphlan_total.index:
        metaphlan_total.index=metaphlan_total.index.str.replace('MetaHIT-M','M')
    
    if metadata:
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
    parser.add_argument('--combine', help='combine mpa report on species level with strain report')
    parser.add_argument('--metadata', '-m', help='metadata file')
    parser.add_argument('--filter_level', '-f', help='select level to filter to', choices=['species', 'genus'])
    parser.add_argument('--species')
    parser.add_argument('--genus')
    args = parser.parse_args()
    
    if args.metaphlan:
        if args.species:
            df = reading_metaphlan(args.metaphlan, args.metadata, 'species')
            df.to_csv('metaphlan_species_table.csv')
        if args.genus:
            df = reading_metaphlan(args.metaphlan, args.metadata, 'genus')
            df.to_csv('metaphlan_genus_table.csv')
            
        df = reading_metaphlan(args.metaphlan, args.metadata, '')
        df.to_csv('metaphlan_table.csv')
        
    if args.kraken2:
        if args.species:
            df = reading_kraken2(args.kraken2, args.metadata, 'species')
            df.to_csv('kraken2_species_table.csv')
        if args.genus:
            df = reading_kraken2(args.kraken2, args.metadata, 'genus')
            df.to_csv('kraken2_genus_table.csv')

        df = reading_kraken2(args.kraken2, args.metadata, args.filter_level)
        df.to_csv('kraken2_table.csv')
        
    if args.mpa_marker:
        df = reading_mpa_marker(args.mpa_marker, args.metadata)
        df.to_csv('strain_table.csv')        
    if args.combine:
        mpa = reading_metaphlan(args.metaphlan, args.metadata, 'species')
        marker = reading_mpa_marker(args.mpa_marker, args.metadata)
        df = pd.concat([mpa, marker], axis=1)
        df.to_csv('combined_table.csv')


if __name__ == '__main__':
    main()
