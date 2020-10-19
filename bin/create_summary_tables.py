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
#    kraken_total=kraken_total[kraken_total.iloc[:,[1]].notnull()]
    return kraken_total.dropna()


# reading in metaphlan reports
def reading_metaphlan(basepath, metadata, species):
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

    return metaphlan_total.fillna(0)


# reading in marker based metaphlan reports
def reading_mpa_marker(basepath, metadata):
    # setting glob path    
    m_path =  glob.glob(basepath + "/*/metaphlan_marker_report.txt")
    # clade names become column names, filenames the index 
    metaphlan_total = pd.concat(
        map(lambda file: 
            pd.read_csv(file, 
                        sep='\t', 
                        skiprows=4, 
                        names=('marker_name', file.split('/')[-2]), 
                        index_col='marker_name').T, 
            m_path))
    df_metadata = pd.read_csv(metadata, index_col=0)
    metaphlan_total = pd.concat([metaphlan_total, df_metadata], axis=1)
    metaphlan_total = metaphlan_total.set_index([metaphlan_total.index, 'disease'])
    return metaphlan_total.fillna(0)


def main():
    # read in reports and write to a single file
    parser = argparse.ArgumentParser(description='Run RF with kraken2 or metaphlan')
    parser.add_argument('--taxo', '-t', choices=['metaphlan', 'kraken2', 'centrifuge', 'mpa_marker'], required=True, help='which taxonomic profiler is used?')
    parser.add_argument('--directory', '-d', required=True, help='directory containing the reports in individual directories with their name corresponding to sample name')
    parser.add_argument('--outdir', '-o', help='directory for output')
    parser.add_argument('--metadata', '-m', help='metadata file')
    parser.add_argument('--species_filter', '-s', help='filter to species level?', default=False)
    args = parser.parse_args()
    
    if args.taxo == 'metaphlan':
        mpa = reading_metaphlan(args.directory, args.metadata, args.species_filter)
        mpa.to_csv(args.outdir+'/metaphlan_table.csv')
        
    if args.taxo == 'kraken2':
        kraken = reading_kraken2(args.directory, args.metadata, args.species_filter)
        kraken.to_csv(args.outdir+'/kraken2_table.csv')
        
    if args.taxo == 'mpa_marker':
        mpa = reading_mpa_marker(args.directory, args.metadata)
        mpa.to_csv(args.outdir+'/metaphlan_marker_table.csv')


if __name__ == '__main__':
    main()
