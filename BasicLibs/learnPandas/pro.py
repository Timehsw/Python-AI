# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018/10/23
    Desc : 
    Note : 
'''

import pandas as pd
from BasicLibs.learnPandas.part2 import part2


def main():
    """Main function"""

    """Read in data"""
    # Read in and clean up data
    df_messy = pd.read_csv("data/under5mortalityper1000.csv")
    df_clean = df_messy.dropna(axis=1, how='all').dropna(axis=0, how='all')

    # Remove missing entries
    df_data = df_clean.dropna(axis=0, how='any')

    # Separate country names and data
    df_countries = df_data['Country']
    df_deaths = df_data.drop('Country', axis=1)



    """Part 2"""
    part2(df_deaths, df_countries)



if __name__ == "__main__":
    main()