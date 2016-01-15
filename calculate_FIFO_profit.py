"""
Module: Calculate First-In-First-Out (FIFO) profits from account transaction data

Author: Peter Lee (mr.peter.lee@hotmail.com)
Last update: 2016-Jan-15

This module calculates the investment profit/loss of each sell in the account transaction history using the First-In-First-Out (FIFO) method.

A number of checks are in place to improve data quality:
    1.) Cash dividends are assumed to be fully reinvested by default.
    2.)


"""

import numpy as np
import pandas as pd
import os.path


def check_missing_in_data_stocks(data_stocks):
    """
    Remove missing entrieces from the End-Of-Day stocks data.
    Notes:
        The variables in this DataFrame are:
        ['isin', 'date', 'ajexdi', 'div', 'prccd']
    """

    missing_isin = len(data_stocks[data_stocks['isin'] == ""])
    print("Number of missing isin entries in the stocks price data: {}".format(missing_isin))
    print("These missing entries are dropped.")
    data_stocks.drop(data_stocks.index[data_stocks['isin'] == ""], inplace=True)
    print("\n")
    # Checking zero dividend payments
    print("Number of zero-dividends paymetns: {}".format(np.sum(data_stocks['div'] == 0)))
    print("zero-dividend payments are replaced as missing.")
    data_stocks.ix[data_stocks['div'] == 0, 'div'] = np.nan
    print("\n")
    # Check missing prccd
    print("Number of missing prccd entries: {}".format(np.sum(np.isnan(data_stocks['prccd']))))
    print("They are dropped")
    data_stocks.drop(data_stocks.index[np.isnan(data_stocks['prccd'])], inplace=True)
    print("\n")
    # Check missing ajexdi
    print("Number of missing ajexdi entries: {}".format(np.sum(np.isnan(data_stocks['ajexdi']))))
    print("They are dropped")
    data_stocks.drop(data_stocks.index[np.isnan(data_stocks['ajexdi'])], inplace=True)

    return data_stocks


def add_div_index(group):
    """
    Added a dividend index column to the group (isin EOD df).
    """
    # Replace NANs in "div" as ZERO
    group.ix[group['div'].isnull(), 'div'] = 0
    group['divIndex'] = (group['div'] / group['prccd'].shift(1) + 1).cumprod()
    # Set the value in the first row
    group.ix[group.index[0], 'divIndex'] = 1
    return group


def compute_dividend_index(data_stocks):
    """
    Calculate the dividend index to show the cumulative effects of dividends payments. The index begins with 1.
    """
    data_stocks['divIndex'] = np.nan
    data_stocks.sort_values(by=['isin', 'date'], inplace=True)
    return data_stocks.groupby(by='isin', as_index=False).apply(add_div_index)


def add_mod_ajexdi(group):
    """
    Added a modified ajexdi column to the group
    """
    # Calculate the change factor
    group['change_factor'] = group['ajexdi'] / group['ajexdi'].shift(1) *\
        group['divIndex'] / group['divIndex'].shift(1)
    group.ix[group.index[0], 'change_factor'] = group.ix[group.index[0], 'ajexdi']
    group['modAjexdi'] = group['change_factor'].cumprod()
    group.drop('change_factor', axis=1, inplace=True)
    return group


def calculate_mod_ajexdi(data_stocks):
    """
    Calculate the modified ajexdi from divIndex per isin.
    """
    return data_stocks.groupby(by='isin', as_index=False).apply(add_mod_ajexdi)


def data_main_add_seq(grp):
    """Add the variable 'seq' to indicate the order of transactions within a day"""
    # Generate an index for multiple transactions within the same day to preserve their order
    grp['seq'] = np.arange(1, len(grp) + 1)
    return grp


def data_main_fill_na(grp):
    lst_vars = ['ajexdi', 'prccd', 'divIndex']
    for x in lst_vars:
        grp[x] = grp[x].fillna(method='ffill')
        grp[x] = grp[x].fillna(method='bfill')
    return grp


if __name__ == "__main__":

    # Data files
    # ==========
    data_folder = "data/new/"
    filename_transactions = "fin_mf_trades_1995_2014_fid.dta"
    filename_stocks = "compustat_finland_1990_2015_all.dta"
    Adjust_Price_Before_1999 = True

    # Loading data_stocks
    # ===================
    data_stocks = pd.read_stata(os.path.join(data_folder, filename_stocks))
    data_stocks['date'] = pd.to_datetime(data_stocks['date'], format="%Y%m%d")
    data_stocks = data_stocks[['isin', 'date', 'ajexdi', 'div', 'prccd']]
    # Check missing data in the data_stocks
    data_stocks = check_missing_in_data_stocks(data_stocks)
    # Calculate cumulative dividend index
    data_stocks = compute_dividend_index(data_stocks)
    # Modify ajexdi to include dividend payment (the original ajexdi is assumed to be net of dividends)
    data_stocks = calculate_mod_ajexdi(data_stocks)
    data_stocks.head()

    # Load data_transactions
    # ======================
    data_main = pd.read_stata(os.path.join(data_folder, filename_transactions))
    data_main.columns = [var.lower() for var in data_main.columns]
    # There are several datetime variables -> remove the unused ones
    data_main = data_main.drop(['date', 'tr_date', 'trans_date'], axis=1)
    # Convert trade_date to datetime
    data_main['date'] = pd.to_datetime(data_main.trade_date)
    date_main = data_main.sort_index(axis=1)
    data_main = data_main.groupby(by=['fund_id', 'isin', 'date'], as_index=False).apply(data_main_add_seq)
    date_main = data_main.sort_values(by=['fund_id', 'isin', 'date', 'seq', 'buy_sell'])
    data_main = data_main[['fund_id', 'isin', 'date', 'seq', 'acc', 'sect', 'owntype', 'legtype', 'ref_code', 'buy_sell', 'volume', 'price']]
    # Merge data_stocks with data_transactions, and impute missing prices.
    data_main = data_main.merge(data_stocks, how='left', on=['isin', 'date'])
    # Inspect missing data
    print(data_main.isnull().sum())
    # Use forward fill to impute ajexdi; prccd, divIndex;
    data_main = data_main.groupby(by=['fund_id', 'isin'], as_index=False).apply(data_main_fill_na)
    if Adjust_Price_Before_1999:
        data_main.ix[data_main['date'] < "1999-Jan-01", 'price'] = data_main['price'] / 5.94573
    # Generate variables
    data_main['buy'] = (data_main.buy_sell == 10) | (data_main.buy_sell == 11)
    data_main['sell'] = (data_main.buy_sell == 20) | (data_main.buy_sell == 21)
    data_main['adjSellVolume'] = 0
    data_main.ix[data_main['sell']==True, 'adjSellVolume'] = data_main['volume']
    data_main['adjBuyVolume'] = 0
    data_main.ix[data_main['buy']==True, 'adjBuyVolume'] = data_main['volume']
    data_main['cumulativeVolume'] = 0

    # check if there is any missing values in ajexdi and divIndex
    lst_vars = ['ajexdi', 'divIndex', 'price', 'buy_sell', 'volume']
    print("Number of obs in data: {}".format(len(data_main)))
    for x in lst_vars:
        print("missing values in {}: {}".format(x, np.sum(np.isnan(data_main[x]))))

    print("Need to get rid of missing ajexdi or divIndex (mostly the outcome of expanding the data)")
    data_main = data_main.ix[data_main['ajexdi'].notnull()].ix[data_main['divIndex'].notnull()]

