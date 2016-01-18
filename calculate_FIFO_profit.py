"""
Module: Calculate First-In-First-Out (FIFO) profits from account transaction data

Author: Peter Lee (mr.peter.lee@hotmail.com)
Last update: 2016-Jan-15

This module calculates the investment profit/loss of each sell in the account transaction history using the First-In-First-Out (FIFO) method.

A number of checks are in place to improve data quality:
    1.) Cash dividends are assumed to be fully reinvested by default.
    2.)
"""
##
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
    group['change_factor'] = group['ajexdi'] / group['ajexdi'].shift(1) /\
        group['divIndex'] * group['divIndex'].shift(1)
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
    lst_vars = ['modAjexdi', 'prccd', 'divIndex']
    for x in lst_vars:
        grp[x] = grp[x].fillna(method='ffill')
        grp[x] = grp[x].fillna(method='bfill')
    return grp


def calculate_fifo_profit(data_main):
    """
    Correct any "over-sale" transactions.
    Calculate FIFO profits.
    """
    return data_main.groupby(by=['fund_id', 'isin'], as_index=False).apply(data_main_add_profit)


def data_main_add_profit(group):
    """
    Add a new column - FIFO profit; while correct any "over-sale" transactions.
    """

    # Setting up some test data
    # gp = data_main.groupby(by=['fund_id', 'isin'], as_index=False)
    # temp = gp.get_group(('FI0008804422', 'FI0009000707'))
    # group = temp.copy()
    # group = temp.ix[fuck.index[:15], ['fund_id', 'isin', 'date', 'buy_sell', 'volume', 'price', 'modAjexdi']]
    # group

    # Identify all buys
    buys = group.ix[(group['buy_sell'] == 10) | (group['buy_sell'] == 11), ['date', 'price', 'volume', 'modAjexdi']]

    # position (stock volume)
    group['position'] = np.nan  # Stock volumes in position
    group['trans'] = np.nan  # Transaction volume after correcting over-sale
    group['profit'] = 0  # Fifo-profit
    # Calculate positions
    previous_ajexdi = None
    previous_position = None
    position = 0
    ajexdi = 0
    volume = 0
    order = None

    # ith_sell = 1
    for i, row in group.iterrows():
        ajexdi = row['modAjexdi']
        volume = row['volume']
        # Identify order type
        if (row['buy_sell'] == 10) | (row['buy_sell'] == 11):
            order = 1
        elif (row['buy_sell'] == 20) | (row['buy_sell'] == 21):
            order = -1
        else:
            order = 0

        # 1st row
        if previous_position is None:
            # if there is a sell or no trans
            group.ix[i, 'trans'] = 0
            group.ix[i, 'position'] = 0
            previous_position = 0
            previous_ajexdi = row['modAjexdi']
            # if there is a buy
            if (order == 1):
                group.ix[i, 'trans'] = volume
                group.ix[i, 'position'] = volume
                previous_position = row['volume']
            continue

        # >1st row
        # Convert position before buy/sell
        position = previous_position / ajexdi * previous_ajexdi

        # Check if there is a over-sale
        if (order == -1) & (volume > position):
                volume = position

        # Update position after buy/sell
        position += volume * order
        group.ix[i, 'trans'] = (volume if order >= 0 else volume * order)
        group.ix[i, 'position'] = position
        # Update previous_row
        previous_ajexdi = row['modAjexdi']
        previous_position = position

        # Calculate profit if there is a sell
        if (order == -1):
            # print('Sell {} at index {}'.format(ith_sell, row.name))
            # ith_sell += 1
            sell_revenue = volume * row['price']
            sell_index = row.name
            buy_index = 0
            purchase_cost = 0
            # The correct sell volume is "volume"
            # k = 1
            while volume >= 1 and buy_index < sell_index:
                # print('k is', k)
                # print(buys)
                # k += 1
                for j, buy in buys.iterrows():
                    buy_index = buy.name
                    buy_volume_at_sell = buy['volume'] * buy['modAjexdi'] / row['modAjexdi']
                    # print('j is', j, 'Cvt_buy_vol', buy_volume_at_sell)
                    if volume > buy_volume_at_sell:
                        volume -= buy_volume_at_sell
                        buys.ix[j, 'volume'] = 0
                        purchase_cost += buy['volume'] * buy['price']
                    else:
                        offset_vol = (volume * row['modAjexdi'] / buy['modAjexdi'])
                        buys.ix[j, 'volume'] = buy['volume'] - offset_vol
                        purchase_cost += offset_vol * buy['price']
                        volume = 0

            group.ix[i, 'profit'] = sell_revenue - purchase_cost
    return group


def expand_data(data_main):
    """
    Expand the holdings dataset so that that there is one End-of-Day entry for the entire holding period.
    """
    data_main = data_main.groupby(by=['fund_id', 'isin'], as_index=False).apply(add_eod_rows_to_holding_period)
    data_main.reset_index(drop=True, inplace=True)
    return data_main


def add_eod_rows_to_holding_period(group):

    # Create time series
    group.index = group['date']
    group.drop('date', axis=1, inplace=True)

    # Resample to End-of-Day data
    transform = {
        'agg_volume': group['volume'].resample('D', how='sum', fill_method='ffill', label='right', closed='right'),
        'num_acts': group['buy_sell'].resample('D', how='count', fill_method='ffill', label='right', closed='right'),
        'avg_price': group['price'].resample('D', how='mean', fill_method='ffill', label='right', closed='right'),
        'avg_ajexdi': group['modAjexdi'].resample('D', how='mean', fill_method='ffill', label='right', closed='right'),
        'position': group['position'].resample('D', how='last', fill_method='ffill', label='right', closed='right'),
        'trans': group['trans'].resample('D', how='sum', fill_method='ffill', label='right', closed='right'),
        'profit': group['profit'].resample('D', how='sum', fill_method='ffill', label='right', closed='right')
    }

    # Up-sample
    ts_daily = pd.DataFrame(transform)

    # Deal with string variables specifically
    lst_fill = ['fund_id', 'isin']
    for x in lst_fill:
        ts_daily[x] = group.ix[0, x]

    ts_daily.reset_index(inplace=True)
    ts_daily.rename(columns={'index': 'date'}, inplace=True)
    ts_daily.head()

    return ts_daily
##

if __name__ == "__main__":

    # Data files
    # ==========
    data_folder = "data/new/"
    filename_transactions = "fin_mf_trades_1995_2014_fid.dta"
    filename_stocks = "compustat_finland_1990_2015_all.dta"
    debug = True
    debug_file = "test1.xlsx"
    Account_For_Dividend = True
    Adjust_Price_Before_1999 = True

    if debug:
        test_folder = "data/tests/"

        # Test 1
        test_file = "test1.xlsx"
        test = pd.read_excel(os.path.join(test_folder, test_file))

        test = calculate_fifo_profit(test)
        test.to_excel(os.path.join(test_folder, test_file.split('.')[0] + "_output.xlsx"))
        test = expand_data(test)
        test.to_excel(os.path.join(test_folder, test_file.split('.')[0] + "_output2.xlsx"))

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
    if Account_For_Dividend:
        data_stocks = calculate_mod_ajexdi(data_stocks)
    else:
        data_stocks['modAdjexi'] = data_stocks['ajexdi']

##
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
    if Adjust_Price_Before_1999:
        data_main.ix[data_main['date'] < "1999-Jan-01", 'price'] = data_main['price'] / 5.94573
    # Merge data_stocks with data_transactions, and impute missing prices.
    data_main.drop('prccd', inplace=True, axis=1)
    data_main = data_main.merge(data_stocks, how='left', on=['isin', 'date'])
    data_main = data_main[['fund_id', 'isin', 'date', 'seq', 'acc', 'sect', 'owntype', 'legtype', 'ref_code', 'buy_sell', 'volume', 'price', 'prccd', 'divIndex', 'modAjexdi', 'div']]
    data_main.head()
##

    # Inspect missing data
    # check if there is any missing values in ajexdi and divIndex
    lst_vars = ['modAjexdi', 'divIndex', 'price', 'buy_sell', 'volume']
    print("Number of obs in data: {}".format(len(data_main)))
    for x in lst_vars:
        print("missing values in {}: {}".format(x, np.sum(np.isnan(data_main[x]))))

    # Use forward fill to impute ajexdi; prccd, divIndex;
    data_main = data_main.groupby(by=['fund_id', 'isin'], as_index=False).apply(data_main_fill_na)
    print("Need to get rid of missing ajexdi or divIndex.")
    data_main = data_main.ix[data_main['modAjexdi'].notnull()].ix[data_main['divIndex'].notnull()]

    # Check for over-sales and compute holdings
    # Some funds may "over-sale" their stocks and causes negative positions. This is prohibited. Any sells that cause negative balance are lowered to result in a zero balance.
    data_main = data_main.sort_values(by=['fund_id', 'isin', 'date', 'seq']).reset_index(drop=True)
##

    data_main = calculate_fifo_profit(data_main)
    # Expand the data_main - to fill in daily holdings in between transaction records.
    data_main = expand_data(data_main)
