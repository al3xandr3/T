# -*- coding: utf-8 -*-
"""
Created on Sat May 16 15:53:36 2020

@author: amatos
"""

import pandas as pd
import t.table as t
import operator as op

##################################
# Financial 
# https://www.youtube.com/watch?v=XLL9KGeQltQ&feature=youtu.be
# https://github.com/marekkolman/yt_backtest_trading_demo/blob/master/backtest.py

def get_quotes(symbol, date_from):
    import pandas as pd
    import pandas_datareader.data as web
    raw_data = web.DataReader(symbol, 'yahoo', pd.to_datetime(date_from), pd.datetime.now())
    data = raw_data.stack(dropna=False).reset_index().rename(columns = {'Symbols':'symbol', 'Date':'date'}).sort_values(by = ['symbol', 'date'])
    return data
# > prices = get_quotes(['SPY', '^GSPC', '^VIX'], date_from = '2000-01-01')



def get_quotes_close(symbol, date_from):
    import pandas as pd
    import pandas_datareader.data as web
    raw_data = web.DataReader(symbol, 'yahoo', pd.to_datetime(date_from), pd.datetime.now())
    data = raw_data.stack(dropna=False)['Adj Close'].to_frame().reset_index().rename(columns = {'Symbols':'symbol', 'Date':'date', 'Adj Close':'value'}).sort_values(by = ['symbol', 'date'])
    return pd.pivot_table(data, columns = 'symbol', index = 'date', values ='value')    
# > prices = get_quotes_close(['SPY', '^GSPC', '^VIX'], date_from = '2000-01-01')



# https://youtu.be/XLL9KGeQltQ    
def backtest_strategy(symbol_price, trade_orders, capital):
    """
    Calculates the lift of buy and sell orders on a symbol 

    Parameters
    ----------
    symbol_price : df
        Columns: index as date, the price of the symbol 
    trade_orders : df
        Columns: index as date, order:(buy, sell, stay), capital for the order (0 for stay)
    capital : int
        Starting capital to use

    Returns
    -------
    output : (results: df, lift: float)
        columns: index as date, total gains
    """
    
    symbol = symbol_price.columns[1]
    df = t.sort(pd.merge(symbol_price, trade_orders, on=["date"], how="outer", indicator=True), "date", ascending=True)
    
    df["date"] = pd.to_datetime(df["date"])
    
    df["pct_change"] = df[symbol].pct_change()
    df["invested_start_day"] = 0
    df["invested_end_day"] = 0
    df["account_cash_start_day"] = 0
    df["account_cash_end_day"] = 0  
    df["net_worth"] = 0
    df["nb"] = ""

    calendar = pd.Series( t.column(df, "date") )
    
    for date in calendar:
        
        invested_start_day, invested_end_day, account_cash_start_day, account_cash_end_day, net_worth = 0,0,0,0,0
        
        if t.column(df, "date")[0] == date: # first day
            invested_start_day = 0
            account_cash_start_day = capital

        else:
            invested_start_day = t.select(t.where(df, "date", date, op.lt), "invested_end_day").tail(1).values[0][0] + ( t.select(t.where(df, "date", date, op.lt), "invested_end_day").tail(1).values[0][0] * t.select(t.where(df, "date", date, op.eq), "pct_change").values[0][0])
            account_cash_start_day = t.select(t.where(df, "date", date, op.lt), "account_cash_end_day").tail(1).values[0][0]
            
        invested_end_day   = invested_start_day + t.select(t.where(df, "date", date, op.eq), "order_size").values[0][0]
        account_cash_end_day = account_cash_start_day - t.select(t.where(df, "date", date, op.eq), "order_size").values[0][0]
        net_worth = invested_end_day + account_cash_end_day
        
        # update table
        df.loc[t.where(df, "date", date, op.eq).index[0], "invested_start_day"] = invested_start_day
        df.loc[t.where(df, "date", date, op.eq).index[0], "invested_end_day"] = invested_end_day
        df.loc[t.where(df, "date", date, op.eq).index[0], "account_cash_start_day"] = account_cash_start_day
        df.loc[t.where(df, "date", date, op.eq).index[0], "account_cash_end_day"] = account_cash_end_day
        df.loc[t.where(df, "date", date, op.eq).index[0], "net_worth"] = net_worth
        # include warning in case we spend more than we have (borrow)
        # or we sell more than we have (shorting)
        if account_cash_end_day < 0:
            df.loc[t.where(df, "date", date, op.eq).index[0], "nb"] = "account_cash_end_day is negative"
        if invested_end_day < 0:
            df.loc[t.where(df, "date", date, op.eq).index[0], "nb"] = df.loc[t.where(df, "date", date, op.eq).index[0], "nb"] + " invested_end_day is negative"
        
    # lift
    net_worth_start = t.column(df, "net_worth")[0]
    net_worth_end = t.column(df, "net_worth")[-1]
    lift = (net_worth_end - net_worth_start) / net_worth_start

    return ({'transactions': df, 'lift': lift})
        




if __name__ == "__main__":
    #print("Run as a lib:")
    l = []
    for key, value in list(locals().items()):
        if callable(value) and value.__module__ == __name__:
            l.append(key)
    print(l)
    
    
    print("Example> backtest_strategy(...)")
    _symbol_price = """date,SPY
    2020-05-15,286.2799987792969
    2020-05-18,295.0
    2020-05-19,291.9700012207031
    2020-05-20,296.92999267578125
    2020-05-21,294.8800048828125
    """

    import io as io 
    symbol_price = pd.read_csv(io.StringIO(_symbol_price), sep=",")

    _trade_orders = """date,order_size
    2020-05-15,50
    2020-05-18,0
    2020-05-19,-50
    2020-05-20,150
    2020-05-21,0
    """
    trade_orders = pd.read_csv(io.StringIO(_trade_orders), sep=",")

    capital = 500
    result = backtest_strategy(symbol_price, trade_orders, capital)
    print(result["lift"])
    print(result["transactions"])
