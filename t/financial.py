# -*- coding: utf-8 -*-
"""
Created on Sat May 16 15:53:36 2020

@author: amatos
"""

import pandas as pd
import t.table as t
import operator as op
from datetime import datetime
#from datetime import date
#import numpy as np
from scipy import optimize


#datetime_object = datetime.strptime('Jun 1 2005  1:33PM', '%b %d %Y %I:%M%p')

##################################
# Financial 
# https://www.youtube.com/watch?v=XLL9KGeQltQ&feature=youtu.be
# https://github.com/marekkolman/yt_backtest_trading_demo/blob/master/backtest.py

def get_quotes(symbol, date_from, date_to=datetime.today()):
    import pandas as pd
    import pandas_datareader.data as web
    raw_data = web.DataReader(symbol, 'yahoo', datetime.strptime(date_from, "%Y-%m-%d"), date_to)
    data = raw_data.stack(dropna=False).reset_index().rename(columns = {'Symbols':'symbol', 'Date':'date'}).sort_values(by = ['symbol', 'date'])
    return data
# > prices = get_quotes(['SPY', '^GSPC', '^VIX'], date_from = '2000-01-01')



def get_quotes_close(symbol, date_from, date_to=datetime.today() ):
    import pandas as pd
    import pandas_datareader.data as web
    raw_data = web.DataReader(symbol, 'yahoo', datetime.strptime(date_from, "%Y-%m-%d"), date_to)
    data = raw_data.stack(dropna=False)['Adj Close'].to_frame().reset_index().rename(columns = {'Symbols':'symbol', 'Date':'date', 'Adj Close':'value'}).sort_values(by = ['symbol', 'date'])
    return pd.pivot_table(data, columns = 'symbol', index = 'date', values ='value')    
# > prices = get_quotes_close(['SPY', '^GSPC', '^VIX'], date_from = '2000-01-01')


def xnpv(rate,cashflows):
    """
    Calculate the net present value of a series of cashflows at irregular intervals.
    Arguments
    ---------
    * rate: the discount rate to be applied to the cash flows
    * cashflows: a list object in which each element is a tuple of the form (date, amount), where date is a python datetime.date object and amount is an integer or floating point number. Cash outflows (investments) are represented with negative amounts, and cash inflows (returns) are positive amounts.
    
    Returns
    -------
    * returns a single value which is the NPV of the given cash flows.
    Notes
    ---------------
    * The Net Present Value is the sum of each of cash flows discounted back to the date of the first cash flow. The discounted value of a given cash flow is A/(1+r)**(t-t0), where A is the amount, r is the discout rate, and (t-t0) is the time in years from the date of the first cash flow in the series (t0) to the date of the cash flow being added to the sum (t).  
    * This function is equivalent to the Microsoft Excel function of the same name. 
    """

    chron_order = sorted(cashflows, key = lambda x: x[0])
    t0 = chron_order[0][0] #t0 is the date of the first cash flow

    return sum([cf/(1+rate)**((t-t0).days/365.0) for (t,cf) in chron_order])

def xirr(cashflows,guess=0.1):
    """
    Calculate the Internal Rate of Return of a series of cashflows at irregular intervals.
    Arguments
    ---------
    * cashflows: a list object in which each element is a tuple of the form (date, amount), where date is a python datetime.date object and amount is an integer or floating point number. Cash outflows (investments) are represented with negative amounts, and cash inflows (returns) are positive amounts.
    * guess (optional, default = 0.1): a guess at the solution to be used as a starting point for the numerical solution. 
    Returns
    --------
    * Returns the IRR as a single value
    
    Notes
    ----------------
    * The Internal Rate of Return (IRR) is the discount rate at which the Net Present Value (NPV) of a series of cash flows is equal to zero. The NPV of the series of cash flows is determined using the xnpv function in this module. The discount rate at which NPV equals zero is found using the secant method of numerical solution. 
    * This function is equivalent to the Microsoft Excel function of the same name.
    * For users that do not have the scipy module installed, there is an alternate version (commented out) that uses the secant_method function defined in the module rather than the scipy.optimize module's numerical solver. Both use the same method of calculation so there should be no difference in performance, but the secant_method function does not fail gracefully in cases where there is no solution, so the scipy.optimize.newton version is preferred.
    
     _irr = xirr( [ (date(2010, 12, 29), -10000),
                   (date(2012, 1, 25), 20),
                   (date(2012, 3, 8), 10100)] )
    """
    
    val = -666
    
    try:
        val = optimize.newton(lambda r: xnpv(r,cashflows),guess)
    except:
        print("Failed to converge after, returning: -666")
    
    return val



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
    df = t.drop(df, "_merge")
    
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

    df["flux"] = - df["order_size"]
    df.loc[df.index[-1], "flux"] = t.column(df, "flux")[-1] + net_worth_end
    _irr = xirr(list(zip(df.date, df.flux)))
    
    return ({'transactions': df, 'ROI': lift, 'IRR': _irr, 'return': net_worth_end})
        




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
