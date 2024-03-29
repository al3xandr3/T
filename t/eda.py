# -*- coding: utf-8 -*-
"""
Created on Sat May 16 16:13:37 2020

@author: amatos
"""

import pandas    as pd
import numpy     as np
import matplotlib.pyplot as plt
import seaborn           as sns
import t.table as t
import t.stats as s

######################################
# EDA

## EDA: 1 Var    
def ten_buckets(df, column1, doPlot=True):

    df2 = t.sort(df, column1)

    # reset index
    df2 = df2.reset_index(drop=True)

    df2['decile'] = pd.qcut(df2.index, 10, labels=False)
    df2['decile'] = df2['decile']+1

    # maybe i could go all the way, calculate stuff, and plot it too
    def decile_agg(df):
        names = {
            'median':     np.median(df[column1])
            ,'avg':       np.mean(df[column1])
            ,'count':     len(df[column1])
            ,'min':       np.min(df[column1])
            ,'max':       np.max(df[column1])
        }
        return pd.Series(names, index=names.keys())

    _agg = df2.groupby('decile').apply(decile_agg).reset_index()

    if(doPlot):
        bar(_agg, "decile", "avg", figsize=(11,6))
        for i in range(0, 10):
            _ = plt.text(i - 0.4, t.column(_agg, "avg")[i] + 0.2
                        #,"avg: {0:.1f} \nmedian: {1:.1f} \ncount: {2:.1f}".format(  T(_agg).column("avg")[i], T(_agg).column("median")[i], T(_agg).column("count")[i]  )
                        ,"min: {0:.1f} \nmax: {1:.1f} \navg: {2:.1f} \nmed: {3:.1f}".format( t.column(_agg, "min")[i], t.column(_agg, "min")[i], t.column(_agg, "avg")[i], t.column(_agg, "median")[i], )
                        #," {0:.1f}".format(  T(_agg).column("avg")[i] )
                        #, color='blue'
                        , fontweight='bold')
        plt.ylabel("Average")
        plt.xlabel("Deciles" + " of (ascending) " + column1)
        #plt.title('Decile')

    return _agg


## APRIORI
# reference: https://engineering.fb.com/2021/02/09/developer-tools/minesweeper/
def diff_patterns(df, column, classificationA, classificationB, withExtra=False, _min_support=0.2, _min_threshold=0.01):
    
    #from mlxtend.preprocessing import TransactionEncoder
    from mlxtend.frequent_patterns import apriori
    from mlxtend.frequent_patterns import association_rules

    ## Input Interface
    Dataset = df
    Column  = column
    ClassificationA = classificationA
    ClassificationB = classificationB
    # Other filtering parameters

    cola = f"{Column}:{ClassificationA}"
    colb = f"{Column}:{ClassificationB}"
    
    # ---------------------------------------
    
    # dataset cleanup
    Dataset = Dataset.fillna("NA")
    
    df = pd.get_dummies(Dataset.astype(str), prefix_sep=":")
    
    # frequent item sets, in same basket (with a min support)
    frequent_itemsets = apriori(df, min_support=_min_support, use_colnames=True)
    frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
    frequent_itemsets
    
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=_min_threshold)
    
    cola_rules = t.where(rules, "consequents", {cola})

    # Count occurences
    for index, row in cola_rules.iterrows():
        dt_w = df
        dt_wout = df
        #print(list(row['antecedents']))
        for ant in list(row['antecedents']):
            dt_w = t.where(dt_w, ant, True)
            #dt_wout = t.where(dt_wout, ant, False)

        dt_wout = pd.merge(df, dt_w, how='outer', indicator=True)
        dt_wout = t.where(dt_wout, "_merge", "left_only")
        
        #print(dt_w)
        #print(dt_wout)

        tp    = t.where(dt_w,    cola, True)
        fn    = t.where(dt_wout, cola, True)
        fp    = t.where(dt_w,    colb, True)
        tn    = t.where(dt_wout, colb, True)

        cola_rules.at[index,'TP'] = tp.shape[0]
        cola_rules.at[index,'FN'] = fn.shape[0]
        cola_rules.at[index,'FP'] = fp.shape[0]
        cola_rules.at[index,'TN'] = tn.shape[0]
            
    output = cola_rules
    
    total     = len(df)
    
    if 'TP' not in output.columns:
        return "Empty"
    
    precision = output["TP"] / (output["TP"] + output["FP"])
    recall    = output["TP"] / (output["TP"] + output["FN"])
    accuracy  = (output["TP"] + output["TN"]) / total
    f1  = 2*((precision*recall)/(precision+recall))
    f05 = (1.25 * precision * recall) / (0.25 * precision + recall)
    f2  = (5 * precision * recall) / (4 * precision + recall)

    output["total"] = total

    output["precision"] = precision
    output["recall"]    = recall
    output["accuracy"]  = accuracy
    output["f1"] = f1
    output["f05"] = f05
    output["f2"] = f2
    
    output = t.relabel(output, "A_cnt",   f"{ClassificationA}_cnt")
    output = t.relabel(output, "B_cnt",   f"{ClassificationB}_cnt")
    output = t.relabel(output, "A_ratio", f"{ClassificationA}_ratio")
     
    output_clean = output.drop(['consequents', 'antecedent support', 'consequent support', 'support', 'confidence', 'lift', 'leverage', 'conviction', 'total'], axis=1)
    output_clean = t.relabel(output_clean,"antecedents", "pattern")
    
    output_clean = t.sort(output_clean, "f1", ascending=False)
    output_clean["rules_count"] = output_clean["pattern"].apply(lambda x: len(x))
    #rules = t.sort(rules, "f1", ascending=False)
    
    if withExtra:
        return output_clean, cola_rules   
    else: 
        return output_clean




## EDA: Viz

def ecdf(df, col1, col2='', xlabel='', ylabel=''):

    if (col2==''):
        return s.plt_1ecdf( t.column(df, col1), xlabel, ylabel)
    else: 
        return s.plt_2ecdf( t.column(df, col1), t.column(df, col2), _xlabel=label )


# needs to accepts bins
# plot percentages
# copy layout and approach from data8, nicer
def histogram(df, col1, xlabel='', **vargs):
    
    plot = sns.distplot( df[col1], kde=False, rug=True, **vargs)
    plt.ylabel("Count")
    if xlabel != '':
        plt.xlabel(xlabel)

    return plot


# https://chrisalbon.com/python/data_visualization/matplotlib_scatterplot_from_pandas/
def scatter(df, x, y1, **vargs):
    return df.plot.scatter(x=x, y=y1, **vargs)


def bar(df, x, y1, y2='', **vargs):
    if (y2==''):
        return df.plot.bar(x=x, y=y1, **vargs)        
    else:
        return df.plot.bar(x=x, y=[y1, y2], **vargs)


def line(df, x, y1, y2='', **vargs):
    if (y2==''):
        return df.plot(x=x, y=y1, **vargs)        
    else:
        return df.plot(x=x, y=[y1, y2], **vargs)


def barh(df, x, y1, y2='', **vargs):
    if (y2==''):
        return df.plot.barh(x=x, y=y1, **vargs)        
    else:
        return df.plot.barh(x=x, y=[y1, y2], **vargs)

    
    

if __name__ == "__main__":
    print("Run as a lib:")
    l = []
    for key, value in list(locals().items()):
        if callable(value) and value.__module__ == __name__:
            l.append(key)
    print(l)

    
