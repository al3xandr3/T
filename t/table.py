# -*- coding: utf-8 -*-

import pandas    as pd
import numpy     as np
import operator  as op
import seaborn   as sns


# http://data8.org/datascience/_modules/datascience/tables.html


#####################
# Frame Manipulation

def relabel(df, OriginalName, NewName):
    return df.rename(index=str, columns={OriginalName: NewName})


# https://docs.python.org/3.4/library/operator.html
def where(df, column, value, operation=op.eq):
    return pd.DataFrame( df.loc[operation(df.loc[:,column], value) ,:] )


def select(df, *column_or_columns):
    table = pd.DataFrame()

    for column in column_or_columns:
        table[column] = df.loc[:, column].values

    return table


def column(df, index_or_label):
    """Return the values of a column as an array.

    Args:
        label (int or str): The index or label of a column

    Returns:
        An instance of ``numpy.array``.

    Raises:
        ``ValueError``: When the ``index_or_label`` is not in the table.
    """
    if (isinstance(index_or_label, str)):
        if (index_or_label not in df.columns):
            raise ValueError(
                'The column "{}" is not in the table. The table contains '
                'these columns: {}'
                .format(index_or_label, ', '.join(df.labels))
            )
        else:
            return df.loc[:, index_or_label].values

    if (isinstance(index_or_label, int)):
        if (not 0 <= index_or_label < len(df.columns)):
            raise ValueError(
                'The index {} is not in the table. Only indices between '
                '0 and {} are valid'
                .format(index_or_label, len(df.labels) - 1)
            )
        else:
            return df.iloc[:,index_or_label].values

def drop(df, index_or_label):
    
    if (isinstance(index_or_label, str)):
        if (index_or_label not in df.columns):
            raise ValueError(
                'The column "{}" is not in the table. The table contains '
                'these columns: {}'
                .format(index_or_label, ', '.join(df.labels))
            )
        else:
            return df.drop(index_or_label, axis=1)

    if (isinstance(index_or_label, int)):
        if (not 0 <= index_or_label < len(df.columns)):
            raise ValueError(
                'The index {} is not in the table. Only indices between '
                '0 and {} are valid'
                .format(index_or_label, len(df.labels) - 1)
            )
        else:
            return df.drop(index_or_label, axis=0)
    return 


def row(df, index):
    """Return the values of a row as an array.

    Args:
        label (int): The index or label of a column

    Returns:
        An instance of ``numpy.array``.

    Raises:
        ``ValueError``: When the ``index_or_label`` is not in the table.
    """
    return df.iloc[index,:].values



def cell(df, row, column):
    return df.iloc[column, row]


def exclude(df, toexclude_df, column):
    the_join = pd.merge(df, toexclude_df, on=[column], how="outer", indicator=True)
    return ( pd.DataFrame(the_join).where('_merge', "left_only") )
    
 
def format(df, num_format=lambda x: '{:,.1f}'.format(x)):
    """Returns a better number formated table. Is Slow

    Args:
        label (int or str): The index or label of a column

    Returns:
        pandas dataframe
    """
    #TODO: this looks inefficient 
    def build_formatters_ints(df):
        return {
            column:lambda x: '{:,}'.format(x) 
            for column, dtype in df.dtypes.items()
            if dtype in [ np.dtype('int64') ] 
        }

    def build_formatters_floats(df):
        return {
            column:lambda x: '{:.1f}'.format(x) 
            for column, dtype in df.dtypes.items()
            if dtype in [  np.dtype('float64') ] 
        }
    
    format_int   = build_formatters_ints(df)
    format_float = build_formatters_floats(df)
    style = '<style>.dataframe td { text-align: right; }</style>'

    return df.style.set_table_styles(style).format(format_int).format(format_float)


def group(df, column, rename=""):
    df_gp = pd.DataFrame(df[column].value_counts())
    if rename != "":
        return relabel(df_gp,column,rename)
    else:
        return relabel(df_gp,column,column + "_count")

def count(df, column):
    return len( np.unique( df[column] ))

def showna(df):
    return sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')

def sort(df, col, ascending=True):
    return pd.DataFrame(df.sort_values(col, ascending=ascending))

##
def variance(df, column1):
    return np.var( pd.DataFrame(df)[column1] )


def median(df, column1):
    return np.median( pd.DataFrame(df)[column1] )

def avg(df, column1):
    return np.mean( pd.DataFrame(df)[column1] )

def average(df, column1):
    return np.mean( pd.DataFrame(df)[column1] ) 

def std(df, column1):
    return np.std( pd.DataFrame(df)[column1] )

def sum(df, column1):
    return np.sum( pd.DataFrame(df)[column1] )



if __name__ == "__main__":
    print("Run as a lib:")
    l = []
    for key, value in list(locals().items()):
        if callable(value) and value.__module__ == __name__:
            l.append(key)
    print(l)
