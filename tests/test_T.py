
import doctest
import re
import pytest
import numpy as np
from datetime import datetime
from numpy.testing import assert_array_equal

from T import *

import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

#########
# Utils #
#########

@pytest.fixture(scope='function')
def table1():
    """Setup Scrabble table"""
    return T({'user':['k','j','k','t','k','j'] \
                    , 'date': [datetime.strptime(d, '%Y-%m-%d') for d in ['2018-10-16', '2018-10-16', '2018-10-17', '2018-10-17', '2018-10-18', '2018-10-19']]
                    , 'period':['before', 'before', 'before', 'before', 'after','after'] \
                    , 'cohort':['control','control','control','control','control','control'] \
                    , 'kpi':[5,6,7,8,9,10]
                    , 'isLabel':[0,1,1,1,0,1]
                    , 'kpi2':[51234123, 641234123, 74123123, 812412314, 91234123, 101234123]
            })


###############
# Table query #
###############

def test_select(table1):
    """Tests select and column by index"""
    assert T(table1).select("kpi").column(0)[0] == 5


def test_where(table1):
    """Tests where and column by index"""
    assert T(table1).where("kpi", 10).column(0)[0] == 'j'


def test_where2(table1):
    """Tests where and column by index"""
    assert T(table1).where("kpi", 11, op.ne).column(0)[0] == 'k'


def test_where3(table1):
    """Tests where and column by index"""
    assert T(table1).where("kpi", 9, op.gt).column(0)[0] == 'j'


def test_relabel(table1):
    """Tests relabel column by name, multiple select"""
    t2 = T(table1).relabel("kpi", "bacalhau")
    assert T(t2).select("bacalhau", "isLabel").column("bacalhau")[0] == 5
    assert T(t2).select("bacalhau", "isLabel").column("isLabel")[0]  == 0

def test_row(table1):
    """Tests row and column by name"""
    assert T(table1).row(0)[0] == 'k'


def test_group(table1):
    """Tests row and column by name"""
    assert T(table1).group("user").where("user", 1).reset_index()['index'].values[0] == 't'


def test_sort(table1):
    """Tests row and column by name"""
    assert T(table1).sort("kpi").select("kpi2").column(0)[0] == 51234123
    assert T(table1).sort("kpi", ascending=False).select("kpi2").column(0)[0] == 101234123


def test_numeric_eda(table1):
    assert round(T(table1).variance("kpi")) == 3
    assert round(T(table1).std("kpi"))      == 2
    assert round(T(table1).avg("kpi"))      == 8 
    assert round(T(table1).median("kpi"))   == 8

def test_ci(table1):
    assert T(table1).ci_mean("kpi")['mean'] == 7.5
    assert T(table1).ci_median("kpi")['median'] == 7.5


"""
    print(

        # Query
        ,"\n\n", df.format()
        ,"\n\n", df.format(lambda x: '{0:.1%}'.format(x))

        # EDA Viz
        ,"\n\n", T(np.random.normal(size=(37,2)), columns=['A', 'B']).histogram('A'),         plt.show()
        ,"\n\n", T(np.random.normal(size=(37,2)), columns=['A', 'B']).histogram('A', 'B'),    plt.show()
        ,"\n\n", T(np.random.normal(size=(37,2)), columns=['A', 'B']).histogram('A', 'B', side_by_side=True), plt.show()
        ,"\n\n", T(np.random.normal(size=(37,2)), columns=['A', 'B']).scatter('A', 'B'),  plt.show()
        ,"\n\n", T(np.random.normal(size=(37,2)), columns=['A', 'B']).line('A', 'B'),     plt.show()
        ,"\n\n", T(np.random.normal(size=(37,2)), columns=['A', 'B']).bar('A', 'B'),      plt.show()
        ,"\n\n", T(np.random.normal(size=(37,2)), columns=['A', 'B']).barh('A', 'B'),     plt.show()
        ,"\n\n", T(np.random.normal(size=(37,2)), columns=['A', 'B'] ).ecdf('A', label='actions per day')
        ,"\n\n", T(np.random.normal(size=(37,2)), columns=['A', 'B'] ).ecdf('A', 'B', label='actions per day')
        ,"\n\n", T(np.random.normal(size=(37,2)), columns=['A', 'B'] ).showna()


        # EDA Number
        ,"\n\n", T(pd.DataFrame( np.arange(100), columns=['kpi']) ).decile('kpi')

        # Stat Inference, CI
        ,"\n\n", print(T(np.hstack((np.ones(200), np.zeros(800) )), columns=['isLabel'] ).ci_proportion('isLabel')),  plt.show()
        
        # Hypothesis
        ,"\n\n", df.hypothesis_mean_diff("kpi", "isLabel", repetitions=3)

        )
"""