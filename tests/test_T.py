

import pytest
from datetime import datetime
from numpy.testing import assert_array_equal
import pandas    as pd
import operator  as op

import t as t

import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

#########
# Utils #
#########

@pytest.fixture(scope='function')
def table1():
    """Setup Scrabble table"""
    return pd.DataFrame({'user':['k','j','k','t','k','j'] \
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
    assert t.column(t.select(table1, "kpi"), 0)[0] == 5


def test_where(table1):
    """Tests where and column by index"""
    assert t.column(t.where(table1, "kpi", 10), 0)[0] == 'j'


def test_where2(table1):
    """Tests where and column by index"""
    assert t.column(t.where(table1, "kpi", 11, op.ne), 0)[0] == 'k'


def test_where3(table1):
    """Tests where and column by index"""
    assert t.column(t.where(table1, "kpi", 9, op.gt), 0)[0] == 'j'


def test_relabel(table1):
    """Tests relabel column by name, multiple select"""
    t2 = t.relabel(table1,"kpi", "bacalhau")
    assert t.column(t.select(t2, "bacalhau", "isLabel"), "bacalhau")[0] == 5
    assert t.column(t.select(t2, "bacalhau", "isLabel"), "isLabel")[0]  == 0


def test_row(table1):
    """Tests row and column by name"""
    assert t.row(table1, 0)[0] == 'k'


def test_group(table1):
    """Tests row and column by name"""
    assert t.where(t.group(table1, "user"), "user", 1).reset_index()['index'].values[0] == 't'


def test_sort(table1):
    """Tests row and column by name"""
    assert t.column(t.select(t.sort(table1, "kpi"), "kpi2"), 0)[0] == 51234123
    assert t.column( t.select( t.sort(table1, "kpi", ascending=False), "kpi2"), 0)[0] == 101234123


def test_numeric_eda(table1):
    assert round(t.variance(table1, "kpi")) == 3
    assert round(t.std(table1, "kpi"))      == 2
    assert round(t.avg(table1, "kpi"))      == 8 
    assert round(t.median(table1, "kpi"))   == 8

def test_ci(table1):
    assert t.ci_mean(table1, "kpi")['mean'] == 7.5
    assert t.ci_median(table1, "kpi")['median'] == 7.5

