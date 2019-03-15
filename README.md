
# T is for Table

T makes Pandas more intuitive while extending it further with DataScience tools. In particular, Bootstrap ReSampling methods (a.k.a [Hacker Statistics](https://speakerdeck.com/jakevdp/statistics-for-hackers))

Under the covers it uses Pandas dataframe for speed and for its rich library.

## Basic Usage

```python
>>> df = pd.DataFrame( {
    'user':['k','j','k','t','k','j']
    ,'period':['pre', 'pre', 'pre', 'pre', 'post','post'] 
    , 'kpi':[13,12,2,12,43,34]
    })
```

**.** |**user**|**period**|**kpi**
:-----:|:-----:|:-----:|:-----:
0|k|pre|13
1|j|pre|12
2|k|pre|2
3|t|pre|12
4|k|post|43
5|j|post|34

Wrap a pandas dataframe with a T( ) to have access to this library methods 

```python
>>> T(df).where("period", "post").select("user", "kpi")
```
**.** |**user**|**kpi**
:-----:|:-----:|:-----:
0|k|43
1|j|34


T( ) takes the same arguments as a data frame and then includes new methods, like this example that uses Bootstrap ReSampling approach to calculate a 95% confidence interval of the mean:

```python
>>> T(np.random.normal(size=(37,2)), columns=['A', 'B']).ci_mean('A')
```

    {'mean': -0.33, '95% conf int of mean': array([-0.64, -0.03])}

including plotting it:

![ci_mean](docs/ci_mean.png)

