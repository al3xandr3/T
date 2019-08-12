# -*- coding: utf-8 -*-

import pandas    as pd
import numpy     as np
import operator  as op
import T._stats   as stats
import matplotlib.pyplot as plt
import seaborn   as sns


# http://data8.org/datascience/_modules/datascience/tables.html#Table.to_df

class T(pd.DataFrame):

    #####################
    # Frame Manipulation

    def relabel(self, OriginalName, NewName):
        return self.rename(index=str, columns={OriginalName: NewName})

    # https://docs.python.org/3.4/library/operator.html
    def where(self, column, value, operation=op.eq):
        return T( self.loc[operation(self.loc[:,column], value) ,:]  )


    def select(self, *column_or_columns):
        table = T()

        for column in column_or_columns:
            table[column] = self.loc[:, column].values

        return table


    def column(self, index_or_label):
        """Return the values of a column as an array.

        Args:
            label (int or str): The index or label of a column

        Returns:
            An instance of ``numpy.array``.

        Raises:
            ``ValueError``: When the ``index_or_label`` is not in the table.
        """
        if (isinstance(index_or_label, str)):
            if (index_or_label not in self.columns):
                raise ValueError(
                    'The column "{}" is not in the table. The table contains '
                    'these columns: {}'
                    .format(index_or_label, ', '.join(self.labels))
                )
            else:
                return self.loc[:, index_or_label].values

        if (isinstance(index_or_label, int)):
            if (not 0 <= index_or_label < len(self.columns)):
                raise ValueError(
                    'The index {} is not in the table. Only indices between '
                    '0 and {} are valid'
                    .format(index_or_label, len(self.labels) - 1)
                )
            else:
                return self.iloc[:,index_or_label].values
    

    def row(self, index):
        """Return the values of a row as an array.

        Args:
            label (int): The index or label of a column

        Returns:
            An instance of ``numpy.array``.

        Raises:
            ``ValueError``: When the ``index_or_label`` is not in the table.
        """
        return self.iloc[index,:].values


    def exclude(self, toexclude_df, column):
        the_join = pd.merge(self, toexclude_df, on=[column], how="outer", indicator=True)
        return ( T(df).where('_merge', "left_only") )
        
 
    def format(self, num_format=lambda x: '{:,}'.format(x)):
        """Returns a better number formated table. Is Slow

        Args:
            label (int or str): The index or label of a column

        Returns:
            pandas dataframe
        """
        def build_formatters(df, format):
            return {
                column:format 
                for column, dtype in df.dtypes.items()
                if dtype in [ np.dtype('int64'), np.dtype('float64') ] 
            }
        formatters = build_formatters(self, num_format)
        style = '<style>.dataframe td { text-align: right; }</style>'

        return self.style.set_table_styles(style).format(formatters)


    def group(self, column):
        return T(self[column].value_counts())

    def count(self, column):
        return len( np.unique( self[column] ))

    def showna(self):
        return sns.heatmap(self.isnull(),yticklabels=False,cbar=False,cmap='viridis')

    def sort(self, col, ascending=True):
        return T(self.sort_values(col, ascending=ascending))


    def variance(self, column1):
        return np.var( T(self).column(column1) )

    def median(self, column1):
        return np.median( T(self).column(column1) )
    
    def avg(self, column1):
        return np.mean( T(self).column(column1) )

    def std(self, column1):
        return np.std( T(self).column(column1) ) 


    ########################################
    # Confidence Interval
    def ci_mean(self, column1, withChart=True):
        res = stats.bs_mean_95ci( T(self).column(column1) )

        # plot
        if (withChart):
            resampled_proportions = T(res["Bootstrap Samples"], columns=['Bootstrap Samples'])
            resampled_proportions.histogram('Bootstrap Samples')
            _ = plt.plot([res["95% conf int of mean"][0], res["95% conf int of mean"][1]], [0, 0], color='yellow', lw=8)
            _ = plt.plot([res["mean"]], [0.05], marker='o', markersize=3, color="red")

        return ({  
         "mean": res["mean"]
        ,"95% conf int of mean": res["95% conf int of mean"]
    })


    def ci_median(self, column1, withChart=True):
        res = stats.bs_median_95ci( T(self).column(column1) )

        # plot
        if (withChart):
            resampled_proportions = T(res["Bootstrap Samples"], columns=['Bootstrap Samples'])
            resampled_proportions.histogram('Bootstrap Samples')
            _ = plt.plot([res["95% conf int of median"][0], res["95% conf int of median"][1]], [0, 0], color='yellow', lw=8)
            _ = plt.plot([res["median"]], [0.05], marker='o', markersize=3, color="red")

        return ({  
         "median": res["median"]
        ,"95% conf int of median": res["95% conf int of median"]
        })
        

    def ci_proportion(self, column1, repetitions=5000, withChart=True):
    
        just_one_column = T(self).select(column1)
        proportions = []
        for i in np.arange(repetitions):
            bootstrap_sample = just_one_column.sample(n=len(T(self).column(column1)), replace=True) # sample with replacement
            resample_array = T(bootstrap_sample).column(0)
            resampled_proportion = np.count_nonzero(resample_array) / len(resample_array)
            proportions = np.append(proportions, resampled_proportion)
            
        # Get the endpoints of the 95% confidence interval
        left  = np.percentile(proportions, 2.5)
        right = np.percentile(proportions, 97.5)

        ## plot
        if (withChart):
            resampled_proportions = T(proportions, columns=['Bootstrap Sample Proportion'])
            resampled_proportions.histogram('Bootstrap Sample Proportion')
            _ = plt.plot([left, right], [0, 0], color='yellow', lw=8)
            _ = plt.plot([np.count_nonzero(T(self).column(column1) ) / len(T(self).column(column1))], [0.05], marker='o', markersize=3, color="red")

        return {
            "Proportion of 1s": np.count_nonzero(T(self).column(column1) ) / len(T(self).column(column1))
            ,"95% Conf. Int. of Proportion":  [left, right]
        }

















    ######################################
    # EDA

    ## EDA: 1 Var    
    def decile(self, column1, doPlot=True):

        df2 = T(self).sort(column1)

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
            T(_agg).bar("decile", "avg", figsize=(11,6))
            for i in range(0, 10):
                _ = plt.text(i - 0.4, T(_agg).column("avg")[i] + 0.2
                            #,"avg: {0:.1f} \nmedian: {1:.1f} \ncount: {2:.1f}".format(  T(_agg).column("avg")[i], T(_agg).column("median")[i], T(_agg).column("count")[i]  )
                            ,"min: {0:.1f} \nmax: {1:.1f} \navg: {2:.1f} \nmed: {3:.1f}".format( T(_agg).column("min")[i], T(_agg).column("max")[i], T(_agg).column("avg")[i], T(_agg).column("median")[i], )
                            #," {0:.1f}".format(  T(_agg).column("avg")[i] )
                            , color='blue', fontweight='bold')

        return _agg








    ## EDA: Viz

    def ecdf(self, col1, col2='', label=''):

        if (col2==''):
            return stats.plt_1ecdf( T(self).column(col1), label )
        else: 
            return stats.plt_2ecdf( T(self).column(col1), T(self).column(col2), _xlabel=label )


    # needs to accepts bins
    # plot percentages
    # copy layout and approach from data8, nicer
    def histogram(self, col1, xlabel='', **vargs):
        
        plot = sns.distplot( self[col1], kde=False, rug=True, **vargs)
        plt.ylabel("Count")
        if xlabel != '':
            plt.xlabel(xlabel)

        return plot

        #import datascience as ds

        #if (col2==''):
        #    ds.Table.from_df(self).select(col1).hist(unit=unit, **vargs)
            #self.hist(column=col1, alpha=0.9, **vargs)

        #elif (not side_by_side):
        #else:
        #    ds.Table.from_df(self).select(col1, col2).hist(unit=unit, **vargs)
            #self.hist(column=col1, alpha=0.5, **vargs)
            #self.hist(column=col2, alpha=0.5, **vargs)

        #else: # side_by_side
        #    fig, ax = plt.subplots()

        #    a_heights, a_bins = np.histogram(self[col1])
        #    b_heights, b_bins = np.histogram(self[col2], bins=a_bins)

        #    width = (a_bins[1] - a_bins[0])/3

        #    ax.bar(a_bins[:-1], a_heights, width=width, facecolor='cornflowerblue')
        #    ax.bar(b_bins[:-1]+width, b_heights, width=width, facecolor='seagreen')



    # https://chrisalbon.com/python/data_visualization/matplotlib_scatterplot_from_pandas/
    def scatter(self, x, y1, **vargs):
        return self.plot.scatter(x=x, y=y1, **vargs)


    def bar(self, x, y1, y2='', **vargs):
        if (y2==''):
            return self.plot.bar(x=x, y=y1, **vargs)        
        else:
            return self.plot.bar(x=x, y=[y1, y2], **vargs)


    def line(self, x, y1, y2='', **vargs):
        if (y2==''):
            return self.plot(x=x, y=y1, **vargs)        
        else:
            return self.plot(x=x, y=[y1, y2], **vargs)


    def barh(self, x, y1, y2='', **vargs):
        if (y2==''):
            return self.plot.barh(x=x, y=y1, **vargs)        
        else:
            return self.plot.barh(x=x, y=[y1, y2], **vargs)










    ##################################
    # Hypothesis Testing 

    # https://www.inferentialthinking.com/chapters/12/1/AB_Testing
    def hypothesis_mean_diff(self, label, group_label, repetitions=10000):
        
        tbl = T(self).select(group_label, label)
        
        differences = []
        for i in np.arange(repetitions):
            #shuffled = T(tbl.sample(n=len(T(self).column(0)), replace=False)).column(1)
            shuffled = T(tbl.sample(n=len(self.index), replace=False)).column(1)
            tbl['Shuffled'] = shuffled

            #shuffled_means       = tbl.groupby(group_label).apply(lambda df: np.mean(df['Shuffled'])).reset_index()
            shuffled_means       = tbl.groupby(group_label).mean() #.reset_index()
            simulated_difference = T(shuffled_means).column(1)[1] - T(shuffled_means).column(1)[0]
            differences          = np.append(differences, simulated_difference)
        
        ## Chart
        _ = T(differences, columns=['Difference Between Group Averages']).hist('Difference Between Group Averages')
        #odf = tbl.groupby(group_label).apply(lambda df: np.mean(df[label])).reset_index()
        odf = tbl.groupby(group_label).mean().reset_index()
        odiff = T(odf).column(1)[1] - T(odf).column(1)[0]

        plt.plot( odiff, 0,  'ro', color='red')
        #plots.scatter(observed_difference, 0, color='red', s=30)
        plt.title('Prediction Under the Null Hypothesis')
        # print('Observed Difference:', observed_difference)
        plt.ylim(bottom=-1)

        # Compute p-value: p
        # % time that permutations higher than observed
        #p = np.sum(differences <= odiff) / len(differences)
        p = np.mean(differences >= odiff)
        p = 2*min(p, 1-p) # both sides

        return {
            'Observed Difference': odiff
            ,'p-value': p
            ,'decision': "No significant difference" if p>0.05 else "Significant Difference"
            ,'Null Hypothesis Bootstrap Differences': differences   
    }


    #differences = permuted_sample_average_difference(baby, 'Maternal Age', 'Maternal Smoker', 5000)





    ##################################
    # Model Scoring

    def confusion(self, actuals, predicted):
        from sklearn import metrics
        from sklearn.metrics import confusion_matrix
        confm = confusion_matrix(actuals, predicted)

        # plot
        class_names=[0,1] # name  of classes
        fig, ax = plt.subplots()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names)
        plt.yticks(tick_marks, class_names)
        # create heatmap
        #plt.rcParams['font.size'] = 16
        sns.set(font_scale=1.5)
        #plt.rcParams['figure.figsize'] = 5,3
        #plt.figure(figsize=(16, 6))
        sns.heatmap(pd.DataFrame(confm), annot=True, cmap="YlGnBu" ,fmt=',')
        ax.xaxis.set_label_position("top")
        plt.tight_layout()
        plt.title('Confusion matrix', y=1.1)
        plt.ylabel('Actual label')
        plt.xlabel('Predicted label')

        return confm


    def roc(self, actuals, predicted):
        from sklearn.metrics import roc_curve, auc

        # plot
        
        false_positive_rate, true_positive_rate, thresholds = roc_curve(actuals, predicted)
        auc = auc(false_positive_rate, true_positive_rate)
        plt.plot(false_positive_rate,true_positive_rate,label="data 1, auc="+str(auc))
        plt.legend(loc=4)
        #plt.rcParams['font.size'] = 17
        #plt.rcParams['figure.figsize'] = 12,7
        plt.title('ROC curve')
        plt.xlabel('False Positive Rate (1 - Specificity)')
        plt.ylabel('True Positive Rate (Sensitivity)')
        plt.grid(True)
 
        return [false_positive_rate, true_positive_rate, thresholds]


    def scores(self, actuals, predicted):
        from sklearn.metrics import classification_report
        return print(classification_report(actuals, predicted, digits=3))


    def score(self, actuals, predicted):
        from sklearn.metrics import precision_recall_fscore_support as score
        from sklearn.metrics import accuracy_score
        precision,recall,fscore,support = score(actuals, predicted,average='macro')

        print ('Precision : {}'.format(precision) )
        print ('Recall    : {}'.format(recall) )
        print ('F-score   : {}'.format(fscore) )
        print ('Accuracy  : {}'.format(accuracy_score(actuals,predicted)) )
        print ('Support   : {}'.format(support) )

        return [precision,recall,fscore,support]


    def prediction_eval(self, actuals, predicted):
        T().score(actuals, predicted)

        T().confusion(actuals, predicted)
        plt.show()

        T().roc(actuals, predicted)
        plt.show()

        return 1


    def tree_explainer(self, model, features):

        import shap

        # load JS visualization code to notebook
        shap.initjs()
        if len(features.index) > 70000:
            data = T(features).sample(n=70000)
        else:
            data = features

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(data)
        print('Expected Value:', explainer.expected_value)
        pd.DataFrame(shap_values).head()

        return shap.summary_plot(shap_values, data)

        
    def tree_viz(self, model, feature_names, class_names=[]):

        #https://chrisalbon.com/machine_learning/trees_and_forests/visualize_a_decision_tree/
        
        import pydotplus
        from sklearn import tree
        from IPython.display import Image

        # Create DOT data
        dot_data = tree.export_graphviz(model 
                                        ,out_file=None 
                                        ,filled=True
                                        ,rounded=True  
                                        ,special_characters=True
                                        ,proportion=True
                                        ,feature_names=feature_names
                                        ,class_names=class_names
                                       )

        # Draw graph
        graph = pydotplus.graph_from_dot_data(dot_data)  

        # Show graph
        return Image(graph.create_png())





if __name__ == "__main__":
    print("run as a lib")
