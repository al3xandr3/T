# -*- coding: utf-8 -*-
"""
Created on Sat May 16 16:18:40 2020

@author: amatos
"""

import pandas    as pd
import numpy     as np

import matplotlib.pyplot as plt


##################################
# Model Scoring

def confusion(actuals, predicted):
    from sklearn.metrics import confusion_matrix
    confm = confusion_matrix(actuals, predicted)
    tn, fp, fn, tp = confm.ravel()

    cmtx = pd.DataFrame(
        confm, 
        index   = ['Actual: 0', 'Actual: 1'], 
        columns = ['Pred: 0', 'Pred:   1'] )

    #Total sum per column: 
    cmtx.loc['total',:]= cmtx.sum(axis=0)

    #Total sum per row: 
    cmtx.loc[:,'total'] = cmtx.sum(axis=1)

    cmtx.iloc[0,0] = "TN= " + str(cmtx.iloc[0,0])
    cmtx.iloc[1,1] = "TP= " + str(cmtx.iloc[1,1])
    cmtx.iloc[0,1] = "FN= " + str(cmtx.iloc[0,1])
    cmtx.iloc[1,0] = "FP= " + str(cmtx.iloc[1,0])
        

    prec = tp/(tp+fp)
    rec  = tp/(tp+fn)
    p =  'Precision (Pre) =TP/(TP+FP)                      : {}'.format(prec)
    r =  'Recall    (Rec) =TP/(TP+FN)                      : {}'.format(rec)
    f1=  'F1-score        =2*(Pre*Rec)/(Pre+Rec)           : {}'.format(2*(prec*rec)/(prec+rec))
    ac=  'Accuracy        =(TP+TN)/Total                   : {}'.format((tp+tn)/(tp+tn+fp+fn))
    f05= 'F05-score       =1.25*(Pre*Rec)/((0.25*Pre)+Rec) : {}'.format(1.25*prec*rec/((0.25*prec)+rec))
    f2 = 'F2-score        =5*(Pre*Rec)/((4*Pre)+Rec)       : {}'.format(5*prec*rec/((4*prec)+rec))

    return(cmtx, '', p, r, f1, ac, f05, f2)


def roc(actuals, predicted):
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


def classification_score(actuals, predicted):

    confusion(actuals, predicted)

    roc(actuals, predicted)
    plt.show()

    return 1


def regression_score(actuals, predicted):
    from sklearn import metrics
    print('MAE:' , metrics.mean_absolute_error(actuals, predicted))
    print('MSE:' , metrics.mean_squared_error(actuals, predicted))
    print('RMSE:', np.sqrt(metrics.mean_squared_error(actuals, predicted)))
    print('r2:'  , metrics.r2_score (actuals, predicted))


def cross_validation_score(model, actuals, predicted):
    from sklearn.model_selection import cross_val_score   
    _scores = cross_val_score(model, actuals, predicted, scoring="neg_mean_squared_error", cv=10)
    scores = np.sqrt(-_scores)
    print("Scores:", scores)
    print("")
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())



def tree_explainer( model, features):

    import shap

    # load JS visualization code to notebook
    shap.initjs()
    if len(features.index) > 70000:
        data = features.sample(n=70000)
    else:
        data = features

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(data)
    print('Expected Value:', explainer.expected_value)
    pd.DataFrame(shap_values).head()
    
    shap.summary_plot(shap_values, data)

    return [explainer, shap_values]


def grid_search(estimator, x_train, y_train, param_grid, scoring=None, n_jobs=-1, cv=10):

    from sklearn.model_selection import GridSearchCV

    # Apply the cross-validation iterator on the Training set using  GridSearchCV. This will run the classifier on the 
    # different train/cv splits using parameters specified and return the model that has the best results 
    # Note that we are tuning based on the F1 score 2PR/P+R where P is Precision and R is Recall. This may not always be 
    # the best score to tune our model on. I will explore this area further in a seperate exercise. For now, we'll use F1. 
    classifier = GridSearchCV(estimator=estimator, cv=cv, param_grid=param_grid, n_jobs=n_jobs, verbose=2, scoring=scoring, return_train_score=True) 
    
    # Also note that we're feeding multiple neighbors to the GridSearch to try out. 
    # We'll now fit the training dataset to this classifier 
    classifier.fit(x_train, y_train) 
    
    #Let's look at the best estimator that was found by GridSearchCV 
    print("")
    print ("Best Estimator") 
    print("---------------------------" )
    print (classifier.best_estimator_)
    print("")

    print("")
    print("Best Estimator Parameters" )
    print("---------------------------" )
    print(classifier.best_params_)
    print("")

    # Each of these parameters is critical to learning. Some of them will help address overfitting issues as well
    return classifier

    ## Call like so:        
    #param_grid={'n_estimators':    [1, 3], 
    #            'learning_rate':   [0.1, 0.05],
    #            'max_depth':       [2,10], 
    #            'min_samples_leaf':[3,17], 
    #            'max_features':    [1.0,0.1]
    #           }
    #n_jobs=4

    ## Let's fit GBRT to the digits training dataset by calling the function we just created.
    #cv, best_est = GradientBooster(param_grid, n_jobs)


    
def tree_viz(model, feature_names, class_names=[]):

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
    print("Run as a lib:")
    l = []
    for key, value in list(locals().items()):
        if callable(value) and value.__module__ == __name__:
            l.append(key)
    print(l)
