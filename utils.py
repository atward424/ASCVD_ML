import numpy as np
import pandas as pd
import scipy.stats as st

#from medical_ML import Experiment
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.dummy import DummyRegressor

def split_cohort(datafile, to_exclude = None, test_ind_col = None, drop = 'some'):
    """ Load and clean the dataset
    """
    if isinstance(datafile, str):
        data = pd.read_csv(datafile)
    else:
        data = datafile
    test_data = None
    if to_exclude is not None:
        for k in to_exclude.keys():
            if k == 'race':
                data = data[data[k].isin(to_exclude[k])]
                
            elif k == 'agebl':
                data = data[data[k] >= to_exclude[k]]
                
            elif to_exclude[k]:
                data = data[data[k] == 0]
                if drop == 'some':
                    data = data.drop(k, axis = 1)
                    
            if drop == 'all':
                if (k != 'race') & (k != 'agebl'):
                    data = data.drop(k, axis = 1)
#         self.data = self.data[self.data['year'] <= 2010]
#         self.data = self.data.drop(['year'], axis = 1)
    if test_ind_col is not None:
        test_data = data[data[test_ind_col] == 1]
        test_data = test_data.drop(test_ind_col, axis = 1)
        data = data[data[test_ind_col] == 0]
        data = data.drop(test_ind_col, axis = 1)
    
    return(data, test_data)

def calc_auc_conf_interval(AUC, N1, N2, ci = 0.95):
    # from https://ncss-wpengine.netdna-ssl.com/wp-content/themes/ncss/pdf/Procedures/PASS/Confidence_Intervals_for_the_Area_Under_an_ROC_Curve.pdf
    zsc = st.norm.ppf(1 - (1-ci)/2.)
    q1 = AUC / (2 - AUC)
    q2 = (2 * AUC * AUC) / (1 + AUC)
    numerator = AUC * (1 - AUC) + (N1 - 1) * (q1 - AUC * AUC) + (N2 - 1) * (q2 - AUC * AUC)
    denom = N1 * N2
    se_AUC = np.sqrt(numerator / denom)
    return (se_AUC, AUC - zsc * se_AUC, AUC, AUC + zsc * se_AUC) 
    
def load_models_and_parameters_default():
    models_and_parameters = {
            
                'dummy_reg': (DummyRegressor(),
                          {"strategy": ["mean"]}),
                'lasso_reg': (linear_model.Lasso(),
                          {'alpha': np.arange(0.1, 1.0, 0.01),
                           'max_iter': [10000]}),
                'rf_reg': (RandomForestRegressor(),
                       {'n_estimators': [501],
                        'criterion': ['mae'],
                        'max_depth': [3, 5, 10],
                        'max_features': ['auto', 'sqrt', 'log2']}),
                'gbm_reg': (GradientBoostingRegressor(),
                        {'n_estimators': [501],
                         'criterion': ['mae'],
                         # 'loss': ['ls', 'lad'],
                         'max_depth': [3, 5, 10],
                         'max_features': ['auto', 'sqrt', 'log2']}),
                'dummy': (DummyClassifier(),
                          {"strategy": ["most_frequent"]}),
#                 'logreg': (LogisticRegression(),
#                           {"class_weight": [None], 
#                            "C":[0.1, 0.3, 1,5, 10]}), #, "balanced"
#                'logreg': (LogisticRegression(),
#                          {"class_weight": [None], 
#                           "C":[0.01,0.1, 1]}), #, "balanced"
#                            "C":[0.1]}), #, "balanced"
                'logreg': (LogisticRegression(),
                          {}), #, "balanced"
#                            "C":[0.1]}), #, "balanced"
                
                'lasso': (Lasso(),
                          {"alpha": [0.0001, 0.001],#np.arange(0.01, 1.01, 0.05),
                           'max_iter': [10000]}), 
                
#                'lasso2': (LogisticRegression(penalty = 'l1'),
#                          {"C":[0.001, 0.01,0.1, 1]}), 
                'lasso2': (LogisticRegression(penalty = 'l1',solver ='saga'),
                          {}), 
                           
                'elnet': (LogisticRegression(penalty = 'elasticnet', solver = 'saga'),
                          {"C":[0.001, 0.01,0.1, 1], 
                           "l1_ratio":[0.01, 0.1, 0.5, 0.9, 0.99]}), 
                'dt': (DecisionTreeClassifier(),
                        {"criterion": ["entropy"],
                         # "max_depth": [2, 3, 4, 5, 10, 20], # None
                         "max_depth": [1, 2, 3, 4], # None
                         "splitter": ["best", "random"],
                         "min_samples_split": [2, 5, 10],
                         "min_samples_leaf": [3, 5, 10, 15, 20],
                         "random_state": [817263]}),
                'svm': (SVC(),
                       {'C': [ 1],
                        'kernel': ['linear']}), #'poly', 'rbf'
                'knn': (KNeighborsClassifier(),
                       {'n_neighbors': [2, 3, 5, 10, 20, 50],
                        'weights': ['uniform', 'distance']}), 
                
                # 'rf': (RandomForestClassifier(),
                #        {'n_estimators': [501],
                #         'max_depth': [3, 5, 10],
                #         'max_features': ['auto', 'sqrt', 'log2']}),
                # 'rf': (RandomForestClassifier(),
                #        {'n_estimators': [50, 100, 501, 1000],
                #         'max_depth': [3,5,7],
                #          "min_samples_split": [2, 5],
                #         'max_features': ['auto', 0.5],
                #         "class_weight": [None, "balanced"]}),
                # 'rf': (RandomForestClassifier(),
                #        {'n_estimators': [501],
                #         'max_depth': [5],
                #          "min_samples_split": [5],
                #         'max_features': ['auto'],
                #         "class_weight": [None]}),
#                 'rf': (RandomForestClassifier(),
#                        {'n_estimators': [ 501, 1000, 2000, 4000],
#                         'max_depth': [5, 7, 9, 11, 13],
#                          "min_samples_split": [2],
#                         'max_features': ['sqrt', 0.25, 0.5, 0.75, 1.0],
#                         "class_weight": [None]}),
#                 'rf': (RandomForestClassifier(),
#                        {'n_estimators': [200, 500, 1000],
#                         'max_depth': [4, 6, 8, 10],
#                          "min_samples_split": [2, 10],
#                         'max_features': [0.25, 0.5],
#                         "class_weight": [None]}),
                 'rf': (RandomForestClassifier(),
                        {'n_estimators': [800],
                         'max_depth': [8],
                          "min_samples_split": [10],
                         'max_features': [0.25],
                         "class_weight": [None]}),
#                 'rf': (RandomForestClassifier(),
#                        {'n_estimators': [400, 500, 600],
#                         'max_depth': [7,8,9],
#                          "min_samples_split": [5,10],
#                         'max_features': [0.25, 0.5, ]}),
#                'rf': (RandomForestClassifier(),
#                       {}),
                    
                'xgb': (xgb.XGBClassifier(),
                       {}),
#                'rf': (RandomForestClassifier(),
#                       {'n_estimators': [600],
#                        'max_depth': [9],
#                         "min_samples_split": [10],
#                        'max_features': [0.25]}),
#                    
#                'xgb': (xgb.XGBClassifier(),
#                       {'n_estimators': [100,500],
#                        'max_depth': [3,4,5],
#                        'learning_rate': [0.1, 0.3],
#                        "reg_alpha": [0,   1],
#                        "reg_lambda": [0.1, 1]}),
#                'xgb': (xgb.XGBClassifier(),
#                       {'n_estimators': [500],
#                        'max_depth': [4],
#                        'learning_rate': [0.1],
#                        "reg_alpha": [0,   10],
#                        "reg_lambda": [0.1, 10]}),
                
#                'gbm': (GradientBoostingClassifier(),
#                        {'n_estimators': [200, 300],
#                         'learning_rate': [0.01],
#                         'max_depth': [3,4,5],
#                         'subsample': [0.35, 0.7],
#                         'max_features': [0.25]}),
#                 'gbm': (GradientBoostingClassifier(),
#                         {'n_estimators': [400],
#                          'learning_rate': [0.01],
#                          'max_depth': [5],
#                          'subsample': [0.75],
#                          'max_features': [0.25]}),
#                 'gbm': (GradientBoostingClassifier(),
#                         {'n_estimators': [300, 400, 500],
#                          'learning_rate': [0.01, 0.003, 0.4],
#                          'max_depth': [5, 6, 7],
#                          'subsample': [0.85, 1],
#                          'max_features': [0.25, 0.5]}),
                 'gbm': (GradientBoostingClassifier(),
                         {}),
#                 'gbm': (GradientBoostingClassifier(),
#                         {'n_estimators': [100, 200, 300, 500, 1000, 2000, 
#                         4000],
#                          'max_depth': [2, 3, 4, 5, 6, 7, 
#                          9],
#                          'subsample': [0.75, 
#                          1],
#                          'max_features': ['sqrt', 'log2', 0.25, 0.5, 0.75, 
#                          1.0]}),
#                 'gbm': (GradientBoostingClassifier(),
#                         {'n_estimators': [100, 200, 400, 800],
#                          'learning_rate': [0.03, 0.01, 0.001],
#                          'max_depth': [4,5,6,8],
#                          'subsample': [0.85],
#                          'max_features': [0.25, 0.5]}),
#                 'gbm': (GradientBoostingClassifier(),
#                         {'n_estimators': [400, 600],
#                          'learning_rate': [0.01],
#                          'max_depth': [5, 6],
#                          'subsample': [0.85],
#                          'max_features': [0.25]}),
                # 'gbm': (GradientBoostingClassifier(),
                #         {'n_estimators': [25, 50, 75, 100, 200],
                #          'max_depth': [2,3,5],
                #          'subsample': [0.25, 0.5, 0.75, 1],
                #          'max_features': [None, 'sqrt', 'log2', 0.5]}),
            }
             
    return(models_and_parameters)
    
def load_models_and_parameters():
    models_and_parameters = {
            
                'dummy_reg': (DummyRegressor(),
                          {"strategy": ["mean"]}),
                'lasso_reg': (linear_model.Lasso(),
                          {'alpha': np.arange(0.1, 1.0, 0.01),
                           'max_iter': [10000]}),
                'rf_reg': (RandomForestRegressor(),
                       {'n_estimators': [501],
                        'criterion': ['mae'],
                        'max_depth': [3, 5, 10],
                        'max_features': ['auto', 'sqrt', 'log2']}),
                'gbm_reg': (GradientBoostingRegressor(),
                        {'n_estimators': [501],
                         'criterion': ['mae'],
                         # 'loss': ['ls', 'lad'],
                         'max_depth': [3, 5, 10],
                         'max_features': ['auto', 'sqrt', 'log2']}),
                'dummy': (DummyClassifier(),
                          {"strategy": ["most_frequent"]}),
#                 'logreg': (LogisticRegression(),
#                           {"class_weight": [None], 
#                            "C":[0.1, 0.3, 1,5, 10]}), #, "balanced"
               'logreg': (LogisticRegression(),
                         {"class_weight": [None], 
                          "C":[0.01,0.1, 1]}), #, "balanced"
#                            "C":[0.1]}), #, "balanced"
        
#                 'logreg': (LogisticRegression(),
#                           {}), #, "balanced"
# #                            "C":[0.1]}), #, "balanced"
                
                'lasso': (Lasso(),
                          {"alpha": [0.0001, 0.001],#np.arange(0.01, 1.01, 0.05),
                           'max_iter': [10000]}), 
                
               'lasso2': (LogisticRegression(penalty = 'l1', solver ='saga'),
                         {"C":[0.001, 0.01,0.1, 1]}), 
#                 'lasso2': (LogisticRegression(penalty = 'l1'),
#                           {}), 
                           
                'elnet': (LogisticRegression(penalty = 'elasticnet', solver = 'saga'),
                          {"C":[0.001, 0.01,0.1, 1], 
                           "l1_ratio":[0.01, 0.1, 0.5, 0.9, 0.99]}), 
                'dt': (DecisionTreeClassifier(),
                        {"criterion": ["entropy"],
                         # "max_depth": [2, 3, 4, 5, 10, 20], # None
                         "max_depth": [1, 2, 3, 4], # None
                         "splitter": ["best", "random"],
                         "min_samples_split": [2, 5, 10],
                         "min_samples_leaf": [3, 5, 10, 15, 20],
                         "random_state": [817263]}),
                'svm': (SVC(),
                       {'C': [ 1],
                        'kernel': ['linear']}), #'poly', 'rbf'
                'knn': (KNeighborsClassifier(),
                       {'n_neighbors': [2, 3, 5, 10, 20, 50],
                        'weights': ['uniform', 'distance']}), 
                
                # 'rf': (RandomForestClassifier(),
                #        {'n_estimators': [501],
                #         'max_depth': [3, 5, 10],
                #         'max_features': ['auto', 'sqrt', 'log2']}),
                # 'rf': (RandomForestClassifier(),
                #        {'n_estimators': [50, 100, 501, 1000],
                #         'max_depth': [3,5,7],
                #          "min_samples_split": [2, 5],
                #         'max_features': ['auto', 0.5],
                #         "class_weight": [None, "balanced"]}),
                # 'rf': (RandomForestClassifier(),
                #        {'n_estimators': [501],
                #         'max_depth': [5],
                #          "min_samples_split": [5],
                #         'max_features': ['auto'],
                #         "class_weight": [None]}),
#                 'rf': (RandomForestClassifier(),
#                        {'n_estimators': [ 501, 1000, 2000, 4000],
#                         'max_depth': [5, 7, 9, 11, 13],
#                          "min_samples_split": [2],
#                         'max_features': ['sqrt', 0.25, 0.5, 0.75, 1.0],
#                         "class_weight": [None]}),
#                 'rf': (RandomForestClassifier(),
#                        {'n_estimators': [200, 500, 1000],
#                         'max_depth': [4, 6, 8, 10],
#                          "min_samples_split": [2, 10],
#                         'max_features': [0.25, 0.5],
#                         "class_weight": [None]}),
                'rf': (RandomForestClassifier(),
                       {'n_estimators': [500, 1000],
                        'max_depth': [8],
                         "min_samples_split": [10],
                        'max_features': [0.25],
                        "class_weight": [None]}),
#                 'rf': (RandomForestClassifier(),
#                        {'n_estimators': [400, 500, 600],
#                         'max_depth': [7,8,9],
#                          "min_samples_split": [5,10],
#                         'max_features': [0.25, 0.5, ]}),
#                 'rf': (RandomForestClassifier(),
#                        {}),
                    
#                 'xgb': (xgb.XGBClassifier(),
#                        {}),
#                'rf': (RandomForestClassifier(),
#                       {'n_estimators': [600],
#                        'max_depth': [9],
#                         "min_samples_split": [10],
#                        'max_features': [0.25]}),
#                    
#                'xgb': (xgb.XGBClassifier(),
#                       {'n_estimators': [100,500],
#                        'max_depth': [3,4,5],
#                        'learning_rate': [0.1, 0.3],
#                        "reg_alpha": [0,   1],
#                        "reg_lambda": [0.1, 1]}),
#                'xgb': (xgb.XGBClassifier(),
#                       {'n_estimators': [500],
#                        'max_depth': [4],
#                        'learning_rate': [0.1],
#                        "reg_alpha": [0,   10],
#                        "reg_lambda": [0.1, 10]}),
                
#                'gbm': (GradientBoostingClassifier(),
#                        {'n_estimators': [200, 300],
#                         'learning_rate': [0.01],
#                         'max_depth': [3,4,5],
#                         'subsample': [0.35, 0.7],
#                         'max_features': [0.25]}),
#                 'gbm': (GradientBoostingClassifier(),
#                         {'n_estimators': [400],
#                          'learning_rate': [0.01],
#                          'max_depth': [5],
#                          'subsample': [0.75],
#                          'max_features': [0.25]}),
#                 'gbm': (GradientBoostingClassifier(),
#                         {'n_estimators': [300, 400, 500],
#                          'learning_rate': [0.01, 0.003, 0.4],
#                          'max_depth': [5, 6, 7],
#                          'subsample': [0.85, 1],
#                          'max_features': [0.25, 0.5]}),
#                  'gbm': (GradientBoostingClassifier(),
#                          {}),
#                 'gbm': (GradientBoostingClassifier(),
#                         {'n_estimators': [100, 200, 300, 500, 1000, 2000, 
#                         4000],
#                          'max_depth': [2, 3, 4, 5, 6, 7, 
#                          9],
#                          'subsample': [0.75, 
#                          1],
#                          'max_features': ['sqrt', 'log2', 0.25, 0.5, 0.75, 
#                          1.0]}),
                'gbm': (GradientBoostingClassifier(),
                        {'n_estimators': [100, 200, 400, 800],
                         'learning_rate': [0.03, 0.01, 0.001],
                         'max_depth': [4,5,6,8],
                         'subsample': [0.85],
                         'max_features': [0.25, 0.5]}),
#                 'gbm': (GradientBoostingClassifier(),
#                         {'n_estimators': [400, 600],
#                          'learning_rate': [0.01],
#                          'max_depth': [5, 6],
#                          'subsample': [0.85],
#                          'max_features': [0.25]}),
                # 'gbm': (GradientBoostingClassifier(),
                #         {'n_estimators': [25, 50, 75, 100, 200],
                #          'max_depth': [2,3,5],
                #          'subsample': [0.25, 0.5, 0.75, 1],
                #          'max_features': [None, 'sqrt', 'log2', 0.5]}),
            }
             
    return(models_and_parameters)

def calc_metrics(y_true, y_pred, return_all = False):
    res_df = pd.DataFrame({'y_true' : y_true, 
                    'y_pred': y_pred}, columns = ['y_pred', 'y_true'])
    res_df = res_df.sort_values(by = 'y_pred')
    res_df['TN'] = (res_df.y_true == 0).cumsum()
    res_df['FN'] = (res_df.y_true == 1).cumsum()
    if return_all == False:
        res_df = pd.concat([pd.DataFrame({'y_true' : -1, 
                                        'y_pred': -1, 
                                           "TN": 0,
                                           "FN":0}, 
                                         index = [-1],
                                        columns = ['y_pred', 'y_true', 'TN', "FN"]),
                           res_df], axis = 0)
                                         
    res_df['TP'] = (res_df.y_true == 1).sum() - res_df['FN']
    res_df['FP'] = (res_df.y_true == 0).sum() - res_df['TN']
    res_df['sens'] = res_df.TP / (res_df.TP + res_df.FN)
    res_df['spec'] = res_df.TN / (res_df.TN + res_df.FP)
    res_df['PPV'] = res_df.TP / (res_df.TP + res_df.FP)
    res_df['accuracy'] = (res_df.TP + res_df.TN) / (res_df.shape[0])
    res_df['f1_score'] = 2 * res_df.PPV * res_df.sens / (res_df.PPV  + res_df.sens)
    res_df['youdens_index'] =  res_df.sens + res_df.spec - 1
    # remove predictions which represent non-separable decision points (i.e., y_pred is equal)
    if return_all == False:
        res_df = res_df[(res_df.y_pred.duplicated('last') == False)]
    return(res_df)

def set_up_plot():
#    plt.grid(True, 'major', color = 'w', linewidth = 0.7)
    plt.grid(True, 'major', color = '0.85', linewidth = 0.7)
    plt.grid(True, 'minor', color = "0.92", linestyle = '-', linewidth = 0.7)
    ax = plt.gca()
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    ax.set_axisbelow(True)
#    ax.patch.set_facecolor("0.85")
    
def train_val(RESULT_DIR, alldata, models, label = 'Label', 
              cv = 5, 
              score_name = "AUC", 
              to_exclude = None, 
              test_ind_col = None,   oversample_rate = 1,
                  imputer = 'iterative', add_missing_flags = True):
    
    from medical_ML import Experiment
    print('\n\n' + 'STARTING EXPERIMENT FOR ' + RESULT_DIR + '\n\n')
    expt = Experiment(alldata, label = label, 
                      to_exclude = to_exclude, 
                      test_ind_col = test_ind_col, drop = 'all', 
                      result_dir = RESULT_DIR)
    expt.predict_models_from_groups(0, models, cv=cv, score_name=score_name, mode='classification',
                                                    oversample_rate = oversample_rate, 
                                                   imputer = imputer, add_missing_flags = add_missing_flags)
    expt.save_and_plot_results(models, 
                               cv = cv, test = False)
    return(expt)