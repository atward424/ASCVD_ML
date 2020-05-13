""" Scipy wrappers for Medical machine learning models
"""
import os

import numpy as np
import json
import itertools
#import subprocess
#import pickle 
#import time
from scipy import interp

#from sympy import Matrix

from joblib import dump, load
from sklearn.preprocessing import LabelBinarizer, PolynomialFeatures, StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer, MissingIndicator

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import KFold, cross_val_score, cross_validate, GridSearchCV, RandomizedSearchCV, StratifiedKFold

from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error

#import xgboost as xgb
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, auc, roc_auc_score, roc_curve
import scipy.stats as st

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd

FIG_HT = 6
FIG_WD = 4.5

FIG_XLABEL = "1 - Specificity"
FIG_YLABEL = "Sensitivity"

class Experiment():
    """
    Experimentation model for the carotid ultrasound data.
    Handles data loading/cleaning, model training, fitting, and result formatting.
    """

    def __init__(self, datafile, label, to_exclude = None, test_ind_col = None, drop = 'none', result_dir = None):
        """ Load and clean the dataset
        """
        assert result_dir is not None, 'Must specify a result directory'
        if not os.path.isdir(result_dir): os.mkdir(result_dir)
        self.result_dir = result_dir
        self.model_names = {'xgb': 'XGBoost',
                            'rf': 'RF',
                            'gbm':'GBM',
                            'lasso':'Lasso',
                            'logreg':'LR',
                            'PCE': 'PCE',
                            'lasso2':'Lasso', 
                            'nb':'Naive-Bayes', 
                            'svm':'SVM', 
                            'elnet':'E-Net'}
        if datafile is None:
            return
        data, test_data = split_cohort(datafile, to_exclude, test_ind_col, drop = drop)
        self.data = data
        self.test_data = test_data
        assert label in self.data.columns, "Label {} not in dataset".format(label)
        self.label = label
        print("Loaded {} samples from {}".format(len(self.data), datafile))
        self.format_data()

    def format_data(self):
        """ Format the dataframe for prediction models
        """

        columns = self.data.columns
#         self.data[columns] = self.data[columns].convert_objects(convert_numeric=True)

        # Convert all string columns to one-hot dummy columns
        self.categorical_prefixes = [c for c in columns if self.data[c].dtype=='O']
        # print("String Columns:\n{}".format(string_columns))
        print("Found {} categorical variables.  Total of {} variables".format(len(self.categorical_prefixes), len(self.data.columns)))
        self.data = pd.get_dummies(self.data, columns = self.categorical_prefixes)
        print("Total of {} variables after dummy conversion dummies".format(len(self.data.columns)))
#         self.data.to_csv("../Data/with_dummies.csv")
        # self.data.drop(string_columns, axis=1, inplace=True)

        # # Convert Date columns
        # date_columns = 'Date of birth'
        # for dc in date_columns:
        #     self.data
        self.features = [f for f in self.data.columns if f != self.label]
        self.valid_features = self.features


    def load_feature_groups(self, filepath, save_categorical=True):
        """
        Load feature groups dictionary from json file

        Args:
            filepath:  (str) full path to groups .json file
        """
        with open(filepath) as jfile:
            raw_feature_groups = json.load(jfile)

        self.feature_groups = {}
        for group in raw_feature_groups:
            self.feature_groups[group] = []
            for feature in raw_feature_groups[group]:
                for column in self.data.columns:
                    if column.startswith(feature):
                        self.feature_groups[group].append(column)

        if save_categorical:
            with open("ACT_feature_groups_with_categoricals.json", 'w') as jfile:
                json.dump(self.feature_groups, jfile, indent=4, separators=(',', ': '))


    def preprocessing(self, features, model='regression', verbose=False):
        """ Preprocess the data and convert into usable format for sklearn
        """

        # Check all columns in dataset
        valid_features = []
        for feature in features:
            if feature in self.data.columns:
                valid_features.append(feature)
            else:
                print("WARNING:  Feature {} not in dataset".format(feature))
        
        # Drop rows with a missing label
        valid_samples = np.isfinite(self.data.loc[:, self.label])
        if np.sum(~valid_samples) > 0:
            print("WARNING:  Dropping {} samples because of empty label {}".format(np.sum(~valid_samples), self.label))

        # Split into input and output
        self.y = self.data.loc[valid_samples, self.label]
        self.X = self.data.loc[valid_samples, valid_features]
        self.valid_features = valid_features
        
        # create interactions between features -- this is a workaround to handle NaNs
#        if self.models[0] == 'lasso':
#            min_x = min(np.min(np.min(self.X[self.X > 0])), 1)
#            X2 = self.X.fillna(-min_x / 2.0)
#            poly = PolynomialFeatures(2, interaction_only = True)
#            X2 = poly.fit_transform(X2)
#            X2 = pd.DataFrame(data = X2, columns = poly.get_feature_names(self.X.columns))
#            X2[((X2 > 0) & (X2 < (min_x / 1.5))) | (X2 < 0)] = np.nan
#            interaction_colsums = np.sum(X2)
#            X3 = X2[interaction_colsums[interaction_colsums > 0].index[1:]]
#    #         _, independent_cols = Matrix(X3).rref()
##             X3 = X3.iloc[:, 0:300]
#            self.X = X3
#            self.valid_features = X3.columns
        


        # Bin the outcome variable for classification model
#        if model == 'classification':
#            self.y = self.classify(y)
#        elif model == 'regression':
#            if verbose: print("Regression model")
#        else:
#            print("Unknown model type {}".format(model))
#            raise NotImplementedError
#
#        # Convert categorical features to one-hot encoded dummy columns
#        lb = LabelBinarizer()

        if verbose: print("y: {}\n{}".format(len(self.y), self.y[:5]))
        if verbose: print("X: {}\n{}".format(len(self.X), self.X[:5]))
        if verbose: print("Data types:\n{}".format([self.X[c].dtype for c in self.X.columns]))


        return self.y, self.X


    def predict_from_groups(self, groups, est, cv=5, scoring=None, verbose=True, custom_features=None):
        """ 
        Run a predictive model on all features from the passed groups

        Args:
            groups: (list) List of groups to take predictive features from
            est:  (estimator object implementing 'fit') The object to use to fit the data.
        """
        if verbose: print("Predicting {} from {} feature groups using {}".format(self.label, groups, est))
        self.features = [f for group in groups for f in self.feature_groups[group]]
        if custom_features is not None:
            self.features = custom_features
        return self.predict(self.features, est, cv=cv, scoring=scoring)


    def predict(self, est, cv, scoring=None, model='regression', verbose=False, imputer = 'iterative', add_missing_flags = True):
        """
        Predict output label from input features using the given model

        Args:
            features: (list) List of predictive input features to use
            est:  (estimator object implementing 'fit') The object to use to fit the data.
        """
        if verbose: print("Predicting {} from {} features using model\n{}".format(self.label, len(self.features), est))
        
        print("dim of data: {}".format(len(self.features)))
        # Preprocess the data and format for sklearn models
        y, X = self.preprocessing(self.features)
        
        self.feature_list = self.valid_features
        if imputer:
            if imputer == 'iterative':
                imputer = IterativeImputer(max_iter = 50, 
                                         n_nearest_features = 10)
            elif imputer == 'simple':
                imputer = SimpleImputer()
            else:
                print("Unknown imputer type {}".format(imputer))
                raise NotImplementedError
            
            if add_missing_flags:
                imputer = FeatureUnion([("missing", MissingIndicator(features = 'all', error_on_new = False)), 
                                       ("imputer", imputer)])
                
                self.feature_list = [s + '_missing' for s in self.valid_features] + self.valid_features
        # Impute using the CV training fold(s)
            est_pipe = Pipeline([
                 ('imputer', imputer),                  
    #             ,
    #                             ('scaler', StandardScaler()),
    #                              ("poly_transform", PolynomialFeatures(degree=2,
    #                                                  interaction_only = True)),
                ("predictor", est)])
        else:
            est_pipe = Pipeline([("predictor", est)])
        # Run the estimator across a K-fold cross-validation
        print("Pipeline:", [name for name, _ in est_pipe.steps])
        # scores = cross_val_score(est_pipe, X, y, cv=cv, scoring=scoring)

#        scores = cross_validate(est_pipe, X, y, cv=cv, scoring=scoring, 
#                                return_train_score=True, 
#                                return_estimator = True)
        
        est_pipe.fit(X.to_numpy(), y)
        self.save_cv_results(est.cv_results_, filepath = self.result_dir + "/" + self.models[0] + "_cv_results.csv")

        best_model = est_pipe.named_steps['predictor'].best_estimator_
        if hasattr(est_pipe.named_steps['predictor'], 'best_params_'):
            print("Best CV parameters:\n{}",format(est_pipe.named_steps['predictor'].best_params_))
            self.save_scores(est_pipe.named_steps['predictor'].best_params_, filepath = self.result_dir + "/" + self.models[0] + "_best_params.json")
        # if hasattr(est_pipe.named_steps['predictor'], 'feature_importances_'):
        if hasattr(best_model, 'feature_importances_'):
#            feature_importances = {self.features[i]: best_model.feature_importances_[i] for i in np.argsort(best_model.feature_importances_)[::-1]
#                                   if best_model.feature_importances_[i] != 0}
            feat_imp_order = np.argsort(best_model.feature_importances_)[::-1]
            feat_import_df = pd.DataFrame(data = {'feature': [self.feature_list[i] for i in feat_imp_order],
                                                                  'importance': [best_model.feature_importances_[i] for i in feat_imp_order]})
            feat_import_df.to_csv(self.result_dir + "/" + self.models[0] + "_feature_importances.csv", index = False)
            print("Feature Importances:\n{}".format(feat_import_df))
        print("Best Model:\n{}".format(best_model))
        
        est_pipe.steps[-1] = ('predictor', best_model)      
        
        dump(est_pipe, os.path.join(self.result_dir, self.models[0] + '_best_model.joblib'))
        # impute missing values in X for ROC plot
#        self.imp = est_pipe.named_steps['imputer']
#        X = self.imp.transform(X)

        self.plot_roc(est_pipe, X, y, cv, filepath = self.result_dir +'/auc_plot_' + self.models[0] + '.png')
        # cv2 = StratifiedKFold(n_splits=cv)


        # Average the results over all CV folds
        # cv_score = np.mean(scores)
#        return scores, best_model
        return {}, best_model


    def predict_models_from_groups(self, groups, models, cv=5, mode="regression", score_name=None, 
                                   graph=None, verbose=True, custom_features=None, 
                                   oversample_rate = 1, imputer = 'iterative', add_missing_flags = True):
        """ Run a list of models on the all features from the passed groups

        Args:
            groups: (list) List of groups to take predictive features from
            models:  (list) All estimator (names) to use to fit the data.
        """
        self.models = models
        if verbose: print("Predicting {} from {} feature groups using [{}] models".format(self.label, groups, models))
        scores = {}

        # `outer_cv` creates K folds for estimating generalization error
        outer_cv = StratifiedKFold(cv)

        # when we train on a certain fold, we use a second cross-validation
        # split in order to choose hyperparameters
        inner_cv = StratifiedKFold(cv-1)

        if mode == "regression":
            # Score all models on MAE
            if score_name is None or score_name == "MAE":
                scorer = make_scorer(mean_absolute_error)
                score_name = "MAE"
            else:
                raise NotImplementedError

            models_and_parameters = {
                'dummy': (DummyRegressor(),
                          {"strategy": ["mean"]}),
                'lasso': (linear_model.Lasso(),
                          {'alpha': np.arange(0.1, 1.0, 0.01),
                           'max_iter': [10000]}),
                'rf': (RandomForestRegressor(),
                       {'n_estimators': [501],
                        'criterion': ['mae'],
                        'max_depth': [3, 5, 10],
                        'max_features': ['auto', 'sqrt', 'log2']}),
                'gbm': (GradientBoostingRegressor(),
                        {'n_estimators': [501],
                         'criterion': ['mae'],
                         # 'loss': ['ls', 'lad'],
                         'max_depth': [3, 5, 10],
                         'max_features': ['auto', 'sqrt', 'log2']}),
            }
        elif mode == "classification":
            # Make classification scorer, default to F1
            if score_name is None or score_name == "F1":
                scorer = make_scorer(f1_score)
                score_name = "F1"
            elif score_name == "AUC":
                scorer = 'roc_auc'
            elif score_name == "Accuracy":
                scorer = make_scorer(accuracy_score)
            else:
                raise NotImplementedError

            models_and_parameters = {
                'dummy': (DummyClassifier(),
                          {"strategy": ["most_frequent"]}),
#                 'logreg': (LogisticRegression(),
#                           {"class_weight": [None], 
#                            "C":[0.1, 0.3, 1,5, 10]}), #, "balanced"
                'logreg': (LogisticRegression(),
                          {"class_weight": [None], 
                           "C":[0.01,0.1, 1]}), #, "balanced"
#                            "C":[0.1]}), #, "balanced"
                
                'lasso': (Lasso(),
                          {"alpha": [0.0001, 0.001],#np.arange(0.01, 1.01, 0.05),
                           'max_iter': [10000]}), 
                
                'lasso2': (LogisticRegression(penalty = 'l1'),
                          {"C":[ 0.3, 1, 5]}), 
                           
                'elnet': (LogisticRegression(penalty = 'elasticnet', solver = 'saga', 
                                             max_iter = 1000),
                          {"C":[0.1, 1, 10], 
                           "l1_ratio":[0.001, 0.01, 0.03]}), 
                'dt': (DecisionTreeClassifier(),
                        {"criterion": ["entropy"],
                         # "max_depth": [2, 3, 4, 5, 10, 20], # None
                         "max_depth": [1, 2, 3, 4], # None
                         "splitter": ["best", "random"],
                         "min_samples_split": [2, 5, 10],
                         "min_samples_leaf": [3, 5, 10, 15, 20],
                         "random_state": [817263]}),

                'svm': (SVC(),
                       {'C': [1, 0.1],
                        'gamma': ['auto', 'scale', 1],
                        'kernel': ['rbf', 'linear', 'poly'],
                    'probability': [True]}), #'poly', 'rbf', #linear
            

                'nb': (GaussianNB(),
                       {}),

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
#                 'rf': (RandomForestClassifier(),
#                        {'n_estimators': [500, 1000],
#                         'max_depth': [8],
#                          "min_samples_split": [10],
#                         'max_features': [0.25],
#                         "class_weight": [None]}),
#                 'rf': (RandomForestClassifier(),
#                        {'n_estimators': [400, 500, 600],
#                         'max_depth': [7,8,9],
#                          "min_samples_split": [5,10],
#                         'max_features': [0.25, 0.5, ]}),
                'rf': (RandomForestClassifier(),
                       {'n_estimators': [600],
                        'max_depth': [9],
                         "min_samples_split": [10],
                        'max_features': [0.25]}),
                    
#                'xgb': (xgb.XGBClassifier(),
#                       {'n_estimators': [100,500],
#                        'max_depth': [3,4,5],
#                        'learning_rate': [0.1, 0.3],
#                        "reg_alpha": [0,   1],
#                        "reg_lambda": [0.1, 1]}),
#                 'xgb': (xgb.XGBClassifier(),
#                        {'n_estimators': [500],
#                         'max_depth': [4],
#                         'learning_rate': [0.1],
#                         "reg_alpha": [2, 10, 30],
#                         "reg_lambda": [5, 10]}),
#                'xgb': (xgb.XGBClassifier(),
#                       {'n_estimators': [500],
#                        'max_depth': [4],
#                        'learning_rate': [0.1],
#                        "reg_alpha": [30],
#                        "reg_lambda": [10]}),
                
                'gbm': (GradientBoostingClassifier(),
                        {'n_estimators': [300],
                         'learning_rate': [0.01],
                         'max_depth': [5],
                         'subsample': [0.35],
                         'max_features': [0.25]}),
#                 'gbm': (GradientBoostingClassifier(),
#                         {'n_estimators': [200, 300],
#                          'learning_rate': [0.01],
#                          'max_depth': [3,4,5],
#                          'subsample': [0.35, 0.7],
#                          'max_features': [0.25]}),
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
#                 'gbm': (GradientBoostingClassifier(),
#                         {'n_estimators': [100, 200, 500, 1000, 2000, 
#                         4000],
#                          'max_depth': [2, 3, 4, 5, 7, 
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

        # Restrict to the currently selected models
        models_and_parameters = {name: mp for name, mp in models_and_parameters.items() if name in models}

        cv_splits = []  
#        print(oversample_rate)
        y, X = self.preprocessing(self.features)
        for train_inds, test_inds in outer_cv.split(X, y):
            pos_train_inds = pd.Series(train_inds[y.iloc[train_inds] == 1])
            if len(pos_train_inds) > 0:
                oversampled = pos_train_inds.sample(int(np.ceil(len(pos_train_inds)*(oversample_rate-1))), replace = True).values
                ti2 = np.concatenate([train_inds, oversampled])
                cv_splits.append((ti2, test_inds))
#                print((len(train_inds), len(ti2)))
            else:
                cv_splits.append((train_inds, test_inds))
                
#        average_scores_across_outer_folds_for_each_model = dict()
#        sd_scores_across_outer_folds_for_each_model = dict()
        score_summary = None
        for name in models:
            model, params = models_and_parameters[name]
#             if name in ['logreg', 'lasso']:
            gsv = GridSearchCV(estimator=model, 
                               param_grid=params, 
                               cv=cv_splits, # used to be inner_cv, when we did the double cv-loop
                               scoring=scorer, 
                               verbose=True, 
                               refit = True)
#             else:
#                 gsv =  RandomizedSearchCV(estimator = model,
#                                        param_distributions = params,
#                                        n_iter = 30,
#                                        cv=inner_cv,
#                                        scoring = scorer,
#                                        verbose = True, 
#                                        refit = True
#                                        n_jobs = -1)

            # scores_across_outer_folds, fit_model = self.predict_from_groups(label, groups, gsv, cv=outer_cv, scoring=scorer, custom_features=custom_features)
            scores_across_outer_folds, fit_model = self.predict(gsv, cv=outer_cv, scoring=scorer, 
                                                                imputer = imputer, add_missing_flags = add_missing_flags)
#             print("Score info:\n{}".format(scores_across_outer_folds))

            # average_scores_across_outer_folds_for_each_model["{}-test-mean".format(name)] = np.mean(scores_across_outer_folds["test_score"])
            # sd_scores_across_outer_folds_for_each_model["{}-test-std".format(name)] = np.std(scores_across_outer_folds["test_score"])
            # error_summary = 'Model: {name}\n{score_name} in the outer folds: {scores}.\nAverage: {avg}'
            # print(error_summary.format(
            #         name=name, score_name=score_name,
            #         scores=scores_across_outer_folds,
            #         avg=np.mean(scores_across_outer_folds)))
            score_summary = {}
#                "{}-test-mean".format(name): np.mean(scores_across_outer_folds["test_score"]),
#                "{}-test-std".format(name): np.std(scores_across_outer_folds["test_score"]),
#                "{}-train-mean".format(name): np.mean(scores_across_outer_folds["train_score"]),
#                "{}-train-std".format(name): np.std(scores_across_outer_folds["train_score"])
#            }
                
        return score_summary
        # return average_scores_across_outer_folds_for_each_model, sd_scores_across_outer_folds_for_each_model

    def predict_on_test(self, models, test_file = None, out_dir = None):
        '''Predict using the best fit models on test data
        '''
        if out_dir is None:
            out_dir = self.result_dir
            
        if test_file is not None:
            test_data = pd.read_csv(test_file)
        else:
            test_data = self.test_data
        
        test_data = pd.get_dummies(test_data, columns = self.categorical_prefixes)

#        valid_samples = np.isfinite(test_data.loc[:, self.label])
#        if np.sum(~valid_samples) > 0:
#            print("WARNING:  Dropping {} testsamples because of empty label {}".format(np.sum(~valid_samples), self.label))
        
        test_y = test_data.loc[:, self.label]
        test_X = test_data.loc[:, test_data.columns != self.label]
        for c in self.valid_features:
            if c not in test_data.columns:
                test_X.insert(len(test_X.columns), c, 0)
        
        for model_name in models:
            model = load(os.path.join(self.result_dir, model_name + '_best_model.joblib'))
            
            if model_name == 'lasso':
                probas_ = model.predict(test_X)
                
            else:
                probas_ = model.predict_proba(test_X.to_numpy())
                probas_ = probas_[:, 1]


            test_result = pd.DataFrame({'y': test_y, 
                                        'est_prob': probas_})
            test_result.to_csv(out_dir + '/test_predicted_probs_' + model_name + '.csv', index = False)



    def plot_roc(self, best_model, X, y, cv, filepath):
        '''plots ROC using cv cross validations
        '''

        plt.clf()
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)

        
        if self.models[0] == 'lasso':
            coefs_ = best_model.fit(X, y).named_steps['predictor'].coef_
            coef_dataframe = pd.DataFrame({'variable': self.feature_list, 'coeff': coefs_})
            coef_dataframe = coef_dataframe.iloc[(-coef_dataframe['coeff'].abs()).argsort()]
            coef_dataframe.to_csv('{}/{}_coefficients.csv'.format(self.result_dir, self.models[0]), index = False)
            print(coef_dataframe)
        if self.models[0] == 'logreg':
            coefs_ = best_model.fit(X, y).named_steps['predictor'].coef_
            coef_dataframe = pd.DataFrame({'variable': self.feature_list, 'coeff': coefs_[0]})
            coef_dataframe = coef_dataframe.iloc[(-coef_dataframe['coeff'].abs()).argsort()]
            coef_dataframe.to_csv('{}/{}_coefficients.csv'.format(self.result_dir, self.models[0]), index = False)
            print(coef_dataframe)
            
        i = 0
        print('refitting on training data...')
        for train, test in cv.split(X, y):
            
            tr_probas_ = 0
            probas_ = 0
            fold_fit = best_model.fit(X.iloc[train].to_numpy(), y.iloc[train])
            if self.models[0] == 'lasso':
                tr_probas_ = fold_fit.predict(X.iloc[train])
                probas_ = fold_fit.predict(X.iloc[test])
                
            else:
                tr_probas_ = fold_fit.predict_proba(X.iloc[train].to_numpy())
                tr_probas_ = tr_probas_[:, 1]
                probas_ = fold_fit.predict_proba(X.iloc[test].to_numpy())
                probas_ = probas_[:, 1]
            
            tr_result = pd.DataFrame({'index': train, 
                                        'y': y.iloc[train], 
                                        'est_prob': tr_probas_})
            tr_result.to_csv(self.result_dir + '/train_predicted_probs_' + self.models[0] + '_fold_' + str(i) + '.csv', index = False)
            cv_result = pd.DataFrame({'index': test, 
                                        'y': y.iloc[test], 
                                        'est_prob': probas_})
            cv_result.to_csv(self.result_dir + '/predicted_probs_' + self.models[0] + '_fold_' + str(i) + '.csv', index = False)
            
            # Compute ROC curve and area under the curve
            fpr, tpr, thresholds = roc_curve(y.iloc[test], probas_)
            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            plt.plot(fpr, tpr, lw=1, alpha=0.3,
                     label='ROC fold %d (AUC = %0.3f)' % (i, roc_auc))

            i += 1
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                 label='Chance', alpha=.8)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        plt.plot(mean_fpr, mean_tpr, color='b',
                 label=r'Mean ROC (AUC = %0.3f $\pm$ %0.3f)' % (mean_auc, std_auc),
                 lw=2, alpha=.8)

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                         label=r'$\pm$ 1 std. dev.')

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic for {}'.format(self.models[0]))
        plt.legend(loc="lower right")
        fig = plt.gcf()
        fig.set_size_inches(6, 4.5)
        fig.savefig(self.result_dir +'/auc_plot_' + self.models[0] + '.png') #str(int(round(time.time() * 1000))) +

    def save_scores(self, scores, filepath):
        """ Save dictionary to .json file
        """
        if not os.path.isdir(os.path.dirname(filepath)): os.mkdir(os.path.dirname(filepath))
        with open(filepath, 'w') as jfile:
            json.dump(scores, jfile, indent=4, separators=(',', ': '))

    def save_cv_results(self, results, filepath):
        """ Save object to a csv
        """
        if not os.path.isdir(os.path.dirname(filepath)): os.mkdir(os.path.dirname(filepath))
        res = pd.DataFrame(results)
        res.to_csv(filepath)
        



#def regression():
#    label = "Pre-procedure percent diameter stenosis (%)"
#    models = ["dummy", "lasso", "rf", "gbm"] # "rf", "gbm"]
#    exp = Experiment("../Data/ACT_Data_Anonymized_v1.csv")
#    exp.load_feature_groups("../Data/ACT_feature_groups.json")
#
#    all_groups = ["ultrasound", "medical history", "demographic"]
#    all_scores = {}
#
#    # Run over all size combinations of the groups
#    for L in range(1, 4): #[3]:#
#        for groups in itertools.combinations(all_groups, L):
#            avg_scores, sd_scores = exp.predict_models_from_groups(label, groups, models, cv=5)
#            all_scores['-'.join(groups)] = [avg_scores, sd_scores]
#            exp.save_scores(all_scores, "../Results/ACT_regression_results_500.json")


    
    def save_and_plot_results(self, model_types, cv, pce_file = None, test = True, 
                              test_pce_file = None, train = True, 
                              title = 'Receiver operating characteristic on validation data'):
        self.result_dir = self.result_dir
        best_result_fns = []
        for rt, drs, fls in os.walk(self.result_dir):
            for fn in fls:
                if 'results.json' in  fn:
                    best_result_fns.append(fn)
        rts = {}
        for brfn in best_result_fns:
        #     brfn = best_result_fns[0]
            with open(os.path.join(self.result_dir, brfn)) as f:
                jres = json.load(f)
            key = list(jres)[0]
            rts[key] = jres[key]
            for key2 in list(rts[key]):#.keys():
                key3 = key2.split("-")[1:]
                key3 = "-".join(key3)
                rts[key][key3] = rts[key].pop(key2)
        if bool(rts):
            rts = pd.DataFrame(rts).transpose()
            rts.to_csv(os.path.join(self.result_dir, 'train_val_results.csv'))
#         ax = rts[['test-mean', 'train-mean']].plot.bar(rot=0, figsize = (10,8))
#         plt.savefig(os.path.join(self.result_dir, 'all_results.png'))
        plt.clf()
        rts = {}
        for model_type in model_types:
            tprs = []
            aucs = []
            mean_fpr = np.linspace(0, 1, 100)
    
            mean_auc = 0
            if model_type == 'PCE':
                if isinstance(pce_file, str):
                    pred_probs = pd.read_csv(pce_file)
                else:
                    pred_probs = pce_file
                fpr, tpr, thresholds = roc_curve(pred_probs.ascvdany5y, pred_probs.ascvd5yest)
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr,
                         label=r'ROC for PCE equation (AUC = %0.3f)' % (roc_auc),
                         lw=2, alpha=.5)
                mean_auc = roc_auc
            else:    
                for i in range(cv):
    
                    pred_probs = pd.read_csv(self.result_dir + '/predicted_probs_' + model_type + '_fold_' + str(i) + '.csv')
    
                    # Compute ROC curve and area under the curve
                    fpr, tpr, thresholds = roc_curve(pred_probs.y, pred_probs.est_prob)
                    tprs.append(interp(mean_fpr, fpr, tpr))
                    tprs[-1][0] = 0.0
                    roc_auc = auc(fpr, tpr)
                    aucs.append(roc_auc)
    
                mean_tpr = np.mean(tprs, axis=0)
                mean_tpr[-1] = 1.0
                mean_auc = auc(mean_fpr, mean_tpr)
                std_auc = np.std(aucs)
                plt.plot(mean_fpr, mean_tpr,
                         label=r'Mean ROC for %s (AUC = %0.3f $\pm$ %0.3f)' % (self.model_names[model_type], mean_auc, std_auc),
                         lw=2, alpha=.5)
    
            rts[model_type] = mean_auc
        rts = pd.DataFrame(rts, index = ['val']).transpose()
        rts.to_csv(os.path.join(self.result_dir, 'all_results_val.csv'))
    
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                 label='Chance', alpha=.5) 
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right")
        fig = plt.gcf()
        fig.set_size_inches(8, 6)
        fig.savefig(os.path.join(self.result_dir, 'auc_plot_all_methods.png'))
        
        if train:
            self.plot_train(model_types, cv, pce_file)
        if test:
            self.save_and_plot_test_results(model_types, cv, pce_file, test_pce_file, self.result_dir)
        
    def plot_train(self, model_types, cv, pce_file = None, 
                   title = 'Receiver operating characteristic on training data'):
        plt.clf()
        rts = {}
        for model_type in model_types:
            tprs = []
            aucs = []
            mean_fpr = np.linspace(0, 1, 100)
    
            mean_auc = 0
            if model_type == 'PCE':
                if isinstance(pce_file, str):
                    pred_probs = pd.read_csv(pce_file)
                else:
                    pred_probs = pce_file
                fpr, tpr, thresholds = roc_curve(pred_probs.ascvdany5y, pred_probs.ascvd5yest)
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr,
                         label=r'ROC for PCE equation (AUC = %0.3f)' % (roc_auc),
                         lw=2, alpha=.5)
                mean_auc = roc_auc
            else:    
                for i in range(cv):
    
                    pred_probs = pd.read_csv(self.result_dir + '/train_predicted_probs_' + model_type + '_fold_' + str(i) + '.csv')
    
                    # Compute ROC curve and area under the curve
                    fpr, tpr, thresholds = roc_curve(pred_probs.y, pred_probs.est_prob)
                    tprs.append(interp(mean_fpr, fpr, tpr))
                    tprs[-1][0] = 0.0
                    roc_auc = auc(fpr, tpr)
                    aucs.append(roc_auc)
    
                mean_tpr = np.mean(tprs, axis=0)
                mean_tpr[-1] = 1.0
                mean_auc = auc(mean_fpr, mean_tpr)
                std_auc = np.std(aucs)
                plt.plot(mean_fpr, mean_tpr,
                         label=r'Mean ROC for %s (AUC = %0.3f $\pm$ %0.3f)' % (self.model_names[model_type], mean_auc, std_auc),
                         lw=2, alpha=.5)
    
            rts[model_type] = mean_auc
        print('done')
        rts = pd.DataFrame(rts, index = ['train']).transpose()
        rts.to_csv(os.path.join(self.result_dir, 'all_results_train.csv'))
    
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                 label='Chance', alpha=.5) 
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right")
        fig = plt.gcf()
        fig.set_size_inches(8, 6)
        fig.savefig(os.path.join(self.result_dir, 'auc_plot_all_methods_train.png'))
            
                
    def save_and_plot_test_results(self, model_types, cv, baseline_str = 'baseline',
                                   test_baseline_prob_file = None, baseline_type = "probability", out_dir = None, 
                                   title = 'Receiver operating characteristic on test data',
                                  pr_title = "Precision recall curve on test data"):
        # plot test AUC
        
        if out_dir == None:
            out_dir = self.result_dir
        plt.clf()
        trts = {}
        for model_type in model_types:
            tprs = []
            aucs = []
    #         mean_fpr = np.linspace(0, 1, 100)
            
            roc_auc = 0
            sdc = 0
            if model_type == 'baseline':
                if isinstance(test_baseline_prob_file, str):
                    pred_probs = pd.read_csv(test_baseline_prob_file)
                else:
                    pred_probs = test_baseline_prob_file
                if baseline_type == 'probability':
                    fpr1, tpr1, thresholds = roc_curve(pred_probs.y, pred_probs.est_prob)
                    fpr = np.sort(np.concatenate([fpr1, np.append(fpr1[1:] - 0.0000001, 1)], axis = 0))
                    tpr = np.sort(np.concatenate([tpr1, tpr1]))
                    roc_auc = auc(fpr, tpr)
                    sdc, lc, _, hc = calc_auc_conf_interval(roc_auc, (pred_probs.y == 0).sum(), (pred_probs.y ==1).sum())
                    try:
                        line_color = self.leg_colors[model_type].get_color()
                        plt.plot(fpr, tpr,
                                 label=r'%s: %0.3f (%0.3f-%0.3f)' % (baseline_str, roc_auc, lc, hc),
                                 lw=2, alpha=.5, color = line_color)
                    except (KeyError, AttributeError):
                        plt.plot(fpr, tpr,
                                 label=r'%s: %0.3f (%0.3f-%0.3f)' % (baseline_str, roc_auc, lc, hc),
                                 lw=2, alpha=.5)
                elif baseline_type == 'binary':
                    metrics = calc_metrics(pred_probs.y, pred_probs.est_prob)
                    if metrics.shape[0] != 3:
                        raise ValueError("Baseline prediction has more than binary values")
                    try:
                        line_color = self.leg_colors[model_type].get_color()
                        plt.scatter(1-metrics['spec'].iloc[1], metrics['sens'].iloc[1],
                                 label=r'%s' % (baseline_str),
                                 lw=4, alpha=.7, color = line_color)
                    except (KeyError, AttributeError):
                        plt.scatter(1-metrics['spec'].iloc[1], metrics['sens'].iloc[1],
                                 label=r'%s' % (baseline_str),
                                 lw=4, alpha=.7)
                    
            else:    
    
                pred_probs = pd.read_csv(out_dir + '/test_predicted_probs_' + model_type + '.csv')
    
                # Compute ROC curve and area under the curve
                fpr1, tpr1, thresholds = roc_curve(pred_probs.y, pred_probs.est_prob)
                fpr = np.sort(np.concatenate([fpr1, np.append(fpr1[1:] - 0.0000001, 1)], axis = 0))
                tpr = np.sort(np.concatenate([tpr1, tpr1]))
                # tpr = interp(mean_fpr, fpr, tpr)
                # tpr[0] = 0.0
                roc_auc = auc(fpr, tpr)
                sdc, lc, _, hc = calc_auc_conf_interval(roc_auc, (pred_probs.y == 0).sum(), (pred_probs.y ==1).sum())

    
                # tpr[-1] = 1.0
                zsc = st.norm.ppf(1 - (1-0.95)/2.)
                try:
                    line_color = self.leg_colors[model_type].get_color()
                    plt.plot(fpr, tpr,
                             label=r'%s: %0.3f (%0.3f-%0.3f)' % (self.model_names[model_type], 
                                               roc_auc, 
                                               roc_auc - zsc * sdc, 
                                               roc_auc + zsc * sdc),
                             lw=2, alpha=.5, color = line_color)
                except (KeyError, AttributeError):
                    plt.plot(fpr, tpr,
                             label=r'%s: %0.3f (%0.3f-%0.3f)' % (self.model_names[model_type], 
                                               roc_auc, 
                                               roc_auc - zsc * sdc, 
                                               roc_auc + zsc * sdc),
                             lw=2, alpha=.5)
#            fpr_tpr = pd.DataFrame({'fpr':fpr1, 
#                                    'tpr':tpr1,
#                                    'thres':thresholds})
#             fpr_tpr.to_csv(os.path.join(out_dir, model_type + '_fpr_tpr_test.csv'))
            trts[model_type] = roc_auc
        trts = pd.DataFrame(trts, index = ['test']).transpose()
        trts.to_csv(os.path.join(out_dir, 'all_results_test.csv'))
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                 label='Chance', alpha=.5) 
        set_up_plot()
        plt.xlabel(FIG_XLABEL)
        plt.ylabel(FIG_YLABEL)
        plt.title(title)
        plt.legend(title="AUC (95% CI)", fontsize="small", loc="lower right")
        fig = plt.gcf()
        fig.set_size_inches(FIG_HT, FIG_WD)
        fig.savefig(os.path.join(out_dir, 'auc_plot_all_methods_test.png'), dpi = 900)
        
        
#    def save_and_plot_test_results(self, model_types, cv, pce_file = None, test_pce_file = None, out_dir = None, 
#                                   title = 'Receiver operating characteristic on test data'):
#        # plot test AUC
#        
#        if out_dir == None:
#            out_dir = self.result_dir
#        plt.clf()
#        trts = {}
#        for model_type in model_types:
#            tprs = []
#            aucs = []
#    #         mean_fpr = np.linspace(0, 1, 100)
#            
#            roc_auc = 0
#            if model_type == 'PCE':
#                if isinstance(test_pce_file, str):
#                    pred_probs = pd.read_csv(test_pce_file)
#                else:
#                    pred_probs = test_pce_file
#                fpr, tpr, thresholds = roc_curve(pred_probs.ascvdany5y, pred_probs.ascvd5yest)
#                roc_auc = auc(fpr, tpr)
#                plt.plot(fpr, tpr,
#                         label=r'ROC for PCE equation (AUC = %0.3f)' % (roc_auc),
#                         lw=2, alpha=.5)
#            else:    
#    
#                pred_probs = pd.read_csv(out_dir + '/test_predicted_probs_' + model_type + '.csv')
#    
#                # Compute ROC curve and area under the curve
#                fpr, tpr, thresholds = roc_curve(pred_probs.y, pred_probs.est_prob)
#                # tpr = interp(mean_fpr, fpr, tpr)
#                # tpr[0] = 0.0
#                roc_auc = auc(fpr, tpr)
#    
#                # tpr[-1] = 1.0
#                plt.plot(fpr, tpr,
#                         label=r'ROC for %s (AUC = %0.3f)' % (self.model_names[model_type], roc_auc),
#                         lw=2, alpha=.5)
#            metrics = calc_metrics(pred_probs.y, pred_probs.est_prob)
#            print(pred_probs.shape)
#            print(metrics.shape)
#            print(metrics.sens.max())
#            metrics.to_csv(os.path.join(out_dir, model_type + '_metrics_test.csv'), index = False)
#            trts[model_type] = roc_auc
#        trts = pd.DataFrame(trts, index = ['test']).transpose()
#        trts.to_csv(os.path.join(out_dir, 'all_results_test.csv'))
#        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
#                 label='Chance', alpha=.5) 
#        plt.xlim([-0.05, 1.05])
#        plt.ylim([-0.05, 1.05])
#        plt.xlabel('False Positive Rate')
#        plt.ylabel('True Positive Rate')
#        plt.title(title)
#        plt.legend(loc="lower right")
#        fig = plt.gcf()
#        fig.set_size_inches(8, 6)
#        fig.savefig(os.path.join(out_dir, 'auc_plot_all_methods_test.png'))
        
    def classification_ascvd(self, model_str, cv = 5, oversample_rate = 1, imputer = 'iterative', add_missing_flags = True):
        models = [model_str]
        label = "ascvdany5y"
        score_name = "AUC"
#        if model_str == None:
#            exp.label = label
#            return None
    
        all_scores = {}
    
        score_summary = self.predict_models_from_groups(0, models, cv=cv, score_name=score_name, mode='classification',
                                                        oversample_rate = oversample_rate, 
                                                       imputer = imputer, add_missing_flags = add_missing_flags)
#         all_scores[model_str] = score_summary
#         self.save_scores(all_scores, self.result_dir + '/' + model_str + "_results.json")

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
    plt.grid(True, 'major', color = '0.82', linewidth = 0.7)
    plt.grid(True, 'minor', color = '0.9', linestyle = '-', linewidth = 0.7)
    ax = plt.gca()
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    ax.set_axisbelow(True)
#    ax.patch.set_facecolor("0.85")
    
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
                
            elif k == 'cvd_bli':
                data = data[data['cvd_bl'] == 1]
                k = 'cvd_bl'
                
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
    
#RESULT_DIR = '../Results/test4'
if __name__ == '__main__':
    datafile = 'allvars.csv'
    alldata = pd.read_csv('../Data/cohort/' + datafile)
    pcefile = 'ascvd_est.csv'
    pcedata = pd.read_csv('../Data/cohort/' + pcefile)
    pce_train_est = pcedata[(pcedata.pce_cohort == 1) &
                            (pcedata.test_ind == 0)]
    pce_test_est = pcedata[(pcedata.pce_cohort == 1) &
                            (pcedata.test_ind == 1)]
    to_exclude = {
                            'pce_cohort': False,
                            'pce_invalid_vars': False,
                            'cvd_bl': True,
                            'antilpd': False,
                            'oldyoung': True} 
#                             'agebl': 80}
    test_ind_col = 'test_ind'
#     expt = Experiment('../Data/cohort/' + datafile, to_exclude, test_ind_col, drop = 'all')
    expt = Experiment(alldata, to_exclude, test_ind_col, drop = 'all')
#     expt = Experiment('../Data/cohort/train_set_missingvalues_200vars.csv')
#     expt = Experiment('../Data/cohort/old/cohort_113k_50vars_cleaned.csv')

    if not os.path.isdir(RESULT_DIR): os.mkdir(RESULT_DIR)

    classification_ascvd(expt, 'logreg')
#     classification_ascvd(expt, 'gbm')
#     classification_ascvd(expt, 'rf')
#     classification_ascvd(expt, 'lasso')
    
#     classification_ascvd(expt, 'dt')
    # classification_ascvd(expt, 'svm')
#     classification_ascvd(expt, 'knn')
    expt.predict_on_test(['logreg', 
                           'gbm', 'rf'
                          ])#, test_file = '../Data/cohort/test_' + datafile)
    to_exclude['pce_invalid_vars'] = True
    pce_train_est2, pce_test_est2 = split_cohort(pcedata, to_exclude, test_ind_col, drop = 'all')
    save_and_plot_results(['logreg',  
# #                            'lasso', 
                           'gbm', 'rf', 
                           'PCE'], cv = 5, pce_file = pce_train_est2, test = True,
                         test_pce_file = pce_test_est2)



