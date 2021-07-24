""" Scipy wrappers for Medical machine learning models
"""
import os

import numpy as np
import json
import sys
#import itertools
#import subprocess
#import pickle 
#import time
from scipy import interp
import scipy.stats as st

sys.path.append(".")
from utils import split_cohort, calc_auc_conf_interval, load_models_and_parameters, load_models_and_parameters_default, calc_metrics, set_up_plot


#from sympy import Matrix

from joblib import dump, load
from sklearn.preprocessing import LabelBinarizer, PolynomialFeatures, StandardScaler, MinMaxScaler

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer, MissingIndicator

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import KFold, cross_val_score, cross_validate, GridSearchCV, RandomizedSearchCV, StratifiedKFold
#
#from sklearn import linear_model
#from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
#from sklearn.dummy import DummyRegressor
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error

#import xgboost as xgb
#from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
#from sklearn.dummy import DummyClassifier
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.linear_model import LogisticRegression, Lasso
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, auc, roc_auc_score, roc_curve, precision_recall_curve, average_precision_score#, PrecisionRecallDisplay

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd

FIG_HT = 5.5
FIG_WD = 5.5

#FIG_XLABEL = "False Positive Rate"
#FIG_YLABEL = "Recall"
FIG_XLABEL = "1 - Specificity"
FIG_YLABEL = "Sensitivity"

def test():
    print('test')
    
class Experiment():
    """
    Experimentation object for machine learning on medical datasets.
    Handles data loading, model training, fitting, and result formatting.
    Data is expected to be cleaned and only contain categorical/binary/continuous 
      columns (i.e., no date columns).
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
                            'lasso':'Lasso', # linear lasso
                            'logreg':'LR',
                            'baseline': 'Current',
                            'lasso2':'LRLasso', 
                           'dummy': 'dummy'} # logistic regression lasso
        if datafile is None:
            return
        data, test_data = split_cohort(datafile, to_exclude, test_ind_col, drop = drop)
        self.data = data
        self.test_data = test_data
        assert label in self.data.columns, "Label {} not in dataset".format(label)
        self.label = label
        print("Loaded {} samples".format(len(self.data)))
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


    def predict(self, est, cv, model_name, scoring=None, model='regression', verbose=False, imputer = 'simple', add_missing_flags = True):
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
                                ('scaler', MinMaxScaler()),
#                                 ('scaler', StandardScaler()),
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
        self.save_cv_results(est.cv_results_, filepath = self.result_dir + "/" + model_name + "_cv_results.csv")

        best_model = est_pipe.named_steps['predictor'].best_estimator_
        if hasattr(est_pipe.named_steps['predictor'], 'best_params_'):
            print("Best CV parameters:\n{}",format(est_pipe.named_steps['predictor'].best_params_))
            self.save_json(est_pipe.named_steps['predictor'].best_params_, filepath = self.result_dir + "/" + model_name + "_best_params.json")
        # if hasattr(est_pipe.named_steps['predictor'], 'feature_importances_'):
        if hasattr(best_model, 'feature_importances_'):
#            feature_importances = {self.features[i]: best_model.feature_importances_[i] for i in np.argsort(best_model.feature_importances_)[::-1]
#                                   if best_model.feature_importances_[i] != 0}
            feat_imp_order = np.argsort(best_model.feature_importances_)[::-1]
            feat_import_df = pd.DataFrame(data = {'feature': [self.feature_list[i] for i in feat_imp_order],
                                                                  'importance': [best_model.feature_importances_[i] for i in feat_imp_order]})
            feat_import_df.to_csv(self.result_dir + "/" + model_name + "_feature_importances.csv", index = False)
            print("Feature Importances:\n{}".format(feat_import_df))
        print("Best Model:\n{}".format(best_model))
        
        est_pipe.steps[-1] = ('predictor', best_model)      
        
        dump(est_pipe, os.path.join(self.result_dir, model_name + '_best_model.joblib'))
        # impute missing values in X for ROC plot
#        self.imp = est_pipe.named_steps['imputer']
#        X = self.imp.transform(X)

        self.plot_roc(est_pipe, X, y, cv, filepath = self.result_dir +'/auc_plot_' + model_name + '.png', model_name = model_name)
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

        models_and_parameters = load_models_and_parameters_default()

        # Restrict to the currently selected models
        models_and_parameters = {name: mp for name, mp in models_and_parameters.items() if name in models}
        
#        self.save_json(models_and_parameters, self.result_dir + '/models_and_hyperparameters_' + pd.Timestamp.now().strftime("%Y%m%d_%H%M%S") + '.csv')

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
            scores_across_outer_folds, fit_model = self.predict(gsv, cv=outer_cv, model_name = name, scoring=scorer, 
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

        valid_samples = np.isfinite(test_data.loc[:, self.label])
        if np.sum(~valid_samples) > 0:
            print("WARNING:  Dropping {} testsamples because of empty label {}".format(np.sum(~valid_samples), self.label))
        
        test_y = test_data.loc[valid_samples, self.label]
        
        valid_feat_series = pd.Series(self.valid_features)
        if not np.all(valid_feat_series.isin(test_data.columns)):
            print("WARNING:  Training columns not found in test data:")
            print(valid_feat_series[~valid_feat_series.isin(test_data.columns)])
            test_X = test_data.loc[valid_samples, valid_feat_series[valid_feat_series.isin(test_data.columns)]]
        else:
            test_X = test_data.loc[valid_samples, self.valid_features]
        
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



    def plot_roc(self, best_model, X, y, cv, filepath, model_name):
        '''plots ROC using cv cross validations
        '''

        plt.clf()
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)

        
        if model_name in ['lasso']:
            coefs_ = best_model.fit(X, y).named_steps['predictor'].coef_
            coef_dataframe = pd.DataFrame({'variable': self.feature_list, 'coeff': coefs_})
            coef_dataframe = coef_dataframe.iloc[(-coef_dataframe['coeff'].abs()).argsort()]
            coef_dataframe.to_csv('{}/{}_coefficients.csv'.format(self.result_dir, model_name), index = False)
            print(coef_dataframe)
        if model_name in ['logreg', 'lasso2']:
            coefs_ = best_model.fit(X, y).named_steps['predictor'].coef_
            coef_dataframe = pd.DataFrame({'variable': self.feature_list, 'coeff': coefs_[0]})
            coef_dataframe = coef_dataframe.iloc[(-coef_dataframe['coeff'].abs()).argsort()]
            coef_dataframe.to_csv('{}/{}_coefficients.csv'.format(self.result_dir, model_name), index = False)
            print(coef_dataframe)
            
        i = 0
        print('refitting on training data...')
        for train, test in cv.split(X, y):
            
            tr_probas_ = 0
            probas_ = 0
            fold_fit = best_model.fit(X.iloc[train].to_numpy(), y.iloc[train])
            if model_name == 'lasso':
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
            tr_result.to_csv(self.result_dir + '/train_predicted_probs_' + model_name + '_fold_' + str(i) + '.csv', index = False)
            cv_result = pd.DataFrame({'index': test, 
                                        'y': y.iloc[test], 
                                        'est_prob': probas_})
            cv_result.to_csv(self.result_dir + '/predicted_probs_' + model_name + '_fold_' + str(i) + '.csv', index = False)
            
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

        set_up_plot()
        plt.xlabel(FIG_XLABEL)
        plt.ylabel(FIG_YLABEL)
        plt.title('Receiver operating characteristic for {}'.format(model_name))
        plt.legend(loc="lower right")
        fig = plt.gcf()
        fig.set_size_inches(6, 4.5)
        fig.savefig(self.result_dir +'/auc_plot_' + model_name + '.png') #str(int(round(time.time() * 1000))) +

    def save_json(self, data, filepath):
        """ Save dictionary to .json file
        """
        if not os.path.isdir(os.path.dirname(filepath)): os.mkdir(os.path.dirname(filepath))
        with open(filepath, 'w') as jfile:
            json.dump(data, jfile, indent=4, separators=(',', ': '))

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
#            exp.save_json(all_scores, "../Results/ACT_regression_results_500.json")


    
    def save_and_plot_results(self, model_types, cv, train = True, 
                              baseline_prob_file = None, baseline_str = None, 
                              test = True, 
                              test_baseline_prob_file = None, 
                              title = 'Receiver operating characteristic on validation data',
                              tr_title = 'Receiver operating characteristic on training data',
                              test_title = 'Receiver operating characteristic on test data'):
        plt.close('all')
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
        
        perf_stats = pd.DataFrame({ 
                        'AUC': 0,
                        'AUC_95_low' : 0,
                        'AUC_95_hi' : 0
                       }, index = model_types)
        
        for model_type in model_types:
            tprs = []
            aucs = []
            mean_fpr = np.linspace(0, 1, 100)
            sdc = 0
            mean_auc = 0
            if model_type != 'baseline':
                pred_probs = None
                for i in range(cv):
    
                    pred_probs = pd.read_csv(self.result_dir + '/predicted_probs_' + model_type + '_fold_' + str(i) + '.csv')
    
                    # Compute ROC curve and area under the curve
                    fpr, tpr, thresholds = roc_curve(pred_probs.y, pred_probs.est_prob)
                    fpr = np.sort(np.concatenate([fpr, np.append(fpr[1:] - 0.0000001, 1)], axis = 0))
                    tpr = np.sort(np.concatenate([tpr, tpr]))
                    tprs.append(interp(mean_fpr, fpr, tpr))
                    tprs[-1][0] = 0.0
                    roc_auc = auc(fpr, tpr)
                    sdc, lc, _, hc = calc_auc_conf_interval(roc_auc, (pred_probs.y == 0).sum(), (pred_probs.y ==1).sum())
                    aucs.append(roc_auc)
    
                mean_tpr = np.mean(tprs, axis=0)
                mean_tpr[-1] = 1.0
                mean_auc = auc(mean_fpr, mean_tpr)
                zsc = st.norm.ppf(1 - (1-0.95)/2.)
                
                sdc, lc, _, hc = calc_auc_conf_interval(mean_auc, (pred_probs.y == 0).sum(), (pred_probs.y ==1).sum())
                perf_stats.loc[model_type, :] = [mean_auc, lc, hc]
                
        
        best_auc_model = perf_stats['AUC'].idxmax()
        
        
        
        for model_type in model_types:
            
#            if model_type == 'lasso2':
#
#                model = load(os.path.join(self.result_dir, model_type + '_best_model.joblib'))
#                best_model = model.named_steps['predictor']
#                coefs_ = best_model.coef_
#                coef_df1 = pd.read_csv('{}/{}_coefficients.csv'.format(self.result_dir, 'logreg'))
#                import pdb; pdb.set_trace()
#                coef_dataframe = pd.DataFrame({'variable': coef_df1.variable, 'coeff': coefs_[0]})
#                coef_dataframe = coef_dataframe.iloc[(-coef_dataframe['coeff'].abs()).argsort()]
#                coef_dataframe.to_csv('{}/{}_coefficients.csv'.format(self.result_dir, model_type), index = False)
            tprs = []
            aucs = []
            mean_fpr = np.linspace(0, 1, 100)
            sdc = 0
            mean_auc = 0
            zsc = st.norm.ppf(1 - (1-0.95)/2.)
            if model_type == best_auc_model:
                line_wid = 1
                alpha = 1
            else:
                line_wid = 0.5
                alpha = 0.7
            if model_type == 'baseline':
                if isinstance(baseline_prob_file, str):
                    pred_probs = pd.read_csv(baseline_prob_file)
                else:
                    pred_probs = baseline_prob_file
                fpr, tpr, thresholds = roc_curve(pred_probs.y, pred_probs.est_prob)
                fpr = np.sort(np.concatenate([fpr, np.append(fpr[1:] - 0.0000001, 1)], axis = 0))
                tpr = np.sort(np.concatenate([tpr, tpr]))
                roc_auc = auc(fpr, tpr)
                sdc, lc, _, hc = calc_auc_conf_interval(roc_auc, (pred_probs.y == 0).sum(), (pred_probs.y ==1).sum())
                plt.plot(fpr, tpr,
                         label=r'%s: %0.3f (%0.3f-%0.3f)' % (baseline_str, roc_auc, lc, hc),
                         lw=1, alpha=1)
                mean_auc = roc_auc
            else:    
                for i in range(cv):
    
                    pred_probs = pd.read_csv(self.result_dir + '/predicted_probs_' + model_type + '_fold_' + str(i) + '.csv')
    
                    # Compute ROC curve and area under the curve
                    fpr, tpr, thresholds = roc_curve(pred_probs.y, pred_probs.est_prob)
                    fpr = np.sort(np.concatenate([fpr, np.append(fpr[1:] - 0.0000001, 1)], axis = 0))
                    tpr = np.sort(np.concatenate([tpr, tpr]))
                    tprs.append(interp(mean_fpr, fpr, tpr))
                    tprs[-1][0] = 0.0
                    roc_auc = auc(fpr, tpr)
                    sdc, lc, _, hc = calc_auc_conf_interval(roc_auc, (pred_probs.y == 0).sum(), (pred_probs.y ==1).sum())
                    aucs.append(roc_auc)
    
                mean_tpr = np.mean(tprs, axis=0)
                mean_tpr[-1] = 1.0
                mean_auc = auc(mean_fpr, mean_tpr)
                zsc = st.norm.ppf(1 - (1-0.95)/2.)
                std_auc = np.std(aucs)
                plt.plot(mean_fpr, mean_tpr,
#                         label=r'Mean ROC for %s (AUC = %0.3f $\pm$ %0.3f)' % (self.model_names[model_type], mean_auc, std_auc),
                         label=r'%s: %0.3f (%0.3f-%0.3f)' % (self.model_names[model_type], 
                                           mean_auc, 
                                           mean_auc - zsc * sdc, 
                                           mean_auc + zsc * sdc),
                             lw=line_wid, alpha=alpha)
    
            rts[model_type] = mean_auc
        rts = pd.DataFrame(rts, index = ['val']).transpose()
        rts.to_csv(os.path.join(self.result_dir, 'all_results_val.csv'))
    
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                 label='Chance', alpha=.5) 
        set_up_plot()
        plt.xlabel(FIG_XLABEL)
        plt.ylabel(FIG_YLABEL)
        plt.title(title)
        plt.legend(title="AUC (95% CI)", fontsize="small", loc="lower right")
        fig = plt.gcf()
        fig.set_size_inches(FIG_WD, FIG_HT)
        
        ax = plt.gca()
        leg = ax.get_legend()
#        self.leg_colors = {handle.get_label(): handle for handle in leg.legendHandles}
        self.leg_colors = {model_types[i]: handle for i, handle in enumerate(leg.legendHandles[:-1])}
        fig.savefig(os.path.join(self.result_dir, 'auc_plot_all_methods.png'), dpi = 900)
        
        if train:
            self.plot_train(model_types, cv, baseline_prob_file, baseline_str = baseline_str, title = tr_title)
        if test:
            self.save_and_plot_test_results(model_types, cv, baseline_str, 
                                            test_baseline_prob_file, self.result_dir, title = test_title)
        
    def plot_train(self, model_types, cv, baseline_prob_file = None, baseline_str = None,
                   title = 'Receiver operating characteristic on training data'):
        plt.clf()
        rts = {}
        for model_type in model_types:
            tprs = []
            aucs = []
            mean_fpr = np.linspace(0, 1, 100)
    
            mean_auc = 0
            sdc = 0
            if model_type == 'baseline':
                if isinstance(baseline_prob_file, str):
                    pred_probs = pd.read_csv(baseline_prob_file)
                else:
                    pred_probs = baseline_prob_file
                fpr, tpr, thresholds = roc_curve(pred_probs.y, pred_probs.est_prob)
                fpr = np.sort(np.concatenate([fpr, np.append(fpr[1:] - 0.0000001, 1)], axis = 0))
                tpr = np.sort(np.concatenate([tpr, tpr]))
                roc_auc = auc(fpr, tpr)
                line_color = self.leg_colors[model_type].get_color()
                sdc, lc, _, hc = calc_auc_conf_interval(roc_auc, (pred_probs.y == 0).sum(), (pred_probs.y ==1).sum())
                plt.plot(fpr, tpr,
                         label=r'%s: %0.3f (%0.3f-%0.3f)' % (baseline_str, roc_auc, lc, hc),
                         lw=1, alpha=1, color = line_color)
                mean_auc = roc_auc
            else:    
                for i in range(cv):
    
                    pred_probs = pd.read_csv(self.result_dir + '/train_predicted_probs_' + model_type + '_fold_' + str(i) + '.csv')
    
                    # Compute ROC curve and area under the curve
                    fpr, tpr, thresholds = roc_curve(pred_probs.y, pred_probs.est_prob)
                    fpr = np.sort(np.concatenate([fpr, np.append(fpr[1:] - 0.0000001, 1)], axis = 0))
                    tpr = np.sort(np.concatenate([tpr, tpr]))
                    tprs.append(interp(mean_fpr, fpr, tpr))
                    tprs[-1][0] = 0.0
                    roc_auc = auc(fpr, tpr)
                    aucs.append(roc_auc)
                    sdc, lc, _, hc = calc_auc_conf_interval(roc_auc, (pred_probs.y == 0).sum(), (pred_probs.y ==1).sum())

                mean_tpr = np.mean(tprs, axis=0)
                mean_tpr[-1] = 1.0
                mean_auc = auc(mean_fpr, mean_tpr)
#                std_auc = np.std(aucs)
                line_color = self.leg_colors[model_type].get_color()
                zsc = st.norm.ppf(1 - (1-0.95)/2.)
                plt.plot(mean_fpr, mean_tpr,
#                         label=r'Mean ROC for %s (AUC = %0.3f $\pm$ %0.3f)' % (self.model_names[model_type], mean_auc, std_auc),
                         label=r'%s: %0.3f (%0.3f-%0.3f)' % (self.model_names[model_type], 
                                           mean_auc, 
                                           mean_auc - zsc * sdc, 
                                           mean_auc + zsc * sdc),
                         lw=2, alpha=.5, color = line_color)
    
            rts[model_type] = mean_auc
        print('done')
        rts = pd.DataFrame(rts, index = ['train']).transpose()
        rts.to_csv(os.path.join(self.result_dir, 'all_results_train.csv'))
    
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                 label='Chance', alpha=.5) 
        set_up_plot()
        plt.xlabel(FIG_XLABEL)
        plt.ylabel(FIG_YLABEL)
        plt.title(title)
        plt.legend(title="AUC (95% CI)", fontsize="small", loc="lower right")
        fig = plt.gcf()
        fig.set_size_inches( FIG_WD, FIG_HT)
        fig.savefig(os.path.join(self.result_dir, 'auc_plot_all_methods_train.png'), dpi = 900)
    
    def plot_baseline(self, pred_probs, baseline_type, baseline_str, model_type, ax, trts):
        if baseline_type == 'probability':
            fpr1, tpr1, thresholds = roc_curve(pred_probs.y, pred_probs.est_prob)
            fpr = np.sort(np.concatenate([fpr1, np.append(fpr1[1:] - 0.0000001, 1)], axis = 0))
            tpr = np.sort(np.concatenate([tpr1, tpr1]))
            roc_auc = auc(fpr, tpr)
            sdc, lc, _, hc = calc_auc_conf_interval(roc_auc, (pred_probs.y == 0).sum(), (pred_probs.y ==1).sum())
            try:
                line_color = self.leg_colors[model_type].get_color()
                ax.plot(fpr, tpr,
                         label=r'%s: %0.3f (%0.3f-%0.3f)' % (baseline_str, roc_auc, lc, hc),
                                 lw=1, alpha=1, color = line_color)
            except (KeyError, AttributeError):
                ax.plot(fpr, tpr,
                         label=r'%s: %0.3f (%0.3f-%0.3f)' % (baseline_str, roc_auc, lc, hc),
                         lw=1, alpha=1)
        elif baseline_type == 'binary':
            metrics = calc_metrics(pred_probs.y, pred_probs.est_prob)
            if metrics.shape[0] != 3:
                raise ValueError("Baseline prediction has more than binary values")
            try:
                line_color = self.leg_colors[model_type].get_color()
                ax.scatter(1-metrics['spec'].iloc[1], metrics['sens'].iloc[1],
                         label=r'%s' % (baseline_str),
                         lw=4, alpha=.7, color = line_color)
            except (KeyError, AttributeError):
                ax.scatter(1-metrics['spec'].iloc[1], metrics['sens'].iloc[1],
                         label=r'%s' % (baseline_str),
                         lw=4, alpha=.7) 
            trts[model_type] = {'auc': roc_auc, 'std': sdc}
                
                
    def save_and_plot_test_results(self, model_types, cv, baseline_str = 'baseline',
                                   test_baseline_prob_file = None, baseline_type = "probability", out_dir = None, 
                                   title = 'Receiver operating characteristic on test data',
                                  pr_title = "Precision recall curve on test data"):
        # plot test AUC
        plt.close('all')
        
        if out_dir == None:
            out_dir = self.result_dir
        plt.clf()
        fig, ax = plt.subplots()
        trts = {}
        
        
        perf_stats = pd.DataFrame({ 
                        'AUC': 0,
                        'AUC_95_low' : 0,
                        'AUC_95_hi' : 0,
                       'mean_avg_precision': 0
                       }, index = model_types)
        
        for model_type in model_types:
            if model_type != 'baseline':
                pred_probs = pd.read_csv(out_dir + '/test_predicted_probs_' + model_type + '.csv')
                fpr1, tpr1, thresholds = roc_curve(pred_probs.y, pred_probs.est_prob)
                fpr = np.sort(np.concatenate([fpr1, np.append(fpr1[1:] - 0.0000001, 1)], axis = 0))
                tpr = np.sort(np.concatenate([tpr1, tpr1]))
                # tpr = interp(mean_fpr, fpr, tpr)
                # tpr[0] = 0.0
                roc_auc = auc(fpr, tpr)
                mean_avg_precision = average_precision_score(pred_probs.y, pred_probs.est_prob)
                sdc, lc, _, hc = calc_auc_conf_interval(roc_auc, (pred_probs.y == 0).sum(), (pred_probs.y ==1).sum())
                perf_stats.loc[model_type, :] = [roc_auc, lc, hc, mean_avg_precision]
                
        
        best_map_model = perf_stats['mean_avg_precision'].idxmax()
        best_auc_model = perf_stats['AUC'].idxmax()
        
        for model_type in model_types:
            tprs = []
            aucs = []
    #         mean_fpr = np.linspace(0, 1, 100)
            
    
            roc_auc = 0
            sdc = 0
            if 'baseline' in model_type:
                if isinstance(test_baseline_prob_file, str):
                    pred_probs = pd.read_csv(test_baseline_prob_file)
                    self.plot_baseline(pred_probs, baseline_type, baseline_str, model_type, ax, trts)
                    metrics = calc_metrics(pred_probs.y, pred_probs.est_prob)
                    metrics.to_csv(os.path.join(out_dir, model_type + '_metrics_test.csv'), index = False)
                elif isinstance(test_baseline_prob_file, list):
                    for i, tbpf in enumerate(test_baseline_prob_file):
                        pred_probs = tbpf
                        self.plot_baseline(pred_probs, baseline_type[i], baseline_str[i], model_type + str(i), ax, trts)
                        
                        metrics = calc_metrics(pred_probs.y, pred_probs.est_prob)
                        metrics.to_csv(os.path.join(out_dir, model_type + str(i) + '_metrics_test.csv'), index = False)
                        trts[model_type + str(i)] = roc_auc
                else:
                    pred_probs = test_baseline_prob_file
                    self.plot_baseline(pred_probs, baseline_type, baseline_str, model_type, ax, trts)
                    metrics = calc_metrics(pred_probs.y, pred_probs.est_prob)
                    metrics.to_csv(os.path.join(out_dir, model_type  + '_metrics_test.csv'), index = False)
                    
                
                    
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
                if model_type == best_auc_model:
                    line_wid = 1
                    alpha = 1
                else:
                    line_wid = 0.5
                    alpha = 0.7
                try:
                    print(line_wid, alpha)
                    line_color = self.leg_colors[model_type].get_color()
                    ax.plot(fpr, tpr,
                             label=r'%s: %0.3f (%0.3f-%0.3f)' % (self.model_names[model_type], 
                                               roc_auc, 
                                               roc_auc - zsc * sdc, 
                                               roc_auc + zsc * sdc),
                             lw=line_wid, alpha=alpha, color = line_color)
                except (KeyError, AttributeError):
                    ax.plot(fpr, tpr,
                             label=r'%s: %0.3f (%0.3f-%0.3f)' % (self.model_names[model_type], 
                                               roc_auc, 
                                               roc_auc - zsc * sdc, 
                                               roc_auc + zsc * sdc),
                             lw=line_wid, alpha=alpha)
#            fpr_tpr = pd.DataFrame({'fpr':fpr1, 
#                                    'tpr':tpr1,
#                                    'thres':thresholds})
#             fpr_tpr.to_csv(os.path.join(out_dir, model_type + '_fpr_tpr_test.csv'))
            trts[model_type] = roc_auc
            metrics = calc_metrics(pred_probs.y, pred_probs.est_prob)
            metrics.to_csv(os.path.join(out_dir, model_type + '_metrics_test.csv'), index = False)
        trts = pd.DataFrame(trts, index = ['test']).transpose()
        trts.to_csv(os.path.join(out_dir, 'all_results_test.csv'))
        plt.sca(ax)
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                 label='Chance', alpha=.5) 
        set_up_plot()
        plt.xlabel(FIG_XLABEL)
        plt.ylabel(FIG_YLABEL)
        plt.title(title, fontsize = 10)
        plt.legend(title="AUC (95% CI)", fontsize="small", loc="lower right")
        fig = plt.gcf()
        fig.set_size_inches( FIG_WD, FIG_HT)
        fig.savefig(os.path.join(out_dir, 'auc_plot_all_methods_test.jpg'), dpi = 400)
        fig.savefig(os.path.join(out_dir, 'auc_plot_all_methods_test.svg'))
        
        
        # plot test PRC


#        plt.clf()
#        ax = plt.gca()
#        trts = {}
#        for model_type in model_types:
#            tprs = []
#            aucs = []
#    #         mean_fpr = np.linspace(0, 1, 100)
#            
#            roc_auc = 0
#            sdc = 0
#            if model_type == 'baseline':
#                if isinstance(test_baseline_prob_file, str):
#                    pred_probs = pd.read_csv(test_baseline_prob_file)
#                else:
#                    pred_probs = test_baseline_prob_file
#                    
#                if baseline_type == 'probability':
#                    fpr, tpr, thresholds = precision_recall_curve(pred_probs.y, pred_probs.est_prob)
#    #                 fpr = np.sort(np.concatenate([fpr1, np.append(fpr1[1:] - 0.0000001, 1)], axis = 0))
#    #                 tpr = np.sort(np.concatenate([tpr1, tpr1]))
#                    roc_auc = average_precision_score(pred_probs.y, pred_probs.est_prob)
#    #                 sdc, lc, _, hc = calc_auc_conf_interval(roc_auc, (pred_probs.y == 0).sum(), (pred_probs.y ==1).sum())
#                    try:
#                        line_color = self.leg_colors[model_type].get_color()
#                        plt.plot(fpr, tpr,
#                                 label=r'%s: %0.3f' % (baseline_str, roc_auc),
#                                 lw=2, alpha=.5, color = line_color)
#                    except (KeyError, AttributeError):
#                        plt.plot(fpr, tpr,
#                                 label=r'%s: %0.3f' % (baseline_str, roc_auc),
#                                 lw=2, alpha=.5)
#                
#                elif baseline_type == 'binary':
#                    metrics = calc_metrics(pred_probs.y, pred_probs.est_prob)
#                    if metrics.shape[0] != 3:
#                        raise ValueError("Baseline prediction has more than binary values")
#                    try:
#                        line_color = self.leg_colors[model_type].get_color()
#                        plt.scatter(metrics['sens'].iloc[1], metrics['PPV'].iloc[1],
#                                 label=r'%s' % (baseline_str),
#                                 lw=4, alpha=.7, color = line_color)
#                    except (KeyError, AttributeError):
#                        plt.scatter(metrics['sens'].iloc[1], metrics['PPV'].iloc[1],
#                                 label=r'%s' % (baseline_str),
#                                 lw=4, alpha=.7)
#                    
#            else:    
#    
#                pred_probs = pd.read_csv(out_dir + '/test_predicted_probs_' + model_type + '.csv')
#    
#                # Compute ROC curve and area under the curve
#                fpr, tpr, thresholds = precision_recall_curve(pred_probs.y, pred_probs.est_prob)
#                roc_auc = average_precision_score(pred_probs.y, pred_probs.est_prob)
#
#    
#                # tpr[-1] = 1.0
#                zsc = st.norm.ppf(1 - (1-0.95)/2.)
#                try:
#                    line_color = self.leg_colors[model_type].get_color()
#                    PrecisionRecallDisplay(fpr, tpr, roc_auc, self.model_names[model_type])\
#                    .plot(ax = ax, label=r'%s: %0.3f' % (self.model_names[model_type], 
#                                               roc_auc),
#                             lw=2, alpha=.5, color = line_color)
##                     plt.plot(fpr, tpr,
##                              label=r'%s: %0.3f' % (self.model_names[model_type], roc_auc),
##                              lw=2, alpha=.5, color = line_color)
#                except (KeyError, AttributeError) as e:
#                    PrecisionRecallDisplay(fpr, tpr, roc_auc, self.model_names[model_type])\
#                    .plot(ax = ax, label=r'%s: %0.3f' % (self.model_names[model_type], roc_auc),
#                             lw=2, alpha=.5)
##                     plt.plot(fpr, tpr,
##                              label=r'%s: %0.3f' % (self.model_names[model_type], 
##                                                roc_auc),
##                              lw=2, alpha=.5)
##            fpr_tpr = pd.DataFrame({'prec':fpr[1:], 
##                                    'rec':tpr[1:],
##                                    'thres':thresholds})
##             fpr_tpr.to_csv(os.path.join(out_dir, model_type + '_prec_rec_test.csv'))
#            metrics = calc_metrics(pred_probs.y, pred_probs.est_prob)
#            print(pred_probs.shape)
#            print(metrics.shape)
#            print(metrics.sens.max())
#            metrics.to_csv(os.path.join(out_dir, model_type + '_metrics_test.csv'), index = False)
#            trts[model_type] = roc_auc
#        trts = pd.DataFrame(trts, index = ['test']).transpose()
#        trts.to_csv(os.path.join(out_dir, 'all_results_prc_test.csv'))
##         plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
##                  label='Chance', alpha=.5) 
#        set_up_plot()
#        plt.xlabel('Recall')
#        plt.ylabel("Precision")
#        plt.title(pr_title)
#        plt.legend(title="Mean Average Precision", fontsize="small", loc="lower left")
#        fig = plt.gcf()
#        fig.set_size_inches(FIG_WD , FIG_HT)
#        fig.savefig(os.path.join(out_dir, 'prc3_plot_all_methods_test.png'), dpi = 900)



