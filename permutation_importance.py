from __future__ import absolute_import
from typing import Tuple, List, Callable, Any

import numpy as np  # type: ignore
from sklearn.utils import check_random_state  # type: ignore
import matplotlib.pyplot as plt

import pickle
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

def iter_shuffled(X, columns_to_shuffle=None, pre_shuffle=True,
                  random_state=None):
    """
    Return an iterator of X matrices which have one or more columns shuffled.
    After each iteration yielded matrix is mutated inplace, so
    if you want to use multiple of them at the same time, make copies.

    ``columns_to_shuffle`` is a sequence of column numbers to shuffle.
    By default, all columns are shuffled once, i.e. columns_to_shuffle
    is ``range(X.shape[1])``.

    If ``pre_shuffle`` is True, a copy of ``X`` is shuffled once, and then
    result takes shuffled columns from this copy. If it is False,
    columns are shuffled on fly. ``pre_shuffle = True`` can be faster
    if there is a lot of columns, or if columns are used multiple times.
    """
    rng = check_random_state(random_state)

    if columns_to_shuffle is None:
        columns_to_shuffle = range(X.shape[1])

    if pre_shuffle:
        X_shuffled = X.copy()
        rng.shuffle(X_shuffled)

    X_res = X.copy()
    for columns in columns_to_shuffle:
        if pre_shuffle:
            X_res[:, columns] = X_shuffled[:, columns]
        else:
            rng.shuffle(X_res[:, columns])
        yield X_res
        X_res[:, columns] = X[:, columns]



def get_score_importances(
        score_func,  # type: Callable[[Any, Any], float]
        X,
        y,
        n_iter=5,  # type: int
        columns_to_shuffle=None,
        random_state=None
    ):
    # type: (...) -> Tuple[float, List[np.ndarray]]
    """
    Return ``(base_score, score_decreases)`` tuple with the base score and
    score decreases when a feature is not available.

    ``base_score`` is ``score_func(X, y)``; ``score_decreases``
    is a list of length ``n_iter`` with feature importance arrays
    (each array is of shape ``n_features``); feature importances are computed
    as score decrease when a feature is not available.

    ``n_iter`` iterations of the basic algorithm is done, each iteration
    starting from a different random seed.

    If you just want feature importances, you can take a mean of the result::

        import numpy as np
        from eli5.permutation_importance import get_score_importances

        base_score, score_decreases = get_score_importances(score_func, X, y)
        feature_importances = np.mean(score_decreases, axis=0)

    """
    rng = check_random_state(random_state)
    base_score = score_func(X, y)
    scores_decreases = []
    for i in range(n_iter):
        scores_shuffled = _get_scores_shufled(
            score_func, X, y, columns_to_shuffle=columns_to_shuffle,
            random_state=rng
        )
        scores_decreases.append(-scores_shuffled + base_score)
    return base_score, scores_decreases



def _get_scores_shufled(score_func, X, y, columns_to_shuffle=None,
                        random_state=None):
    Xs = iter_shuffled(X, columns_to_shuffle, random_state=random_state)
    return np.array([score_func(X_shuffled, y) for X_shuffled in Xs])


# -*- coding: utf-8 -*-
from functools import partial
from typing import List

import numpy as np  # type: ignore
from sklearn.model_selection import check_cv  # type: ignore
from sklearn.utils.metaestimators import if_delegate_has_method  # type: ignore
from sklearn.utils import check_array, check_random_state  # type: ignore
from sklearn.base import (  # type: ignore
    BaseEstimator,
    MetaEstimatorMixin,
    clone,
    is_classifier
)
from sklearn.metrics.scorer import check_scoring  # type: ignore

# from eli5.permutation_importance import get_score_importances
#from eli5.sklearn.utils import pandas_available
import pandas as pd   # type: ignore
pandas_available = True

CAVEATS_CV_NONE = """
Feature importances are computed on the same data as used for training, 
i.e. feature importances don't reflect importance of features for 
generalization.
"""

CAVEATS_CV = """
Feature importances are not computed for the final estimator; 
they are computed for a sequence of estimators trained and evaluated 
on train/test splits. So they tell you about importances of features 
for generalization, but not feature importances of a particular trained model.
"""

CAVEATS_PREFIT = """
If feature importances are computed on the same data as used for training, 
they don't reflect importance of features for generalization. Use a held-out
dataset if you want generalization feature importances.
"""


class PermutationImportance(BaseEstimator, MetaEstimatorMixin):
    """Meta-estimator which computes ``feature_importances_`` attribute
    based on permutation importance (also known as mean score decrease).

    :class:`~PermutationImportance` instance can be used instead of
    its wrapped estimator, as it exposes all estimator's common methods like
    ``predict``.

    There are 3 main modes of operation:

    1. cv="prefit" (pre-fit estimator is passed). You can call
       PermutationImportance.fit either with training data, or
       with a held-out dataset (in the latter case ``feature_importances_``
       would be importances of features for generalization). After the fitting
       ``feature_importances_`` attribute becomes available, but the estimator
       itself is not fit again. When cv="prefit",
       :meth:`~PermutationImportance.fit` must be called
       directly, and :class:`~PermutationImportance` cannot be used with
       ``cross_val_score``, ``GridSearchCV`` and similar utilities that clone
       the estimator.
    2. cv=None. In this case :meth:`~PermutationImportance.fit` method fits
       the estimator and computes feature importances on the same data, i.e.
       feature importances don't reflect importance of features for
       generalization.
    3. all other ``cv`` values. :meth:`~PermutationImportance.fit` method
       fits the estimator, but instead of computing feature importances for
       the concrete estimator which is fit, importances are computed for
       a sequence of estimators trained and evaluated on train/test splits
       according to ``cv``, and then averaged. This is more resource-intensive
       (estimators are fit multiple times), and importances are not computed
       for the final estimator, but ``feature_importances_`` show importances
       of features for generalization.

    Mode (1) is most useful for inspecting an existing estimator; modes
    (2) and (3) can be also used for feature selection, e.g. together with
    sklearn's SelectFromModel or RFE.

    Currently :class:`~PermutationImportance` works with dense data.

    Parameters
    ----------
    estimator : object
        The base estimator. This can be both a fitted
        (if ``prefit`` is set to True) or a non-fitted estimator.

    scoring : string, callable or None, default=None
        Scoring function to use for computing feature importances.
        A string with scoring name (see scikit-learn docs) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.
        If ``None``, the ``score`` method of the estimator is used.

    n_iter : int, default 5
        Number of random shuffle iterations. Decrease to improve speed,
        increase to get more precise estimates.

    random_state : integer or numpy.random.RandomState, optional
        random state

    cv : int, cross-validation generator, iterable or "prefit"
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

            - None, to disable cross-validation and compute feature importances
              on the same data as used for training.
            - integer, to specify the number of folds.
            - An object to be used as a cross-validation generator.
            - An iterable yielding train/test splits.
            - "prefit" string constant (default).

        If "prefit" is passed, it is assumed that ``estimator`` has been
        fitted already and all data is used for computing feature importances.

    refit : bool
        Whether to fit the estimator on the whole data if cross-validation
        is used (default is True).

    Attributes
    ----------
    feature_importances_ : array
        Feature importances, computed as mean decrease of the score when
        a feature is permuted (i.e. becomes noise).

    feature_importances_std_ : array
        Standard deviations of feature importances.

    results_ : list of arrays
        A list of score decreases for all experiments.

    scores_ : array of float
        A list of base scores for all experiments (with no features permuted).

    estimator_ : an estimator
        The base estimator from which the :class:`~PermutationImportance`
        instance  is built. This is stored only when a non-fitted estimator
        is passed to the :class:`~PermutationImportance`, i.e when ``cv`` is
        not "prefit".

    rng_ : numpy.random.RandomState
        random state
    """
    def __init__(self, estimator, scoring=None, n_iter=5, random_state=None,
                 cv='prefit', refit=True):
        # type: (...) -> None
        if isinstance(cv, str) and cv != "prefit":
            raise ValueError("Invalid cv value: {!r}".format(cv))
        self.refit = refit
        self.estimator = estimator
        self.scoring = scoring
        self.n_iter = n_iter
        self.random_state = random_state
        self.cv = cv
        self.rng_ = check_random_state(random_state)

    def _wrap_scorer(self, base_scorer, pd_columns):
        def pd_scorer(model, X, y):
            X = pd.DataFrame(X, columns=pd_columns)
            return base_scorer(model, X, y)
        return pd_scorer

    def fit(self, X, y, groups=None, columns_to_shuffle=None, **fit_params):
        # type: (...) -> PermutationImportance
        """Compute ``feature_importances_`` attribute and optionally
        fit the base estimator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.

        y : array-like, shape (n_samples,)
            The target values (integers that correspond to classes in
            classification, real numbers in regression).

        groups : array-like, with shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set.
            
        columns_to_shuffle : list of lists
            Each element represents the columns to be shuffled simultaneously

        **fit_params : Other estimator specific parameters

        Returns
        -------
        self : object
            Returns self.
        """
        self.scorer_ = check_scoring(self.estimator, scoring=self.scoring)

        if pandas_available and isinstance(X, pd.DataFrame):
            self.scorer_ = self._wrap_scorer(self.scorer_, X.columns)

        if self.cv != "prefit" and self.refit:
            self.estimator_ = clone(self.estimator)
            self.estimator_.fit(X, y, **fit_params)

        column_inds_to_shuffle = get_column_inds_from_names(X.columns, columns_to_shuffle)
        
        X = check_array(X)
        
        if self.cv not in (None, "prefit"):
            si = self._cv_scores_importances(X, y, groups=groups, 
                                             columns_to_shuffle=column_inds_to_shuffle, **fit_params)
        else:
            si = self._non_cv_scores_importances(X, y, columns_to_shuffle=column_inds_to_shuffle)
        scores, results = si
        self.scores_ = np.array(scores)
        self.results_ = results
        self.feature_importances_ = np.mean(results, axis=0)
        self.feature_importances_std_ = np.std(results, axis=0)
        return self


    def _cv_scores_importances(self, X, y, groups=None, 
                               columns_to_shuffle=None, **fit_params):
        assert self.cv is not None
        cv = check_cv(self.cv, y, is_classifier(self.estimator))
        feature_importances = []  # type: List
        base_scores = []  # type: List[float]
        for train, test in cv.split(X, y, groups):
            est = clone(self.estimator).fit(X[train], y[train], **fit_params)
            score_func = partial(self.scorer_, est)
            _base_score, _importances = self._get_score_importances(
                score_func, X[test], y[test], columns_to_shuffle)
            base_scores.extend([_base_score] * len(_importances))
            feature_importances.extend(_importances)
        return base_scores, feature_importances

    def _non_cv_scores_importances(self, X, y, columns_to_shuffle):
        score_func = partial(self.scorer_, self.wrapped_estimator_)
        base_score, importances = self._get_score_importances(score_func, X, y, columns_to_shuffle)
        return [base_score] * len(importances), importances

    def _get_score_importances(self, score_func, X, y, columns_to_shuffle):
        return get_score_importances(score_func, X, y, n_iter=self.n_iter,
                                     columns_to_shuffle=columns_to_shuffle,
                                     random_state=self.rng_)

    @property
    def caveats_(self):
        # type: () -> str
        if self.cv == 'prefit':
            return CAVEATS_PREFIT
        elif self.cv is None:
            return CAVEATS_CV_NONE
        return CAVEATS_CV

    # ============= Exposed methods of a wrapped estimator:

    @if_delegate_has_method(delegate='wrapped_estimator_')
    def score(self, X, y=None, *args, **kwargs):
        return self.wrapped_estimator_.score(X, y, *args, **kwargs)

    @if_delegate_has_method(delegate='wrapped_estimator_')
    def predict(self, X):
        return self.wrapped_estimator_.predict(X)

    @if_delegate_has_method(delegate='wrapped_estimator_')
    def predict_proba(self, X):
        return self.wrapped_estimator_.predict_proba(X)

    @if_delegate_has_method(delegate='wrapped_estimator_')
    def predict_log_proba(self, X):
        return self.wrapped_estimator_.predict_log_proba(X)

    @if_delegate_has_method(delegate='wrapped_estimator_')
    def decision_function(self, X):
        return self.wrapped_estimator_.decision_function(X)

    @property
    def wrapped_estimator_(self):
        if self.cv == "prefit" or not self.refit:
            return self.estimator
        return self.estimator_

    @property
    def _estimator_type(self):
        return self.estimator._estimator_type

    @property
    def classes_(self):
        return self.wrapped_estimator_.classes_

def get_column_inds_from_names(df_column_names, names_to_replace):
    replace_inds = []
    for n2r in names_to_replace:
        replace_inds.append([df_column_names.get_loc(c) for c in n2r])
    return(replace_inds)
 
    
def variable_importance_plot(feature_names, feat_importances, err=None, keep_top = None):
    """
    Purpose
    ----------
    Prints bar chart detailing variable importance for CART model
    NOTE: feature_space list was created because the bar chart
    was transposed and index would be in incorrect order.

    Parameters
    ----------
    * importance: Array returned from feature_importances_ for CART
                models organized by dataframe index

    Returns:
    ----------
    Returns variable importance plot in descending order
    """
#     index = np.arange(len(names_index))

#     importance_desc = sorted(importance)
#     feature_space = []

#     for i in range(indices.shape[0] - 1, -1, -1):
#         feature_space.append(names_index[indices[i]])
    fig, ax = plt.subplots(figsize=(7.5, 12))
    
    if err is None:
        err = np.zeros(len(feat_importances))
    feature_importances = pd.DataFrame([feat_importances, err], columns=feature_names)
    importances_df = feature_importances.sort_values(by=0, axis=1, ascending=True, inplace=False, kind='quicksort', na_position='last').T
    importances_df.columns = ['imps', 'err']
    if keep_top is not None:
        importances_df = importances_df.iloc[(-1*keep_top):]
#     ax.set_axis_bgcolor('#fafafa')
    ax.barh(importances_df.index,
            importances_df.imps,
            xerr=importances_df.err, 
            alpha = 0.9, 
            edgecolor = "black", 
            zorder=3, 
            color='lightblue'
             )
#              align="center",
#              color = '#875FDB')
#     plt.yticks(index,
#                feature_space)

#     plt.ylim(-1, 30)
#     plt.xlim(0, max(importance_desc) + 0.01)
    ax.set_ylabel('Feature')

    fig.subplots_adjust(left=0.3)
    fig.tight_layout()
    return ax, fig

#names_of_feats_all = []
#for feat_group in feature_space.columns:
#    for feat_dict in PATIENT_FEATURES_CONFIG:
#        if feat_dict['name'] == feat_group:
#            names_of_feats_all.append(feat_dict['formatted_name'])
#            break




#feat_list = [['agebl'],
#['female'],
#['race'],
#['hdlchol'],
#['totchol'],
#['systolic'],
#['t2d_history'],
#['bp_antihtn'],
#['cursmk_ever'],
#['ldlchol'],
#['diastolic'],
#['wt'],
#['ht'],
#['medhousincome'],
#['primarycarevsts'],
#['otherservicevsts'],
#['specialtycarevsts'],
#['total_medications'],
#['education5'],
#['education3'],
#['education4'],
#['education6'],
#['education1'],
#['education2'],
#['normal_tests'],
#['abnormal_tests'],
#['CCS_158'],
#['CCS_98'],
#['MONO_1'],
#['CCS_5'],
#['PSA_0'],
#['LYMPH_1'],
#['CCS_79'],
#['MED_4799'],
#['MED_3320'],
#['MED_1630'],
#['EOS_0'],
#['CCS_102'],
#['CCS_8'],
#['MED_3615'],
#['CCS_96'],
#['MED_9646'],
#['MED_6205'],
#['CALCIUM_0'],
#['MED_8672'],
#['MED_6410'],
#['EOS_1'],
#['CCS_33'],
#['BASO_0'],
#['CCS_63'],
#['GLU_1'],
#['CCS_59'],
#['GFR_1'],
#['CRP_1'],
#['CCS_51'],
#['CCS_204'],
#['CCS_95'],
#['CCS_653'],
#['CCS_64'],
#['CCS_244'],
#['CCS_97'],
#['MED_3999'],
#['U_ACR_1'],
#['MED_8625'],
#['K_0'],
#['MED_4630'],
#['U_PROT_1'],
#['MED_4155'],
#['BILI_0'],
#['CCS_83'],
#['BILI_1'],
#['CCS_2'],
#['MED_1220'],
#['MED_0310'],
#['MED_5940'],
#['CCS_11'],
#['CCS_660'],
#['MED_9066'],
#['CCS_104'],
#['MED_3720'],
#['MED_7710'],
#['MED_4240'],
#['CCS_115'],
#['AST_0'],
#['CCS_216'],
#['MED_3760'],
#['CCS_211'],
#['MED_0700'],
#['T4_1'],
#['FIBRINOGEN_1'],
#['BUN_1'],
#['MED_8230'],
#['CCS_152'],
#['CCS_49'],
#['CCS_50'],
#['CCS_651'],
#['CCS_199'],
#['MED_3610'],
#['CCS_99'],
#['MED_4920'],
#['MED_0199'],
#['MED_4650'],
#['Emphysema'],
#['MED_3940'],
#['MED_0230'],
#['MED_9940'],
#['MED_7813'],
#['U_MICALB24_1']]
#
#feat_names = ['agebl',
#'female',
#'race',
#'hdlchol',
#'totchol',
#'systolic',
#'t2d_history',
#'bp_antihtn',
#'cursmk_ever',
#'ldlchol',
#'diastolic',
#'wt',
#'ht',
#'medhousincome',
#'primarycarevsts',
#'otherservicevsts',
#'specialtycarevsts',
#'total_medications',
#'education5',
#'education3',
#'education4',
#'education6',
#'education1',
#'education2',
#'normal_tests',
#'abnormal_tests',
#'CCS_158',
#'CCS_98',
#'MONO_1',
#'CCS_5',
#'PSA_0',
#'LYMPH_1',
#'CCS_79',
#'MED_4799',
#'MED_3320',
#'MED_1630',
#'EOS_0',
#'CCS_102',
#'CCS_8',
#'MED_3615',
#'CCS_96',
#'MED_9646',
#'MED_6205',
#'CALCIUM_0',
#'MED_8672',
#'MED_6410',
#'EOS_1',
#'CCS_33',
#'BASO_0',
#'CCS_63',
#'GLU_1',
#'CCS_59',
#'GFR_1',
#'CRP_1',
#'CCS_51',
#'CCS_204',
#'CCS_95',
#'CCS_653',
#'CCS_64',
#'CCS_244',
#'CCS_97',
#'MED_3999',
#'U_ACR_1',
#'MED_8625',
#'K_0',
#'MED_4630',
#'U_PROT_1',
#'MED_4155',
#'BILI_0',
#'CCS_83',
#'BILI_1',
#'CCS_2',
#'MED_1220',
#'MED_0310',
#'MED_5940',
#'CCS_11',
#'CCS_660',
#'MED_9066',
#'CCS_104',
#'MED_3720',
#'MED_7710',
#'MED_4240',
#'CCS_115',
#'AST_0',
#'CCS_216',
#'MED_3760',
#'CCS_211',
#'MED_0700',
#'T4_1',
#'FIBRINOGEN_1',
#'BUN_1',
#'MED_8230',
#'CCS_152',
#'CCS_49',
#'CCS_50',
#'CCS_651',
#'CCS_199',
#'MED_3610',
#'CCS_99',
#'MED_4920',
#'MED_0199',
#'MED_4650',
#'Emphysema',
#'MED_3940',
#'MED_0230',
#'MED_9940',
#'MED_7813',
#'U_MICALB24_1']

#names_of_feats = []
#for feat_group in feat_list:
#    for feat_dict in PATIENT_FEATURES_CONFIG:
#        if feat_dict['name'] == feat_group[0]:
#            names_of_feats.append(feat_dict['formatted_name'])
#            break
#            
#names_of_feats[0] = 'Clinic Location'
#names_of_feats[1] = 'Clinic Urban/Rural'
#names_of_feats[2] = 'Ethnicity'
#names_of_feats[3] = 'Insurance Type'
#%%
result_dir = '../Results/allvars_pce_pts_0506/'
import os
if not os.path.isdir(os.path.dirname(result_dir)): os.mkdir(os.path.dirname(result_dir))
#result_dir = '../Results/allvars_pce_pts_0925/'
#best_model = 'gbm'
#from joblib import dump, load
#result_dir = '../Results/allvars_oldyoung_missing_0913/'
#best_model = 'gbm'
#model = load(result_dir + best_model + '_best_model.joblib')
run_date_str = '0507'

#feat_import_df = pd.read_csv(result_dir + best_model + "_feature_importances.csv")
##%%
#feat_names = [f for f in feat_import_df.feature if '_missing' not in f]
#feat_list = [[f] for f in feat_names]
##%%
#ax, fig = variable_importance_plot(feat_import_df.feature, feat_import_df.importance.values, keep_top = 30)
#ax.set_title('Feature importances for GBM: Impurity')
#ax.set_xlabel('Mean Decrease in Impurity');
#plt.tight_layout()
#plt.savefig(f'{result_dir}feature_importances_{best_model}_impurity_{run_date_str}.png', dpi = 500)


#%%

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from medical_ML import split_cohort
from datetime import datetime
test_ind_col  = 'test_ind'
label = 'ascvdany5y'
to_exclude = {
                            'pce_cohort': False,
                            'pce_invalid_vars': True,
                            'cvd_bl': True,
                            'antilpd': True,
                            'oldyoung': True}
datafile = 'allvars.csv'
ascvd_est = pd.read_csv('../Data/cohort/' + datafile)
#%%
train_est2, test_est2 = split_cohort(ascvd_est, to_exclude, test_ind_col, drop = 'all')
test_set_data = pd.get_dummies(test_est2, columns = [c for c in test_est2.columns if test_est2[c].dtype=='O'])
train_set_data = pd.get_dummies(train_est2, columns = [c for c in train_est2.columns if train_est2[c].dtype=='O'])
train_set_features = train_set_data[[f for f in train_set_data.columns if f != label]]
test_set_features = test_set_data[[f for f in test_set_data.columns if f != label]]
train_set_labels = train_est2[label]
test_set_labels = test_est2[label]
train_est2 = test_est2 = ascvd_est = None
imp = IterativeImputer(add_indicator=False,
                                                  estimator=None,
                                                  imputation_order='ascending',
                                                  initial_strategy='mean',
                                                  max_iter=50, max_value=None,
                                                  min_value=None,
                                                  missing_values=np.nan,
                                                  n_nearest_features=10,
                                                  random_state=None,
                                                  sample_posterior=False,
                                                  tol=0.001, verbose=0)
imp.fit(train_set_features)
train_set_imp_features = imp.transform(train_set_features)
train_set_imp_features = pd.DataFrame(train_set_imp_features, columns = train_set_features.columns)
test_set_imp_features = imp.transform(test_set_features)
test_set_imp_features = pd.DataFrame(test_set_imp_features, columns = test_set_features.columns)
train_set_features = test_set_features = None
#%%
#fl2 = [[fl[0]] for fl in feat_list if 'race' not in fl[0]]
#
#fl2.append(['race'])
#%%
#gbm = model.named_steps['predictor']
#gbm.n_features_ = test_set_features.shape[1]
#parms = gbm.get_params()
#model.named_steps['predictor'].n_features = test_set_features.shape[1]
parms = {'n_estimators': 300,
                         'learning_rate': 0.01,
                         'max_depth': 5,
                         'subsample': 0.35,
                         'max_features': 0.25}
print('training GBM')
now = datetime.now()
gbm2 = GradientBoostingClassifier(**parms)
print(train_set_imp_features.columns)
gbm2.fit(train_set_imp_features, train_set_labels)
difference = (datetime.now() - now).total_seconds()
print('done, total seconds:', difference)
#%%
ax, fig = variable_importance_plot(train_set_imp_features.columns, gbm2.feature_importances_, keep_top = 30)

ax.set_title('Feature importances for GBM model: Permutation Importance')
ax.set_xlabel('Mean Decrease in AUC')
plt.tight_layout()
plt.savefig(f'{result_dir}feat_imps_gini_{run_date_str}_100.png', dpi = 500)
#dump(gbm2)
#%%
print('calculating permutation importance')
now = datetime.now()
feat_names = [f for f in test_set_imp_features.columns if '_missing' not in f]
feat_list = [[f] for f in feat_names]
perm = PermutationImportance(gbm2, n_iter=5).fit(test_set_imp_features, test_set_labels, columns_to_shuffle = feat_list)
difference = (datetime.now() - now).total_seconds()
print('done, total seconds:', difference)
with open(f'{result_dir}permutation_feat_importances_all_{run_date_str}_5.pkl', "wb") as output_file:
    pickle.dump([perm.results_, perm.feature_importances_, perm.feature_importances_std_], output_file)
    #%%
ax, fig = variable_importance_plot(feat_names, perm.feature_importances_, err=perm.feature_importances_std_, keep_top = 30)

ax.set_title('Feature importances for GBM model: Permutation Importance')
ax.set_xlabel('Mean Decrease in AUC')
plt.tight_layout()
plt.savefig(f'{result_dir}feat_imps_permutation_{run_date_str}_100.png', dpi = 500)


## Create horizontal bars
#y_pos = np.arange(len(top_features_union))
#
#fig, ax = plt.subplots(figsize=(10,8))
#ax.xaxis.grid(True, zorder=0)
#width = 0.40
#
#offset_fix = np.zeros(len(top_features_union))
#offset_fix[top_var_imp_red == 0]= -width/2
##top_var_imp/np.max(top_var_imp) * 100 top_var_imp_red/np.max(top_var_imp_red) * 100 , width
#
#plt.barh(y_pos+width/2 + offset_fix, var_imp_df_top['relative importance']  , width, alpha = 0.5, edgecolor = "black", zorder=3, color='tab:grey')
#plt.barh(y_pos-width/2, var_imp_df_red_top['relative importance'] ,width, alpha = 0.5, edgecolor = "black", zorder=3, color='tab:blue')
#   
## Create names on the y-axis
#plt.yticks(y_pos, top_features)
#
#plt.xlabel('Relative Importance (%)')
#plt.xlim(0, 100)
#plt.legend([ 'All variables','Bedside variables'])
#plt.tight_layout()