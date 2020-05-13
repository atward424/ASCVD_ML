#%%
import os
import pandas as pd
import re
import numpy as np

from medical_ML import calc_auc_conf_interval
from sklearn.metrics import roc_curve, auc
result_dir = '../Results/'

match_str = '0829$'#|0816$'
match_str = '0925$'#|0913$'
match_str = '0913$'

folders = []
for folder in os.listdir(result_dir):
    if re.search(match_str, folder):
        folders.append(folder)
        
res = pd.DataFrame({'folder': folders})
res = res.set_index('folder')

for folder in folders:
    res_file = result_dir + folder + '/all_results_val.csv'
    res_train = result_dir + folder + '/all_results_train.csv'
    res_test = result_dir + folder + '/all_results_test.csv'
    if os.path.exists(res_train):
        res1 = pd.read_csv(res_train, index_col = 0)
        for model_type in res1.index:
            if model_type + '_train' not in res.columns:
                res.insert(len(res.columns), model_type + '_train', np.NaN)
            res.loc[folder, model_type + '_train'] = res1.loc[model_type, 'train']
    if os.path.exists(res_file):
        res1 = pd.read_csv(res_file, index_col = 0)
        for model_type in res1.index:
            if model_type + '_val' not in res.columns:
                res.insert(len(res.columns), model_type + '_val', np.NaN)
            res.loc[folder, model_type + '_val'] = res1.loc[model_type, 'val']
    if os.path.exists(res_test):
        res1 = pd.read_csv(res_test, index_col = 0)
        for model_type in res1.index:
            if model_type + '_test' not in res.columns:
                res.insert(len(res.columns), model_type + '_test', np.NaN)
            res.loc[folder, model_type + '_test'] = res1.loc[model_type, 'test']
#%%
match_str = '0829$'#|0816$'
match_str = '0925$'#|0913$'
match_str = '0925$'

folders = []
for folder in os.listdir(result_dir):
    if re.search(match_str, folder):
        folders.append(folder)
        
res = pd.DataFrame({'folder': folders})
res = res.set_index('folder')

for folder in folders:
    print(folder)
    res_file = result_dir + folder + '/all_results_val.csv'
#    res_train = result_dir + folder + '/all_results_train.csv'
#    res_test = result_dir + folder + '/all_results_test.csv'
#    if os.path.exists(res_train):
#        res1 = pd.read_csv(res_train, index_col = 0)
#        for model_type in res1.index:
#            if model_type + '_train' not in res.columns:
#                res.insert(len(res.columns), model_type + '_train', np.NaN)
#            res.loc[folder, model_type + '_train'] = res1.loc[model_type, 'train']
    if os.path.exists(res_file):
        res1 = pd.read_csv(res_file, index_col = 0)
#        for model_type in res1.index:
#            if model_type + '_val' not in res.columns:
#                res.insert(len(res.columns), model_type + '_val', np.NaN)
#            res.loc[folder, model_type + '_val'] = res1.loc[model_type, 'val']
#    if os.path.exists(res_test):
#        res1 = pd.read_csv(res_test, index_col = 0)
        for model_type in res1.index:
            if model_type + '_test' not in res.columns:
                res.insert(len(res.columns), model_type + '_test', np.NaN)
            if model_type == 'PCE':
                continue
            print(model_type)
            if model_type + '_test' not in res.columns:
                res.insert(len(res.columns), model_type + '_test', np.NaN)
            pred_probs = pd.read_csv(result_dir + folder + '/test_predicted_probs_' + model_type + '.csv')
            
            fpr1, tpr1, thresholds = roc_curve(pred_probs.y, pred_probs.est_prob)
            fpr = np.sort(np.concatenate([fpr1, np.append(fpr1[1:] - 0.0000001, 1)], axis = 0))
            tpr = np.sort(np.concatenate([tpr1, tpr1]))
            roc_auc = auc(fpr, tpr)
            sdc, lc, _, hc = calc_auc_conf_interval(roc_auc, (pred_probs.y == 0).sum(), (pred_probs.y ==1).sum())
            res.loc[folder, model_type + '_test'] = roc_auc
#%%
#order = pd.Series(res['folder']).str.split(pat = '_', expand=True)
order = pd.Series(res.index).str.split(pat = '_', expand=True)
order = order.reset_index()
order.index = res.index
#res = res.reset_index()
order = ['_pce_nhwblack',
         '_pce_nhwblack_missing',
        '_pce_asian',
        '_pce_missing_asian',
        '_pce_hispanic',
        '_pce_missing_hispanic',
        '_pce_pts',
        '_pce_missing',
        '_pce_oldyoung',
        '_pce_missing_oldyoung',
        '_over40',
        '_over40_missing',
        '_over80',
        '_over80_missing', 
        '_oldyoung_missing']
order = [o + '_0925' for o in order]
var_prefs = ['pcevars', 'morevars', 'allvars']
#ress = pd.concat((order[[0,1]], res ), axis = 1)
order2 = [vp + o  for o in order for vp in var_prefs]
ress = res.loc[order2]
#ress = ress.sort_values(by = [1,0], ascending = [True, False])
ress.to_csv(result_dir + 'results_0511_from0925_all.csv', index = True)

#%%
%pylab inline
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
for ff in ress.folder:
    res_file = result_dir + ff + '/auc_plot_all_methods.png'
#    res_test = result_dir + ff + '/pce_pts/auc_plot_all_methods_test.png'
#    if os.path.exists(res_file):
#        print(res_file)
#        plt.figure(figsize = (10,8))
#        img=mpimg.imread(res_file)
#        imgplot = plt.imshow(img, aspect='auto')
#        plt.show()
    if os.path.exists(res_test):
        print(res_test)
        plt.figure(figsize = (10,8))
        img=mpimg.imread(res_test)
        imgplot = plt.imshow(img, aspect='auto')
        plt.show()
#%%
for ff in res.index:
    print(ff)
#    cvs = pd.read_csv(result_dir + ff + '/gbm_cv_results.csv')
#    cvs = cvs.sort_values(by = 'rank_test_score')
#    print(cvs[['param_max_depth',	'param_n_estimators',	'param_subsample']].head(2))
    cvs = pd.read_csv(result_dir + ff + '/rf_cv_results.csv')
    cvs = cvs.sort_values(by = 'rank_test_score')
    print(cvs[['param_max_depth',	'param_n_estimators',	'param_subsample']].head(2))