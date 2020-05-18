# -*- coding: utf-8 -*-
"""
Created on Mon May 11 02:31:35 2020

@author: Andrew
"""
#%%
from medical_ML import calc_auc_conf_interval
import pandas as pd

results_path = '../Results/allvars_pce_nhwblack_0913/pce_asian'

ress = pd.read_csv(results_path + '/all_results_test.csv')
ress.columns = ['model', 'AUC']

pred_probs = pd.read_csv(results_path + '/test_predicted_probs_' + ress.model[0] + '.csv')

new_res = []
for i in range(ress.shape[0]):
    
    sdc, lc, _, hc = calc_auc_conf_interval(ress.AUC.iloc[i], (pred_probs.y == 0).sum(), (pred_probs.y ==1).sum())
    new_res.append([ress.AUC.iloc[i], sdc, lc, hc, '{:.3f} ({:.3f}-{:.3f})'.format(ress.AUC.iloc[i], lc, hc)])
resdf = pd.DataFrame(new_res)
resdf.columns = ['AUC', 'sd', 'lo_c', 'hi_c', 'AUC_with_conf_ints']
resdf.index = ress.model
resdf.to_csv(results_path + '/all_results_test_formatted.csv')