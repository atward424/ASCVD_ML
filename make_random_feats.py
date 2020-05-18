# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 09:29:07 2020

@author: Andrew
"""
#%%
import pandas as pd

asdf = pd.read_csv('../Data/cohort/ascvd_est.csv')

#%%
var_names = pd.read_csv("full_patients_variable_missingness.csv")
var_names.columns = ['var_nm', 'missing']

#%%

full_pts = asdf[~( (asdf.cvd_bl==1) | (asdf.antilpd==1))]

#%%

vars_to_add = var_names.var_nm[9:].tolist() + ['female'] 

import numpy as np

newvars = np.random.rand(asdf.shape[0], len(vars_to_add))

#%%

newvar_pd = pd.DataFrame(newvars)
newvar_pd.columns = vars_to_add
#%%
new_df = pd.concat([asdf, newvar_pd], axis = 1)


new_df = new_df.drop(['old', 'young',  'ascvd5yest'], axis = 1)
#%%

new_df.to_csv('../Data/cohort/ascvd_est_random_feats.csv')