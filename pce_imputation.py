# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 10:50:46 2020

@author: Andrew
"""
#%%
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
import pandas as pd
import numpy as np
import os


#%%
def pce(df):
    df = df.assign(ascvd5yest2 = 0)
#    import pdb; pdb.set_trace()
    xbeta = (-29.799*np.log(df.agebl) + 4.884*np.log(df.agebl)*np.log(df.agebl) + 13.540*np.log(df.totchol) - 3.114*np.log(df.agebl)*np.log(df.totchol) 
    - 13.578*np.log(df.hdlchol) + 3.149*np.log(df.agebl)*np.log(df.hdlchol) + (2.019- 1.957)* df.bp_antihtn*np.log(df.systolic) + 
    1.957* np.log(df.systolic) + 7.574*df.cursmk_ever - 1.665*df.cursmk_ever*np.log(df.agebl) + 0.661*df.t2d_history )
    ascvd5yest= 1- 0.98898**np.exp(xbeta+29.18)
    inds_to_replace = (df.race != 'African_American') & (df.female == 1)
    df.loc[inds_to_replace, 'ascvd5yest2'] = ascvd5yest[inds_to_replace]
    
    xbeta = (12.344*np.log(df.agebl) + 11.853*np.log(df.totchol) - 2.664*np.log(df.agebl)*np.log(df.totchol)
    - 7.990*np.log(df.hdlchol) + 1.769*np.log(df.agebl)*np.log(df.hdlchol)+ (1.797- 1.764)*df.bp_antihtn*np.log(df.systolic) 
    + 1.764*np.log(df.systolic)+ 7.837*df.cursmk_ever - 1.795*df.cursmk_ever*np.log(df.agebl) + 0.658*df.t2d_history)
    ascvd5yest= 1- 0.96254**np.exp(xbeta-61.18)
    inds_to_replace = (df.race != 'African_American') & (df.female == 0)
    df.loc[inds_to_replace, 'ascvd5yest2'] = ascvd5yest[inds_to_replace]
        
    xbeta = (17.114*np.log(df.agebl) + 0.94*np.log(df.totchol) - 18.920*np.log(df.hdlchol) + 4.475*np.log(df.agebl)*np.log(df.hdlchol) 
    + 27.82* np.log(df.systolic) - 6.087*np.log(df.agebl)*np.log(df.systolic) + (29.291- 27.82)*df.bp_antihtn*np.log(df.systolic)
    + (- 6.432+ 6.087)*df.bp_antihtn*np.log(df.systolic)*np.log(df.agebl) + 0.691*df.cursmk_ever + 0.874*df.t2d_history )
    ascvd5yest= 1- 0.98194**np.exp(xbeta-86.61)
    inds_to_replace = (df.race == 'African_American') & (df.female == 1)
    df.loc[inds_to_replace, 'ascvd5yest2'] = ascvd5yest[inds_to_replace]
      
    xbeta = (2.469 *np.log(df.agebl) + 0.302*np.log(df.totchol)- 0.307*np.log(df.hdlchol) + 1.916*df.bp_antihtn+ 1.809*np.log(df.systolic) 
    - 1.809* df.bp_antihtn*np.log(df.systolic) + 0.549*df.cursmk_ever + 0.645*df.t2d_history)
    ascvd5yest= 1- 0.95726**np.exp(xbeta-19.54)
    inds_to_replace = (df.race == 'African_American') & (df.female == 0)
    df.loc[inds_to_replace, 'ascvd5yest2'] = ascvd5yest[inds_to_replace]
    
    return(df)


#pce_ests = pce(df2)
##%%
#print("number of outcomes which are different:")
#print(np.sum(pce_ests.ascvdany5y != res1.ascvdany5y))
#print("number of PCE estimates which are different:")
#print(np.sum(pce_ests.ascvd5yest2 != res1.ascvd5yest))
#
#outcomes = pd.DataFrame({'old_ests':res1.ascvd5yest, 
#                         'new_ests':pce_ests.ascvd5yest2})
#outcomes.to_csv('../Data/cohort/debug_PCE_code.csv')

out_dir = '../Results/pce_imputation/'
if not os.path.isdir(os.path.dirname(out_dir)): os.mkdir(os.path.dirname(out_dir))

datasets = ['pcevars', 'allvars']
#datasets = ['allvars']
imputers = [IterativeImputer(add_indicator=False,
                                                  estimator=None,
                                                  imputation_order='ascending',
                                                  initial_strategy='mean',
                                                  max_iter=50, max_value=None,
                                                  min_value=None,
                                                  missing_values=np.nan,
                                                  n_nearest_features=10,
                                                  random_state=None,
                                                  sample_posterior=False,
                                                  tol=0.001, verbose=0),
           SimpleImputer()]
imputer_names = ['simple', 
                 'iterative']
for dataset in datasets:
    for i, imp in enumerate(imputers):
        imp_name = imputer_names[i]
        
        df1 = pd.read_csv('../Data/cohort/' + dataset + '.csv')
#        df1 = df1.iloc[:1000, :]
        df1 = df1[(df1.agebl >= 40) & 
                  (df1.agebl <= 79) &
                  (df1.cvd_bl == 0) &
                  (df1.antilpd == 0)]
        
        print("df1:" , df1.shape)
        
        categorical_prefixes = [c for c in df1.columns if df1[c].dtype=='O']
        # print("String Columns:\n{}".format(string_columns))
        print("Found {} categorical variables.  Total of {} variables".format(len(categorical_prefixes), len(df1.columns)))
        df_race = df1[['race', 'test_ind', 'ascvdany5y']]
        print("dfrace:" , df_race.shape)
        df1 = df1.drop(['ascvdany5y'], axis = 1)
        print("df1:" , df1.shape)
        df1 = pd.get_dummies(df1, columns = categorical_prefixes)
        print("df1:" , df1.shape)
        df1 = df1.drop(['cvd_bl', 'antilpd', 'pce_invalid_vars', 'oldyoung'], axis = 1)
        print("df1:" , df1.shape)
        print("imputer name:" + imp_name)
        df1_cols = df1.columns
#        ti = df1.test_ind
#        df1 = df1.mask(np.random.random(df1.shape) < .1)
#        df1.test_ind = ti
        imp.fit(df1[df1.test_ind == 0])
        print("fitted on training data:" + imp_name)
        df1 = imp.transform(df1[df1.test_ind == 1])
        print("fitted on test data:" + imp_name)
        df1 = pd.DataFrame(df1, columns = df1_cols)
        print("df1:" , df1.shape)
        print("df_race_test:", df_race.loc[df_race.test_ind == 1, ['race', 'ascvdany5y']].shape)
        df_race_test = df_race.loc[df_race.test_ind == 1, ['race', 'ascvdany5y']]
        df1 = pd.DataFrame(
                    np.column_stack([df1, df_race_test]),
                    columns=df1.columns.append(df_race_test.columns)
        )
        print("df1:" , df1.shape)
        
        to_clip = ['hdlchol', 'totchol', 'systolic', 'diastolic']
        lower_clip = [20,     130,         90,        30]
        upper_clip = [100,    320,         200,       140]
        for i, var in enumerate(to_clip):
            if (dataset == 'pcevars') and (var == 'diastolic'):
                break
            df1.loc[df1[var] < lower_clip[i], var] = lower_clip[i]
            df1.loc[df1[var] > upper_clip[i], var] = upper_clip[i]
        
        print("df1:" , df1.shape)
        for col in df1.columns:
            if col != 'race':
                df1[col] = df1[col].astype('float64')
        df1 = pce(df1)
        print("df1:" , df1.shape)
        pce_est_plus_actual = df1[['ascvdany5y', 'ascvd5yest2']]
        pce_est_plus_actual.to_csv(out_dir + imp_name + '_' + dataset + '_PCE_estimates_clipped_vars2.csv', index = False)
#%%
#res = pd.read_csv('../Data/cohort/ascvd_est.csv')
##%%
##df1.female = np.round(df1.female)
#df2 = df1[df1.pce_cohort == 1]
#res1 = res[df1.pce_cohort == 1]
##%%
#imputer = IterativeImputer(add_indicator=False,
#                                                  estimator=None,
#                                                  imputation_order='ascending',
#                                                  initial_strategy='mean',
#                                                  max_iter=50, max_value=None,
#                                                  min_value=None,
#                                                  missing_values=np.nan,
#                                                  n_nearest_features=10,
#                                                  random_state=None,
#                                                  sample_posterior=False,
#                                                  tol=0.001, verbose=0)
#
##df2
#
#df3 = df2.mask(np.random.random(df2.shape) < .1)
#categorical_prefixes = [c for c in df3.columns if df3[c].dtype=='O']
## print("String Columns:\n{}".format(string_columns))
#print("Found {} categorical variables.  Total of {} variables".format(len(categorical_prefixes), len(df3.columns)))
#df4 = pd.get_dummies(df3, columns = categorical_prefixes)
#aa = imputer.fit(df4[df4.test_ind == 0])
#bb = imputer.transform(df4[df4.test_ind == 1])
##%%
#true_imp = df2[df2.test_ind == 1]
##%%
#bbc = pd.DataFrame(bb, columns = df4.columns)