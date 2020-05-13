#%%

from medical_ML import Experiment, split_cohort
import pandas as pd
import os
#%%
def train_val_test(RESULT_DIR, alldata, to_exclude, test_ind_col, models, ascvd_est, label, oversample_rate = 1,
                  imputer = 'iterative', add_missing_flags = True):
    print('\n\n' + 'STARTING EXPERIMENT FOR ' + RESULT_DIR + '\n\n')
    expt = Experiment(alldata, label = label, 
                      to_exclude = to_exclude, 
                      test_ind_col = test_ind_col, drop = 'all', 
                      result_dir = RESULT_DIR)

    for model in models:
        expt.classification_ascvd(model, oversample_rate = oversample_rate, imputer = imputer, add_missing_flags = add_missing_flags)
    
#    test_on_new_cohort(RESULT_DIR, expt, alldata, to_exclude = to_exclude,
#                       test_ind_col = test_ind_col,
#                       models = models, ascvd_est = ascvd_est)
    expt.predict_on_test(models, out_dir = RESULT_DIR)#, test_file = '../Data/cohort/test_' + datafile)
    to_exclude['pce_invalid_vars'] = True
    pce_train_est2, pce_test_est2 = split_cohort(ascvd_est, to_exclude, test_ind_col, drop = 'all')
    expt.save_and_plot_results(models + ['PCE'], 
                               cv = 5, pce_file = pce_train_est2, test = True,
                         test_pce_file = pce_test_est2)
    
    
def test_on_new_cohort(R2, expt, alldata, to_exclude, test_ind_col, models, 
                       ascvd_est):
    if not os.path.isdir(R2): os.mkdir(R2)
    _, test_data = split_cohort(alldata, to_exclude, test_ind_col, drop = 'all')
    expt.test_data = test_data
    expt.predict_on_test(models, test_file = None,
                        out_dir = R2)
    to_exclude['pce_invalid_vars'] = True
    ascvd_train_est2, ascvd_test_est2 = split_cohort(ascvd_est, to_exclude, test_ind_col, drop = 'all')
    expt.save_and_plot_test_results(models + ['PCE'], 
                               cv = 5, pce_file = ascvd_train_est2, 
                         test_pce_file = ascvd_test_est2,
                              out_dir = R2)
#%%    
if __name__ == '__main__':
    datafile = 'pcevars.csv'
    alldata = pd.read_csv('../Data/cohort/' + datafile)
    datafile2 = 'allvars.csv'
    alldata2 = pd.read_csv('../Data/cohort/' + datafile2)
    
    pcefile = 'ascvd_est.csv'
    ascvd_est = pd.read_csv('../Data/cohort/' + pcefile)
#    pce_train_est = ascvd_est[(ascvd_est.pce_cohort == 1) &
#                            (ascvd_est.test_ind == 0)]
#    pce_test_est = ascvd_est[(ascvd_est.pce_cohort == 1) &
#                            (ascvd_est.test_ind == 1)]
    test_ind_col = 'test_ind'
    label = 'ascvdany5y'
#     expt = Experiment('../Data/cohort/' + datafile, to_exclude, test_ind_col, drop = 'all')
#    import pdb; pdb.set_trace()
    
    models = [
#            'logreg'
#              ,
#              'gbm'
#              ,
#              'rf'
#              ,
              'lasso'
#              ,
#              'xgb'
              ]
    
    expts = {'test_oversampling_0913': {'pce_cohort': False,
                                'pce_invalid_vars': True,
                                'cvd_bl': True,
                                'antilpd': True,
                                'oldyoung': True} 
    #                             'agebl': 80}
                       }
    osr = {'/os2': 2,
           '/os5': 5000,
           '/os10':10}
    for res_dir in expts.keys():
        if not os.path.isdir('../Results/pcevars_' + res_dir): os.mkdir('../Results/pcevars_' + res_dir)
        for os_dir in osr.keys():
            imputer = 'iterative'
            add_missing_flags = True
            if expts[res_dir]['pce_invalid_vars']:
                imputer = None
                add_missing_flags = False
            train_val_test(RESULT_DIR = '../Results/pcevars_' + res_dir + os_dir, alldata = alldata, 
                           to_exclude = expts[res_dir],
                           test_ind_col = test_ind_col, models = models, ascvd_est = ascvd_est, label = label,
                          oversample_rate = osr[os_dir], imputer = imputer, add_missing_flags = add_missing_flags)
    for res_dir in expts.keys():
        if not os.path.isdir('../Results/allvars_' + res_dir): os.mkdir('../Results/allvars_' + res_dir)
        for os_dir in osr.keys():
            imputer = 'iterative'
            add_missing_flags = True
            if expts[res_dir]['pce_invalid_vars']:
                imputer = 'simple'
                add_missing_flags = False
            train_val_test(RESULT_DIR = '../Results/allvars_' + res_dir + os_dir, alldata = alldata2, 
                           to_exclude = expts[res_dir],
                           test_ind_col = test_ind_col, models = models, ascvd_est = ascvd_est, label = label,
                          oversample_rate = osr[os_dir], imputer = imputer, add_missing_flags = add_missing_flags)
    
#    train_val_test(RESULT_DIR = '../Results/test_new2', alldata = alldata, 
#                   to_exclude = {'pce_cohort': False,
#                            'pce_invalid_vars': False,
#                            'cvd_bl': True,
#                            'antilpd': True,
#                            'oldyoung': True} 
##                             'agebl': 80}
#                   ,
#                   test_ind_col = test_ind_col, models = models, ascvd_est = ascvd_est, label = label)
    
    