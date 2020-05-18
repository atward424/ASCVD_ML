#%%

from medical_ML import Experiment, split_cohort
import pandas as pd
import os

#def test(RESULT_DIR, alldata, to_exclude, test_ind_col, models, ascvd_est, label, oversample_rate = 1,
#                  imputer = 'iterative', add_missing_flags = True):

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
    
    test_others = {'pce_nhwblack':{
                            'pce_cohort': False,
                            'pce_invalid_vars': True,
                            'race': ['Non-Hispanic_white', 'African_American'],
                            'cvd_bl': True,
                            'antilpd': True,
                            'oldyoung': True},
                  'pce_hispanic':{
                            'pce_cohort': False,
                            'pce_invalid_vars': True,
                            'race': ['Hispanic'],
                            'cvd_bl': True,
                            'antilpd': True,
                            'oldyoung': True},
                  'pce_asian':{
                            'pce_cohort': False,
                            'pce_invalid_vars': True,
                            'race': ['Asian'],
                            'cvd_bl': True,
                            'antilpd': True,
                            'oldyoung': True},
                  'pce_pts':{
                            'pce_cohort': False,
                            'pce_invalid_vars': True,
                            'cvd_bl': True,
                            'antilpd': True,
                            'oldyoung': True},
                   'over80':{
                            'pce_cohort': True,
                            'pce_invalid_vars': True,
                            'cvd_bl': True,
                            'antilpd': True,
                            'oldyoung': False,
                            'agebl': 80},
                   'over40':{
                            'pce_cohort': False,
                            'pce_invalid_vars': True,
                            'cvd_bl': True,
                            'antilpd': True,
                            'oldyoung': False,
                            'agebl': 40}
#                   'pce_statin_missing':{
#                             'pce_cohort': False,
#                             'pce_invalid_vars': False,
#                             'cvd_bl': True,
#                             'antilpd': False,
#                             'oldyoung': True},
#                    'pce_cvd_missing':{
#                             'pce_cohort': False,
#                             'pce_invalid_vars': False,
#                             'cvd_bl': False,
#                             'antilpd': True,
#                             'oldyoung': True},
#                    'cvd_missing':{
#                             'pce_cohort': True,
#                             'pce_invalid_vars': False,
#                             'cvd_bl': False,
#                             'antilpd': True,
#                             'oldyoung': True},
#                    'oldyoung_missing':{
#                             'pce_cohort': True,
#                             'pce_invalid_vars': False,
#                             'cvd_bl': True,
#                             'antilpd': True,
#                             'oldyoung': False},
#                    'over80':{
#                             'pce_cohort': True,
#                             'pce_invalid_vars': True,
#                             'cvd_bl': True,
#                             'antilpd': True,
#                             'oldyoung': False,
#                             'agebl': 80}
                  }
    test_o_missing = {
                  'pce_missing_hispanic':{
                            'pce_cohort': False,
                            'pce_invalid_vars': False,
                            'race': ['Hispanic'],
                            'cvd_bl': True,
                            'antilpd': True,
                            'oldyoung': True},
                  'pce_missing_asian':{
                            'pce_cohort': False,
                            'pce_invalid_vars': False,
                            'race': ['Asian'],
                            'cvd_bl': True,
                            'antilpd': True,
                            'oldyoung': True},
                  'pce_missing':{
                            'pce_cohort': False,
                            'pce_invalid_vars': False,
                            'cvd_bl': True,
                            'antilpd': True,
                            'oldyoung': True},
                  'pce_missing_oldyoung':{
                            'pce_cohort': False,
                            'pce_invalid_vars': False,
                            'cvd_bl': True,
                            'antilpd': True,
                            'oldyoung': False},
                   'over80_missing':{
                            'pce_cohort': True,
                            'pce_invalid_vars': False,
                            'cvd_bl': True,
                            'antilpd': True,
                            'oldyoung': False,
                            'agebl': 80},
                   'over40_missing':{
                            'pce_cohort': False,
                            'pce_invalid_vars': False,
                            'cvd_bl': True,
                            'antilpd': True,
                            'oldyoung': False,
                            'agebl': 40}
    }
    if imputer is not None:
        test_others.update(test_o_missing)
    for test_res_dir in test_others.keys():
        test_on_new_cohort(RESULT_DIR + '/' + test_res_dir, expt, alldata, 
                           test_others[test_res_dir], 
                           test_ind_col, models, 
                           ascvd_est)
    
    
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
    
def plot_ROCs(RESULT_DIR,  
                       to_exclude,
                       test_ind_col, models, ascvd_est, label,
                      test_models):
    pce_train_est2, pce_test_est2 = split_cohort(ascvd_est, to_exclude, test_ind_col, drop = 'all')

    expt = Experiment(datafile = None, 
                      result_dir = RESULT_DIR, 
                      label = label)
    pce_train_est2, pce_test_est2 = split_cohort(ascvd_est, to_exclude, test_ind_col, drop = 'all')
    expt.save_and_plot_results(models, 
                               cv = 5, pce_file = pce_train_est2, test = False,
                         test_pce_file = pce_test_est2, 
                         train = True)
    expt.save_and_plot_test_results(test_models, 
                               cv = 5, pce_file = pce_train_est2, 
                         test_pce_file = pce_test_est2)   
#%%    
if __name__ == '__main__':
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
           'logreg'
             ,
             'gbm'
             ,
             'rf'
             ,
             'lasso2'
#             ,
#             'elnet'
             ,
             'xgb'
              ]
    test_models = ['lasso']
    
    expts = {
#         'test_nm_0911': {'pce_cohort': False,
#                                 'pce_invalid_vars': True,
#                                 'cvd_bl': True,
#                                 'antilpd': True,
#                                 'oldyoung': True} 
#     #                             'agebl': 80}
                  'pce_nhwblack_missing':{
                            'pce_cohort': False,
                            'pce_invalid_vars': False,
                            'race': ['Non-Hispanic_white', 'African_American'],
                            'cvd_bl': True,
                            'antilpd': True,
                            'oldyoung': True}}
    
    datafile = 'pcevars.csv'
    alldata = pd.read_csv('../Data/cohort/' + datafile)
#     alldata = alldata.iloc[1:5000,:]
    for res_dir in expts.keys():
        
#        if expts[res_dir]['pce_invalid_vars'] == True:
#            models2 = models + ['PCE']
#            test_models2 = test_models + ['PCE']
#        else:
#            models2 = models 
#            test_models2 = test_models 
#            
#        plot_ROCs(RESULT_DIR = '../Results/allvars_' + res_dir + '_0913', 
#                  to_exclude = expts[res_dir], 
#                  test_ind_col = test_ind_col, models = models2, ascvd_est = ascvd_est,
#                  label = label, test_models = test_models2)
#        print(res_dir)
#            
        
        
        
        imputer = 'iterative'
        add_missing_flags = True
        if expts[res_dir]['pce_invalid_vars']:
            imputer = 'simple'
            add_missing_flags = False
        train_val_test(RESULT_DIR = '../Results/pcevars_' + res_dir + '_0925', alldata = alldata, 
                       to_exclude = expts[res_dir],
                       test_ind_col = test_ind_col, models = models, ascvd_est = ascvd_est, label = label,
                      imputer = imputer, add_missing_flags = add_missing_flags)
        break
    

    datafile = 'morevars.csv'
    alldata = pd.read_csv('../Data/cohort/' + datafile)
    for res_dir in expts.keys():
       
        imputer = 'iterative'
        add_missing_flags = True
        if expts[res_dir]['pce_invalid_vars']:
            imputer = 'simple'
            add_missing_flags = False
        train_val_test(RESULT_DIR = '../Results/morevars_' + res_dir + '_0925', alldata = alldata, 
                       to_exclude = expts[res_dir],
                       test_ind_col = test_ind_col, models = models, ascvd_est = ascvd_est, label = label,
                      imputer = imputer, add_missing_flags = add_missing_flags)

    datafile = 'allvars.csv'
    alldata = pd.read_csv('../Data/cohort/' + datafile)
    for res_dir in expts.keys():
       
        imputer = 'iterative'
        add_missing_flags = True
        if expts[res_dir]['pce_invalid_vars']:
            imputer = 'simple'
            add_missing_flags = False
        train_val_test(RESULT_DIR = '../Results/allvars_' + res_dir + '_0925', alldata = alldata, 
                       to_exclude = expts[res_dir],
                       test_ind_col = test_ind_col, models = models, ascvd_est = ascvd_est, label = label,
                      imputer = imputer, add_missing_flags = add_missing_flags)
       
#    train_val_test(RESULT_DIR = '../Results/test_new2', alldata = alldata, 
#                   to_exclude = {'pce_cohort': False,
#                            'pce_invalid_vars': False,
#                            'cvd_bl': True,
#                            'antilpd': True,
#                            'oldyoung': True} 
##                             'agebl': 80}
#                   ,
#                   test_ind_col = test_ind_col, models = models, ascvd_est = ascvd_est, label = label)
    
    