#%%
#import importlib.util
#spec = importlib.util.spec_from_file_location("medical_ML.py", 'C:\\Users\\Andrew\\Documents\\Stanford\\medical\\medicalML_git\\medicalML')
#foo = importlib.util.module_from_spec(spec)
#spec.loader.exec_module(foo)

from medical_ML import Experiment, split_cohort#, test
import pandas as pd
import os
#%%
def plot_ROCs_with_baseline(RESULT_DIR,
                            models, label,
                      test_models, 
                            baseline_probs = None,
                            test_baseline_probs = None,
                            baseline_str = 'baseline',
                            baseline_type = 'probability',
                           title = 'ROC for discharge prediction on test data', 
             pr_title = "Precision-recall curve on test data"):
#    pce_train_est2, pce_test_est2 = split_cohort(ascvd_est, to_exclude, test_ind_col, drop = 'all')

    expt = Experiment(datafile = None, 
                      result_dir = RESULT_DIR, 
                      label = label)
#     pce_train_est2, pce_test_est2 = split_cohort(ascvd_est, to_exclude, test_ind_col, drop = 'all')
#     expt.save_and_plot_results(models, 
#                                cv = 5, pce_file = pce_train_est2, test = False,
#                          test_pce_file = pce_test_est2, 
#                          train = True,
#                          title = 'ROC for full patient cohort on validation data',
#                          tr_title = 'ROC for full patient cohort on training data')
    expt.save_and_plot_test_results(test_models, 
                               cv = 5, 
                                    baseline_str = baseline_str,
                                    baseline_type = baseline_type,
                                 test_baseline_prob_file = test_baseline_probs,   
                         title = title,  
                                   pr_title = pr_title)
def plot_val_ROCs(RESULT_DIR,
                            models, label,
                            baseline_probs = None,
                            baseline_str = 'baseline',
                            baseline_type = 'probability',
                           title = 'ROC for discharge prediction on test data'
             ):
#    pce_train_est2, pce_test_est2 = split_cohort(ascvd_est, to_exclude, test_ind_col, drop = 'all')

    expt = Experiment(datafile = None, 
                      result_dir = RESULT_DIR, 
                      label = label)
#     pce_train_est2, pce_test_est2 = split_cohort(ascvd_est, to_exclude, test_ind_col, drop = 'all')
#     expt.save_and_plot_results(models, 
#                                cv = 5, pce_file = pce_train_est2, test = False,
#                          test_pce_file = pce_test_est2, 
#                          train = True,
#                          title = 'ROC for full patient cohort on validation data',
#                          tr_title = 'ROC for full patient cohort on training data')
    expt.save_and_plot_results(models, 
                               cv = 5, 
                               train = False,
                               test = False,
                                    baseline_str = baseline_str,
#                                    baseline_type = baseline_type,
                                 baseline_prob_file = baseline_probs,   
                         title = title)
    
def plot_ROCs(RESULT_DIR,  
                       models, label,
                      test_models, 
             title = 'ROC for discharge prediction on test data', 
             pr_title = "Precision-recall curve on test data"):
#    pce_train_est2, pce_test_est2 = split_cohort(ascvd_est, to_exclude, test_ind_col, drop = 'all')

    expt = Experiment(datafile = None, 
                      result_dir = RESULT_DIR, 
                      label = label)
#     expt.save_and_plot_results(models, 
#                                cv = 5, test = False,
#                          train = True,
#                          title = 'ROC for full patient cohort on validation data',
#                          tr_title = 'ROC for full patient cohort on training data')
    expt.save_and_plot_test_results(test_models, 
                               cv = 5, 
                         title = title,  
                                   pr_title = pr_title)
    
    
def test_on_new_cohort(R2, expt, test_data, to_exclude = None, test_ind_col = None, models = None, baseline_str = None, 
             test_baseline_prob_file = None, 
             title = 'ROC for discharge prediction on test data'):
    if not os.path.isdir(R2): os.mkdir(R2)
    # _, test_data = split_cohort(alldata, to_exclude, test_ind_col, drop = 'all')
    if test_ind_col is not None:
        expt.test_data = test_data[test_data[test_ind_col] == 1]
    else:
        expt.test_data = test_data
    expt.predict_on_test(models, test_file = None,
                        out_dir = R2)
    # to_exclude['pce_invalid_vars'] = True
    # ascvd_train_est2, ascvd_test_est2 = split_cohort(ascvd_est, to_exclude, test_ind_col, drop = 'all')
    expt.save_and_plot_test_results(models, 
                               cv = 5, 
                              out_dir = R2, 
                                    test_baseline_prob_file = test_baseline_prob_file,
                                   baseline_str = baseline_str,
                                   title = title)
    
def train_val(RESULT_DIR, alldata, models, label = 'Label', 
              cv = 5, 
              score_name = "AUC", 
              to_exclude = None, 
              test_ind_col = None,   oversample_rate = 1,
                  imputer = 'iterative', add_missing_flags = True, 
             baseline_str = None, 
             baseline_prob_file = None, 
             title = 'ROC for discharge prediction on validation data'):
    print('\n\n' + 'STARTING EXPERIMENT FOR ' + RESULT_DIR + '\n\n')
    expt = Experiment(alldata, label = label, 
                      to_exclude = to_exclude, 
                      test_ind_col = test_ind_col, drop = 'all', 
                      result_dir = RESULT_DIR)
    expt.predict_models_from_groups(0, models, cv=cv, score_name=score_name, mode='classification',
                                                    oversample_rate = oversample_rate, 
                                                   imputer = imputer, add_missing_flags = add_missing_flags)
#     expt.save_and_plot_results(models, 
#                                cv = cv, test = False, 
#                                baseline_prob_file = baseline_prob_file,
#                                baseline_str = baseline_str,
#                                title = title)
    return(expt)





models = ['baseline']
t_models = [
                  'logreg'
#                    ,
#                    'lasso2'
                    ,
                    'lasso'
                    ,
                    'rf'
                    ,
                    'gbm'
        #             , 
        #          'svm'
                   ,
                  'xgb'
#                    ,
#             'baseline'
                  ]
baseline_str = ['Simple Imputation/PCE Variables', 'Simple Imputation/All Variables',
                'Iterative Imputation/PCE Variables', 'Iterative Imputation/All Variables']

RESULT_DIR = '../../heart_disease/Results/prior_cvd_pts/allvar_ascvd_cvd_withandwithoutstatin_allage_1101'
cht = 'secondary prevention patients'
#RESULT_DIR = '../../heart_disease/Results/allvars_pce_pts_0925'
#cht = 'PCE-eligible cohort'
t_models.append('baseline')
label = 'ascvdany5y'
#baseline_type = ['probability'] *4
#
#BP_DIR = '../../heart_disease/Results/pce_imputation'
#baseline_prob_files = ['simple_pcevars_PCE_estimates_clipped_vars2.csv',
#                       'simple_allvars_PCE_estimates_clipped_vars2.csv',
#                       'iterative_pcevars_PCE_estimates_clipped_vars2.csv',
#                       'iterative_allvars_PCE_estimates_clipped_vars2.csv']
#
#b_probs = []
#for bpf in baseline_prob_files:
#    bp = pd.read_csv(os.path.join(BP_DIR, bpf))
#    bp.columns = ['y', 'est_prob']
##    bp = bp[~bp.y.isna()]
#    b_probs.append(bp)
#    print(bpf)


test_ind_col  = 'test_ind'
label = 'ascvdany5y'
to_exclude = {
                            'pce_cohort': True,
                            'pce_invalid_vars': True,
                            'cvd_bl': False,
                            'antilpd': False,
                            'oldyoung': False}
datafile = 'ascvd_est.csv'
ascvd_est = pd.read_csv('../../heart_disease/Data/cohort/' + datafile)
train_est2, test_est2 = split_cohort(ascvd_est, to_exclude, test_ind_col, drop = 'none')
#%%
test_est2 = test_est2[test_est2.cvd_bl == True]
#%%
b_probs = test_est2[['ascvdany5y', 'ascvd5yest']]
b_probs.columns = ['y', 'est_prob']
b_probs_tr = train_est2[['ascvdany5y', 'ascvd5yest']]
b_probs_tr.columns = ['y', 'est_prob']
baseline_type = 'probability'
baseline_str = 'PCE'
plot_ROCs_with_baseline(RESULT_DIR + '', 
                 models = models, 
         label = label, 
                        test_baseline_probs = b_probs,
                        baseline_str = baseline_str,
                        baseline_type = baseline_type,
#         title = f"ROC on held-out test data \nfor {cht}",
         title = f"Machine learning and PCE performance for secondary ASCVD risk prediction:\nReceiver operating characteristic curve",
#          pr_title = "Precision-recall curve for patients with in-hospital predictions ({})".format(run_type.upper()),
         test_models = t_models)
#%%
plot_val_ROCs(RESULT_DIR + '', 
                 models = t_models, 
         label = label, 
                        baseline_probs = b_probs_tr,
                        baseline_str = baseline_str,
#                        baseline_type = baseline_type,
         title = f"ROC on cross-validation data for {cht}")
    #%%
#
#models = [
#                  'logreg'
#                    ,
#                    'lasso2'
#                    ,
#                    'rf'
#                    ,
#                    'gbm'
#        #             , 
#        #          'svm'
#        #           ,
#        #          'xgb'
#                  ] 
#label = 'Label'
#run_type = '3pm'
#RESULT_DIR = "../Results/full_run_defaults3_{}".format(run_type)
#
## imputer = 'simple'
## add_missing_flags = False
## alldata = pd.read_csv("../Data/train_dat_{}.csv".format(run_type))
#
## expt = train_val(RESULT_DIR = RESULT_DIR, alldata = alldata, 
##           models = ['dummy'],
##                to_exclude = None,
##                test_ind_col = None,  label = label,
##               imputer = imputer, add_missing_flags = add_missing_flags)
#
#test_file = 'test1'
#test_data = pd.read_csv("../Data/{}_dat_{}.csv".format(test_file, run_type))
## test_on_new_cohort(RESULT_DIR + '/test1', expt, test_data = test_data, 
##           models = models,
##                to_exclude = None,
##                test_ind_col = None,  
##               title = 'ROC on data from held-out facility')
## plot_ROCs(RESULT_DIR + '/test1', 
##                  models = models, 
##          label = label, 
##          title = "ROC for discharge prediction at {}".format(run_type.upper()),
##           pr_title = "Precision-recall curve for {} prediction".format(run_type.upper()),
##          test_models = [
##                   'logreg'
##                     ,
##                     'lasso2'
##                     ,
##                     'rf'
##                     ,
##                     'gbm'
##         #             , 
##         #          'svm'
##         #           ,
##         #          'xgb'
##                   ])
#
#
## excluding all with missing predictions
## test_metadata = pd.read_csv("../Data/{}_dat_{}times.csv".format(test_file, run_type))
## test_data = test_data[~test_metadata.hours_to_disch_exp.isna()]
## test_metadata = test_metadata[~test_metadata.hours_to_disch_exp.isna()]
## newcols = test_metadata.columns.tolist()
## def ff(x):
##     if x == 'Label':
##         return("y")
##     if x == "exp_disch_label":
##         return("est_prob")
##     return x
## test_metadata.columns = [ff(x) for x in newcols]
## if run_type == '3pm':
##     test_metadata.est_prob = (test_metadata.hours_to_disch_exp < 26.2)
## if run_type == '3am':
##     test_metadata.est_prob = (test_metadata.hours_to_disch_exp < 14.2)
#
#
## # test_on_new_cohort(RESULT_DIR + '/test1_withpred', expt, test_data = test_data, 
## #           models = models,
## #                to_exclude = None,
## #                test_ind_col = None,  
## #               title = 'ROC on data from held-out facility')
#
## plot_ROCs_with_baseline(RESULT_DIR + '/test1_withpred', 
##                  models = models, 
##          label = label, 
##                         test_baseline_probs = test_metadata,
##                         baseline_str = "Bedside prediction",
##                         baseline_type = 'binary',
##          title = "ROC for patients with in-hospital predictions ({})".format(run_type.upper()),
##           pr_title = "Precision-recall curve for patients with in-hospital predictions ({})".format(run_type.upper()),
##          test_models = [
##                   'logreg'
##                     ,
##                     'lasso2'
##                     ,
##                     'rf'
##                     ,
##                     'gbm',
##              'baseline'
##         #             , 
##         #          'svm'
##         #           ,
##         #          'xgb'
##                   ])
#
#
## treating all with missing predictions as "will be discharged"
#test_metadata = pd.read_csv("../Data/{}_dat_{}times.csv".format(test_file, run_type))
## test_data = test_data[~test_metadata.hours_to_disch_exp.isna()]
## test_metadata = test_metadata[~test_metadata.hours_to_disch_exp.isna()]
#test_metadata[test_metadata.hours_to_disch_exp.isna()] = 0
#newcols = test_metadata.columns.tolist()
#def ff(x):
#    if x == 'Label':
#        return("y")
#    if x == "exp_disch_label":
#        return("est_prob")
#    return x
#test_metadata.columns = [ff(x) for x in newcols]
#if run_type == '3pm':
#    test_metadata.est_prob = (test_metadata.hours_to_disch_exp < 26.2)
#if run_type == '3am':
#    test_metadata.est_prob = (test_metadata.hours_to_disch_exp < 14.2)
#
#
## test_on_new_cohort(RESULT_DIR + '/test1_withpred', expt, test_data = test_data, 
##           models = models,
##                to_exclude = None,
##                test_ind_col = None,  
##               title = 'ROC on data from held-out facility')
#
#plot_ROCs_with_baseline(RESULT_DIR + '/test1', 
#                 models = models, 
#         label = label, 
#                        test_baseline_probs = test_metadata,
#                        baseline_str = "Bedside prediction (missing = discharge)",
#                        baseline_type = 'binary',
#         title = "ROC for patients with in-hospital predictions ({})".format(run_type.upper()),
#          pr_title = "Precision-recall curve for patients with in-hospital predictions ({})".format(run_type.upper()),
#         test_models = [
#                  'logreg'
#                    ,
#                    'lasso2'
#                    ,
#                    'rf'
#                    ,
#                    'gbm',
#             'baseline'
#        #             , 
#        #          'svm'
#        #           ,
#        #          'xgb'
#                  ])
#
#
## treating all with missing predictions as "wont be discharged"
#test_metadata = pd.read_csv("../Data/{}_dat_{}times.csv".format(test_file, run_type))
## test_data = test_data[~test_metadata.hours_to_disch_exp.isna()]
## test_metadata = test_metadata[~test_metadata.hours_to_disch_exp.isna()]
#test_metadata[test_metadata.hours_to_disch_exp.isna()] = 1000
#newcols = test_metadata.columns.tolist()
#def ff(x):
#    if x == 'Label':
#        return("y")
#    if x == "exp_disch_label":
#        return("est_prob")
#    return x
#test_metadata.columns = [ff(x) for x in newcols]
#if run_type == '3pm':
#    test_metadata.est_prob = (test_metadata.hours_to_disch_exp < 26.2)
#if run_type == '3am':
#    test_metadata.est_prob = (test_metadata.hours_to_disch_exp < 14.2)
#
#
## test_on_new_cohort(RESULT_DIR + '/test1_withpred', expt, test_data = test_data, 
##           models = models,
##                to_exclude = None,
##                test_ind_col = None,  
##               title = 'ROC on data from held-out facility')
#
#plot_ROCs_with_baseline(RESULT_DIR + '/test1_withpredwontbedischarged', 
#                 models = models, 
#         label = label, 
#                        test_baseline_probs = test_metadata,
#                        baseline_str = "Bedside prediction (missing = not disch)",
#                        baseline_type = 'binary',
#         title = "ROC for patients with in-hospital predictions ({})".format(run_type.upper()),
#          pr_title = "Precision-recall curve for patients with in-hospital predictions ({})".format(run_type.upper()),
#         test_models = [
#                  'logreg'
#                    ,
#                    'lasso2'
#                    ,
#                    'rf'
#                    ,
#                    'gbm',
#             'baseline'
#        #             , 
#        #          'svm'
#        #           ,
#        #          'xgb'
#                  ])