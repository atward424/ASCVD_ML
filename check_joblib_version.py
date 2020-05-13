import joblib
print("Joblib version:")
print(joblib.__version__)

import sklearn
print("Sklearn version:")
print(sklearn.__version__)

result_dir = '../Results/allvars_pce_pts_0925/'
best_model = 'gbm'
#result_dir = '../Results/allvars_oldyoung_missing_0913/'
#best_model = 'gbm'
model = joblib.load(result_dir + best_model + '_best_model.joblib')