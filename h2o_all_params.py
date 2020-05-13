import os
import h2o
from h2o.automl import H2OAutoML

h2o.init()
result_dir = '../Results/h2otest_0724/'
if not os.path.isdir(result_dir): os.mkdir(result_dir)
# Import a sample binary outcome train/test set into H2O
train = h2o.import_file('../Data/cohort/train_somefeats_pcepts.csv')
test = h2o.import_file('../Data/cohort/test_somefeats_pcepts.csv')

# Identify predictors and response
x = train.columns
y = "ascvdany5y"
x.remove(y)

# For binary classification, response should be a factor
train[y] = train[y].asfactor()
test[y] = test[y].asfactor()

# Run AutoML for 20 base models (limited to 1 hour max runtime by default)
aml = H2OAutoML(max_runtime_secs=24*60*60, seed=1 
#                exclude_algos = ['DeepLearning'],
#           keep_cross_validation_predictions = True, 
#          keep_cross_validation_fold_assignment = True
               )
aml.train(x=x, y=y, training_frame=train)

# View the AutoML Leaderboard
lb = aml.leaderboard
lb.head(rows=lb.nrows)

h2o.export_file(lb, result_dir + 'lb.csv')
# import pdb; pdb.set_trace()
# preds = aml.leader.predict(test)