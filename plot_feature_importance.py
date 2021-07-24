# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 12:59:21 2020

@author: trishaj
"""
#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%%
num_imp = 26

# prediction_point =  "day_of_surg"

# if prediction_point == "day_of_surg":
#     dataset_type = '100_pct/'
#     results_path = 'S:/group/tanders0/Results/' + dataset_type + 'test_0308_gbm/' 
#     results_path_reduced = 'S:/group/tanders0/Results/' + dataset_type + 'test_0326_gbm/'
#     variable_renaming = pd.read_csv('S:/group/tanders0/Results/Top_variable_rename_mapping.csv')

# else:
#     dataset_type = '100_pct_14d/' 
#     results_path = 'S:/group/tanders0/Results/' + dataset_type + 'test_0308_gbm/' 
#     results_path_reduced = 'S:/group/tanders0/Results/' + dataset_type + 'test_0328_gbm/'
#     variable_renaming = pd.read_csv('S:/group/tanders0/Results/Top_variable_rename_mapping_14d.csv')


results_path = '../../heart_disease/secondary_prevention/Results/allvar_ascvd_all_secondary_pts_051221/'
# results_path = '../../heart_disease/secondary_prevention/Results/allvar_ascvd_cad_cvd_pad_pts_051221/'
results_path_reduced = results_path
variable_renaming = pd.read_csv('../../heart_disease/variables_with_names.csv')


best_model = 'xgb'
#Get the variable importances for both decision points
var_imp_df = pd.read_csv(results_path + best_model + '_feature_importances.csv')
var_imp_df = var_imp_df.merge(variable_renaming, on='feature', how='left')
#%%
var_imp_df = var_imp_df[var_imp_df['feature'] != 'ascvd5yest']
var_imp_df.drop(columns=['feature'], inplace = True)
var_imp_df.rename(columns={"variable_nm": "feature"}, inplace = True)
features_all = var_imp_df['feature']
features_top = features_all[:num_imp]

var_imp_df_red = pd.read_csv(results_path_reduced + best_model + '_feature_importances.csv')
var_imp_df_red = var_imp_df_red.merge(variable_renaming, on='feature', how='left')
var_imp_df_red.drop(columns=['feature'], inplace = True)
var_imp_df_red.rename(columns={"variable_nm": "feature"}, inplace = True) # feature_renamed

features_red = var_imp_df_red['feature']
features_top_red = features_red[:num_imp]

#%%
#Get the union of the top num_imp variables
set_features_top = set(features_top)
set_features_top_red = set(features_top_red)
top_features_union = list(set_features_top.union(set_features_top_red ))
top_features_df = pd.DataFrame(top_features_union, columns=['feature'])

#%%
def variable_importance_plot(feature_names, feature_importances, err=None):
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
    fig, ax = plt.subplots(figsize=(7.5, 12))
    if err is None:
        err = np.zeros(len(feature_names))
    
    feature_importances = pd.DataFrame({'feat_imps': feature_importances,
                                        "err": err}, index=feature_names)
    importances_df = feature_importances.sort_values(by='feat_imps', axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last')
#     ax.set_axis_bgcolor('#fafafa')
    ax.barh(importances_df.index,
            importances_df['feat_imps'],
            xerr=importances_df['err'], 
            alpha = 0.9, 
            edgecolor = "black", 
            zorder=3, 
            color='lightblue'
             )
#              align="center",
#              color = '#875FDB')
#     plt.yticks(index,
#                feature_space)
    ax.set_ylabel('Feature')
    ax.grid(True, axis='x')

    fig.subplots_adjust(left=0.3)
    fig.tight_layout()
    return ax, fig
#%%
#Define new dataframes, only with rows corresponding to the top features
#do a left join to get all the rows you want


var_imp_df_top = top_features_df.merge(var_imp_df, on='feature', how='left').fillna(0)
var_imp_df_red_top = top_features_df.merge(var_imp_df_red, on='feature', how='left').fillna(0)

vdf2 = var_imp_df[~var_imp_df['feature'].isna()].set_index('feature')
ax, fig = variable_importance_plot(features_top,
                                  vdf2.importance_x[features_top])
ax.set_xlabel('Feature Importance')
ax.set_title('Feature Importance for XGBoost Model')
fig
plt.savefig(results_path + 'feat_importance.svg', bbox_inches='tight')
#%%

#use this to flip which one is sorted
#var_imp_df_red_top.sort_values(by=['importance'], inplace = True)
#var_imp_df_top.set_index('feature', inplace = True)
#var_imp_df_top = var_imp_df_top.reindex(index=var_imp_df_red_top['feature'])
#var_imp_df_top.reset_index(inplace = True)



var_imp_df_top.sort_values(by=['importance'], inplace = True)
var_imp_df_red_top.set_index('feature', inplace = True)
var_imp_df_red_top = var_imp_df_red_top.reindex(index=var_imp_df_top['feature'])
var_imp_df_red_top.reset_index(inplace = True)



var_imp_df_top['relative importance'] = var_imp_df_top['importance']/np.max(var_imp_df_top['importance']) * 100
var_imp_df_red_top['relative importance'] = var_imp_df_red_top['importance']/np.max(var_imp_df_red_top['importance']) * 100



top_features = var_imp_df_top['feature']
top_features_red = var_imp_df_red_top['feature']
top_var_imp = var_imp_df_top['importance']
top_var_imp_red = var_imp_df_red_top['importance']

#%%
# Create horizontal bars
y_pos = np.arange(len(top_features_union))

fig, ax = plt.subplots(figsize=(10,8))
ax.xaxis.grid(True, zorder=0)
width = 0.40

offset_fix = np.zeros(len(top_features_union))
offset_fix[top_var_imp_red == 0]= -width/2
#top_var_imp/np.max(top_var_imp) * 100 top_var_imp_red/np.max(top_var_imp_red) * 100 , width

plt.barh(y_pos+width/2 + offset_fix, var_imp_df_top['relative importance']  , width, alpha = 0.5, edgecolor = "black", zorder=3, color='tab:grey')
plt.barh(y_pos-width/2, var_imp_df_red_top['relative importance'] ,width, alpha = 0.5, edgecolor = "black", zorder=3, color='tab:blue')
   
# Create names on the y-axis
plt.yticks(y_pos, top_features)

plt.xlabel('Relative Importance (%)')
plt.xlim(0, 100)
plt.legend([ 'All variables','Bedside variables'])
plt.tight_layout()

if prediction_point == "day_of_surg":
    plt.title('Top 20 Important Variables Selected by GBM \n for day-of-surgery prediction')
    plt.savefig('S:/group/tanders0/Results/' + 'top_20_importances_pct.png')
else:
    plt.title('Top 20 Important Variables Selected by GBM \n for 14-days-after-surgery prediction')
    plt.savefig('S:/group/tanders0/Results/' + 'top_20_importances_pct_14d.png')

    