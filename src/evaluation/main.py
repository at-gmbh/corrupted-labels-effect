# imports
import os
import random
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import util
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=DeprecationWarning)

#------------------------------------------------------------------------------#
print('Loading data...')
# define path and file filter
regr_results_log_path = './logs/regr_results/'
class_report_log_path = './logs/class_report'
aggr_class_report_log_path = './data/'
filter = 'class_report_19epoch.json'

# aggregate class reports
# util.aggr_class_reports(
#     class_report_log_path,
#     filter,
#     aggr_class_report_log_path
# )

# load classification results table
df = util.load_aggr_class_reports(aggr_class_report_log_path)

#------------------------------------------------------------------------------#
print('Setting regression data properties...')
# --> TODO: set regression properties below <--
regressor = 'accuracy' # regressor to use
multi_regression = False # TODO: implement multi-regression
cnn = 'resnet' # model to filter for ('resnet', 'basic'), False for all
classes = 14 # classification task to filter for (4 or 14), False for all
use_delta = True # use regressor delta to 0 ratio model for regression

group_by = {
    'model': False,
    'classes': False,
    'ratio': False
} # vars to group data by
zscore_threshold = False # threshold for removing outlier, False for no filtering

use_grouping = any(group_by.values())

# check properties
if cnn not in ['resnet', 'basic', False] or classes not in [4, 14, False]:
    raise ValueError('Invalid model or classes filter')

#------------------------------------------------------------------------------#
print('Initializing logging...')

# define regression properties for logging
regr_results = util.regr_results_logger(
    group_by, regressor, cnn, classes, use_delta, zscore_threshold
)

#------------------------------------------------------------------------------#
print('Preprocessing data...')
# preprocess dataframe
df_regression = df.query(f'metric == "{regressor}"')
df_regression = df_regression.drop(['run'], axis=1)
df_regression = df_regression.rename(columns={'value': regressor})
df_regression = util.calc_delta(df, df_regression, regressor)

# filter data for set properties
df_regression = util.filter_data(
    use_grouping, group_by, df_regression, cnn, classes
)

# filter outliers if properties are set
df_regression = util.filter_outliers(
    df_regression, regressor, use_grouping, zscore_threshold
)

df_regression.reset_index(inplace=True, drop=True)
print(f'\tRegression df.shape: {df_regression.shape}')
regr_results['properties']['n'] = df_regression.shape[0]

# view pair and boxplot of regression data
sns.pairplot(
    df_regression[[regressor, f'{regressor}_delta', 'ratio']],
    diag_kind='kde'
).fig.savefig('./assets/regr_pairplot.png')

plt.clf()

sns.boxplot(
    x='ratio', y=regressor, data=df_regression
).get_figure().savefig('./assets/regr_boxplot.png')
print('\tPlots saved to ./assets/')

#------------------------------------------------------------------------------#
print('Setting up training...')
# get independent and dependent variables
if use_delta:
    X_df = df_regression[[f'{regressor}_delta']]
    print(f'\tUsing {regressor} delta')
else:
    X_df = df_regression[[regressor]]
y_df = df_regression['ratio']

# calculate correlation (pearsonr and p-values)
for column in X_df.columns:
    r, p = stats.pearsonr(X_df[column], y_df)
    print(f'\t{round(r, 4)} pearsonr, {round(p, 10)} p-value, Variable: {column}')

# log correlation
regr_results['properties']['pearsonr'] = r
regr_results['properties']['p_value'] = p

# log random state for train/test split reproducibility
random_state = random.randint(0, 1000)
regr_results['properties']['split_random_state'] = random_state

# split data - random state to reproduce split in initial model selection
test_split = 0.2 if len(df_regression) > 100 else 0.1

# stratify only if test data can include all 10 ratios
if len(df_regression) * test_split > 9:
    X_train, X_test, y_train, y_test = train_test_split(
                                                X_df
                                                , y_df
                                                , test_size=test_split
                                                , random_state=random_state
                                                , stratify=y_df
                                            )
    print('\tStratified train/test split')
else:
    X_train, X_test, y_train, y_test = train_test_split(
                                                X_df
                                                , y_df
                                                , test_size=test_split
                                                , random_state=random_state
                                            )

# scale X data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(
    f'\tX_train.shape {X_train.shape} - y_train.shape {y_train.shape}\
    \n\tX_test.shape {X_test.shape} - y_test.shape {y_test.shape}'
)

#------------------------------------------------------------------------------#
print('Training estimators...')
regr_results = util.train_estimators(regr_results, X_train, y_train)

# get estimator with best score
best_score = -999
for k, v in regr_results['regr_results'].items():
    if v['neg_rmse'] > best_score:
        best_score = v['neg_rmse']
        best_estimator = k
print(f'\t--> Best estimator: {best_estimator} <--')

#------------------------------------------------------------------------------#
print('Searching for best hyperparameters...')
regr_results = util.hyperparameter_tuning(
    best_estimator, X_train, y_train, X_test, y_test, regr_results, 5
)


#------------------------------------------------------------------------------#
print('Logging regression results...')

row = [[
    regr_results['properties']['regressor'],
    regr_results['properties']['regressand'],
    regr_results['properties']['model'],
    regr_results['properties']['classification_task'],
    regr_results['properties']['n'],
    regr_results['properties']['split_random_state'],
    regr_results['properties']['group_by'],
    regr_results['properties']['delta'],
    regr_results['properties']['zscore_threshold'],
    regr_results['properties']['pearsonr'],
    regr_results['properties']['p_value'],
    regr_results['best_estimator']['estimator'],
    regr_results['best_estimator']['neg_rmse_train'],
    regr_results['best_estimator']['neg_rmse_test'],
    regr_results['best_estimator']['r2_train'],
    regr_results['best_estimator']['r2_test'],
    regr_results['best_estimator']['best_params'],
]]

df_temp = pd.DataFrame(
    data = row,
    columns = [
        'regressor', 'regressand', 'model', 'classification_task',
        'n', 'split_random_state', 'group_by', 'delta', 'zscore_threshold',
        'pearsonr', 'p_value', 'estimator', 'neg_rmse_train', 'neg_rmse_test',
        'r2_train', 'r2_test', 'best_params'
    ]
)

# add results to regression results dataframe and save as csv
result_file = regr_results_log_path + "regr_results.csv"
if os.path.exists(result_file):
    df_results = pd.read_csv(result_file)
    df_results = pd.concat([df_results, df_temp])
    df_results.to_csv(result_file, index=False)
else:
    df_temp.to_csv(result_file, index=False)
